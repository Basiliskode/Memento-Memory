"""Hybrid retriever for memento — combines FTS5, HRR vectors, Jaccard similarity,
and optional embedding vector search with RRF fusion.

Search strategy:
1. FTS5 candidate fetch (limit × 2 for scoring headroom)
2. HRR phase vector similarity (if numpy available)
3. Jaccard n-gram overlap for lexical re-ranking
4. Optional embedding vector search (if ``compute_embedding`` provided)
5. RRF (Reciprocal Rank Fusion) of FTS5 and vector streams
"""

import datetime as dt
import logging
import re
from typing import Any, Optional, Callable

from . import hrr
from .store import EtchStore

logger = logging.getLogger(__name__)

# Default HRR weight — 0.3 keeps HRR as a tiebreaker over FTS5
_DEFAULT_HRR_WEIGHT = 0.3
_DEFAULT_FTS5_LIMIT_MULTIPLIER = 2
_DEFAULT_MATCH_THRESHOLD = 3
_DEFAULT_MAX_DEPTH = 3
_DEFAULT_FALLBACK_THRESHOLDS = [3, 3]

# Stopwords for keyword extraction in FTS5 expansion.
# Simple Python list — no external NLP dependency.
_STOPWORDS = [
    "the", "a", "is", "what", "does", "have", "any", "do", "are",
    "was", "were", "can", "will", "would", "could", "may", "this",
    "that", "with", "for", "to", "in", "on", "at", "by", "of",
    "and", "or", "not", "be", "has", "had", "it", "its", "an",
    "as", "from", "which", "who", "whom", "whose", "how", "when",
    "where", "why",
]
_STOPWORD_SET = frozenset(_STOPWORDS)


def _extract_keywords(query: str) -> list[str]:
    """Extract content-bearing keywords from a query by removing stopwords.

    Args:
        query: Raw search query.

    Returns:
        List of content keywords (stopwords removed, FTS5-sanitized,
        deduplicated, preserving original order).
    """
    cleaned = _sanitize_fts5(query)
    tokens = cleaned.split()
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in _STOPWORD_SET:
            continue
        if token_lower not in seen:
            seen.add(token_lower)
            keywords.append(token)
    return keywords


def _sanitize_fts5(query: str) -> str:
    """Sanitize a natural-language query for FTS5 MATCH syntax.

    FTS5 tokenizes by whitespace and punctuation. Special characters like
    ``?``, ``'``, ``!``, ``.`` etc. cause syntax errors or silently
    produce no matches. This strips them and returns clean search tokens.
    """
    # Remove characters that FTS5 interprets as query operators: ?, ', ", !,
    # ., ,, -, +, =, ~, `, [, ], {, }, |, ;, :, ^, *, @, #, $, %, &
    cleaned = re.sub(r"""[?!'".;:\-+=~`@#$%^&*()\[\]{}|,<>]""", " ", query)
    # Collapse whitespace
    return " ".join(cleaned.split())


class EtchRetriever:
    """Hybrid search over an EtchStore.

    Args:
        store: EtchStore instance.
        hrr_dim: Optional HRR vector dimension. When omitted, the retriever
            uses the store's effective HRR dimension.
        hrr_weight: Blend weight for HRR vs FTS5 (0.0 = FTS5 only, 1.0 = HRR only).
        reranker: Optional callback reranker(query, candidates) → ranked candidates.
        rerank_min_score: Minimum top score to skip reranker (0.0 = always rerank).
        compute_embedding: Optional callable ``encode(text: str) → list[float]``.
            When ``None``, vector search path is skipped gracefully.
    """

    def __init__(
        self,
        store: EtchStore,
        hrr_dim: Optional[int] = None,
        hrr_weight: float = _DEFAULT_HRR_WEIGHT,
        reranker: Optional[Callable] = None,
        rerank_min_score: float = 0.0,
        compute_embedding: Optional[Callable[[str], list[float]]] = None,
        recency_weight: float = 0.15,
        recency_half_life: float = 7.0,
    ):
        self._store = store
        self._hrr_dim = hrr_dim if hrr_dim is not None else store.get_effective_hrr_dim()
        self._hrr_weight = hrr_weight
        self._reranker = reranker
        self._rerank_min_score = rerank_min_score
        self._compute_embedding = compute_embedding
        self._recency_weight = recency_weight
        self._recency_half_life = recency_half_life

    def search(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
        project: str = "",
        mode: str = "",
        fallback_thresholds: Optional[list[int]] = None,
        scope: str = "canonical",
        source_harness: str = "",
        source_agent: str = "",
        source_kind: str = "",
    ) -> list[dict]:
        """Hybrid search: FTS5 + optional HRR + Jaccard + optional vector.

        Two modes:
        - ``mode=""`` (default, empty string): existing behavior — single FTS5
          query + scoring + optional vector search + RRF fusion.
        - ``mode="auto"``: smarter fallback cascade —
          1. ``search_expanded`` (FTS5 expansion with keyword breadth)
          2. If results < threshold: HRR multi-query (semantic variations)
          3. If results < threshold: embedding vector search (if configured)
          4. Results merged at each cascade level

        Default scope is ``'canonical'`` — only canonical facts are returned
        unless an explicit scope is requested.

        Args:
            query: Search text.
            limit: Max results.
            exclude_deleted: Whether to exclude soft-deleted facts.
            project: Optional project filter.
            mode: ``"auto"`` for smart fallback cascade; empty string for
                the existing single-pass hybrid search.
            fallback_thresholds: Per-level minimum results to stop cascading.
                Default: ``[3, 3]`` (stop after FTS5 if ≥3, stop after HRR if ≥3).
            scope: Scope filter (default: ``'canonical'``).
            source_harness: Optional source harness filter.
            source_agent: Optional source agent filter.
            source_kind: Optional source kind filter.

        Returns list of dicts sorted by combined relevance score (``score`` key).
        """
        if mode == "auto":
            return self._search_auto(
                query, limit, exclude_deleted, project, fallback_thresholds,
                scope=scope, source_harness=source_harness,
                source_agent=source_agent, source_kind=source_kind,
            )

        # --- Original behavior (backward compatible) ---
        fts5_stream = self._fts_candidates(query, limit * 2, exclude_deleted, project,
                                           scope=scope, source_harness=source_harness,
                                           source_agent=source_agent, source_kind=source_kind)
        if not fts5_stream:
            return []

        # Scored FTS5 candidates
        scored = self._score_candidates(query, fts5_stream)
        scored.sort(key=lambda x: x.get("_score", 0), reverse=True)
        for r in scored:
            r.pop("_hrr_vec", None)

        # Vector stream (optional)
        vector_stream: list[dict] = []
        if self._compute_embedding is not None:
            try:
                q_vec = self._compute_embedding(query)
                if q_vec:
                    import struct
                    vec_bytes = struct.pack(f"{len(q_vec)}f", *q_vec)
                    vector_stream = self._store.search_by_vector(
                        vec_bytes, limit=limit * 2, project=project,
                        scope=scope, source_harness=source_harness,
                        source_agent=source_agent, source_kind=source_kind,
                    )
            except Exception:
                logger.exception("Vector search failed, falling back to FTS5-only")

        # RRF fusion
        merged = self._rrf_merge(scored, vector_stream, limit=limit, k=60)

        # Apply reranker if configured
        if self._reranker and merged:
            try:
                top_score = merged[0].get("score", 0)
                if top_score < self._rerank_min_score or self._rerank_min_score <= 0:
                    reranked = self._reranker(query, merged)
                    if reranked:
                        return reranked
            except Exception:
                logger.exception("Reranker failed, returning RRF results")

        return merged

    def _fts_candidates(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
        project: str = "",
        scope: str = "canonical",
        source_harness: str = "",
        source_agent: str = "",
        source_kind: str = "",
    ) -> list[dict]:
        """Fetch candidates from FTS5 with headroom for re-scoring.

        Optionally filters by project, scope, and source metadata.
        """
        with self._store._lock:
            try:
                fetch_limit = limit * _DEFAULT_FTS5_LIMIT_MULTIPLIER

                sql = """SELECT f.fact_id, f.content, f.category, f.tags,
                                f.trust_score, f.hrr_vector, f.created_at, f.updated_at,
                                f.project
                         FROM facts f
                            JOIN facts_fts fts ON fts.rowid = f.fact_id
                         WHERE facts_fts MATCH ?"""
                safe_query = _sanitize_fts5(query)
                params: list = [safe_query]
                conditions: list[str] = []
                if exclude_deleted:
                    conditions.append("(f.deleted IS NULL OR f.deleted = 0)")
                if project:
                    conditions.append("f.project = ?")
                    params.append(project)
                if scope:
                    conditions.append("f.scope = ?")
                    params.append(scope)
                if source_harness:
                    conditions.append("f.source_harness = ?")
                    params.append(source_harness)
                if source_agent:
                    conditions.append("f.source_agent = ?")
                    params.append(source_agent)
                if source_kind:
                    conditions.append("f.source_kind = ?")
                    params.append(source_kind)
                if conditions:
                    sql += " AND " + " AND ".join(conditions)
                sql += " ORDER BY fts.rank LIMIT ?"
                params.append(fetch_limit)

                rows = self._store._conn.execute(sql, params).fetchall()
                results = [dict(r) for r in rows]
                # Add summary for progressive disclosure
                for r in results:
                    if "summary" not in r:
                        r["summary"] = r.get("content", "")[:200]
                # Reinforce retrieved facts (retrieval feedback loop)
                if results:
                    ids = [r["fact_id"] for r in rows]
                    placeholders = ",".join("?" for _ in ids)
                    self._store._conn.execute(
                        f"""UPDATE facts SET
                                retrieval_count = retrieval_count + 1,
                                trust_score = MIN(1.0, ROUND(trust_score + 0.01, 4))
                            WHERE fact_id IN ({placeholders})""",
                        ids,
                    )
                    self._store._conn.commit()
                return results
            except Exception:
                logger.exception("FTS5 search failed")
                return []

    def _score_candidates(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Score candidates with hybrid FTS5 + HRR + Jaccard."""
        if not candidates:
            return []

        # 1. Compute HRR query vector (if available)
        query_vec = None
        if hrr.HAS_NUMPY and self._hrr_weight > 0:
            try:
                query_vec = hrr.encode_text(query, self._hrr_dim)
            except Exception:
                logger.exception("HRR query encoding failed")

        # 2. Score each candidate
        for c in candidates:
            score = c.get("trust_score", 0.5)  # base trust

            # FTS5 rank contribution (normalize to 0-1)
            # Note: c is a dict, so check key (not attribute) for "rank"
            fts_rank = c.get("rank", 0.5) if "rank" in c else 0.5
            score += fts_rank * 0.3

            # Jaccard n-gram overlap
            jac = self._jaccard_ngram(query, c.get("content", ""))
            score += jac * 0.2

            # HRR similarity
            hrr_sim = 0.0
            if query_vec is not None and hrr.HAS_NUMPY:
                try:
                    blob = c.get("hrr_vector")
                    if blob:
                        existing_vec = self._store._get_hrr_cached(c["fact_id"])
                        if existing_vec is not None:
                            if len(query_vec) != len(existing_vec):
                                logger.warning(
                                    "Skipping HRR similarity for fact %s: query dim %s != stored dim %s",
                                    c.get("fact_id"),
                                    len(query_vec),
                                    len(existing_vec),
                                )
                            else:
                                hrr_sim = max(0, hrr.similarity(query_vec, existing_vec))
                        else:
                            vec = hrr.bytes_to_phases(blob)
                            if len(query_vec) != len(vec):
                                logger.warning(
                                    "Skipping HRR similarity for fact %s: query dim %s != stored dim %s",
                                    c.get("fact_id"),
                                    len(query_vec),
                                    len(vec),
                                )
                            else:
                                hrr_sim = max(0, hrr.similarity(query_vec, vec))
                except Exception:
                    logger.exception("HRR similarity failed for fact %s", c.get("fact_id"))

            score += hrr_sim * self._hrr_weight

            # Recency boost: exponential decay (halves every recency_half_life days)
            created_raw = c.get("created_at")
            if created_raw:
                try:
                    if isinstance(created_raw, str):
                        created_dt = dt.datetime.fromisoformat(created_raw)
                    else:
                        created_dt = created_raw
                    if created_dt.tzinfo:
                        now = dt.datetime.now(dt.timezone.utc)
                    else:
                        now = dt.datetime.now()
                    days = max(0, (now - created_dt).total_seconds() / 86400)
                    recency = 2.0 ** (-days / self._recency_half_life)
                    score += recency * self._recency_weight
                except Exception:
                    pass

            c["_score"] = score
            c["_hrr_sim"] = hrr_sim

            # Progressive disclosure: add summary field
            if "summary" not in c:
                content = c.get("content", "")
                c["summary"] = content[:200]

        return candidates

    @staticmethod
    def _jaccard_ngram(a: str, b: str, n: int = 3) -> float:
        """Token-level Jaccard similarity on n-grams."""
        if not a or not b:
            return 0.0
        a_tokens = set(re.findall(r"\w+", a.lower()))
        b_tokens = set(re.findall(r"\w+", b.lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    @staticmethod
    def _compute_rrf_k(num_results: int) -> int:
        """Compute dynamic RRF k from result count.

        Formula: ``k = max(10, min(100, num_results * 2))``

        This ensures:
        - Small result sets (≤5) floor at ``k=10`` (more weight on top ranks).
        - Medium sets get ``k = 2× num_results`` (balanced fusion).
        - Large sets (≥50) cap at ``k=100`` (avoid fusion dilution).

        Args:
            num_results: Number of potentially relevant results.

        Returns:
            RRF k constant.
        """
        return max(10, min(100, num_results * 2))

    @staticmethod
    def _rrf_merge(
        stream_a: list[dict],
        stream_b: list[dict],
        limit: int,
        k: int = 60,
        num_results: Optional[int] = None,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of two ranked streams.

        Items appearing in both streams get a boosted rank.
        When one stream is empty, the other is returned with RRF scores applied.

        When ``num_results`` is provided, ``k`` is computed dynamically via
        ``_compute_rrf_k`` instead of using the static ``k`` parameter.

        Args:
            stream_a: First ranked list (must have ``fact_id`` key).
            stream_b: Second ranked list.
            limit: Max items to return.
            k: RRF constant (default 60). Ignored when ``num_results`` is set.
            num_results: Optional — compute ``k`` dynamically from result count.

        Returns:
            List of merged dicts with a ``score`` key.
        """
        if num_results is not None:
            k = EtchRetriever._compute_rrf_k(num_results)
        if not stream_a and not stream_b:
            return []
        if not stream_b:
            # Single source — assign RRF scores
            result = []
            for rank, item in enumerate(stream_a):
                d = dict(item)
                d["score"] = 1.0 / (k + rank + 1)
                result.append(d)
            return result[:limit]
        if not stream_a:
            result = []
            for rank, item in enumerate(stream_b):
                d = dict(item)
                d["score"] = 1.0 / (k + rank + 1)
                result.append(d)
            return result[:limit]

        scores: dict[int, float] = {}
        items: dict[int, dict] = {}

        for rank, item in enumerate(stream_a):
            fid = item.get("fact_id")
            if fid is not None:
                scores[fid] = scores.get(fid, 0) + 1.0 / (k + rank + 1)
                items.setdefault(fid, item)

        for rank, item in enumerate(stream_b):
            fid = item.get("fact_id")
            if fid is not None:
                scores[fid] = scores.get(fid, 0) + 1.0 / (k + rank + 1)
                items.setdefault(fid, item)

        ranked = sorted(scores.keys(), key=lambda fid: scores[fid], reverse=True)
        result = []
        for fid in ranked[:limit]:
            d = dict(items[fid])
            d["score"] = scores[fid]
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Phase 3: Smarter Search — FTS5 expansion, HRR multi-query, cascade
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_dedup_max_score(sets: list[list[dict]]) -> list[dict]:
        """Merge multiple result sets with dedup by ``fact_id``, keep max score.

        Each result dict is expected to have a ``_score`` key (from
        ``_score_candidates``). When the same ``fact_id`` appears in multiple
        sets, the dict with the highest ``_score`` is kept.

        Args:
            sets: List of scored result lists.

        Returns:
            Merged list sorted by ``_score`` descending.
        """
        best: dict[int, dict] = {}
        for results in sets:
            for r in results:
                fid = r.get("fact_id")
                if fid is None:
                    continue
                existing = best.get(fid)
                if existing is None or r.get("_score", 0) > existing.get("_score", 0):
                    best[fid] = r
        merged = sorted(best.values(), key=lambda x: x.get("_score", 0), reverse=True)
        return merged

    def search_expanded(
        self,
        query: str,
        limit: int = 10,
        match_threshold: int = _DEFAULT_MATCH_THRESHOLD,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        exclude_deleted: bool = True,
        project: str = "",
        scope: str = "canonical",
        source_harness: str = "",
        source_agent: str = "",
        source_kind: str = "",
    ) -> list[dict]:
        """FTS5 search with progressive query expansion.

        Strategy:
        1. **Full query** (depth 0): FTS5 MATCH with query as-is.
        2. **Keywords OR** (depth 1, if < threshold): remove stopwords, OR-join
           remaining content keywords.
        3. **Single terms** (depth 2, if < threshold): search each keyword
           independently, union all results.
        4. Results from each stage are scored via ``_score_candidates`` and
           merged with dedup by ID, keeping the max score per document.

        Args:
            query: Raw search query.
            limit: Max results.
            match_threshold: Minimum results to stop expanding (default: 3).
            max_depth: Number of expansion stages (default: 3).
            exclude_deleted: Whether to exclude soft-deleted facts.
            project: Optional project filter.
            scope: Scope filter (default: ``'canonical'``).
            source_harness: Optional source harness filter.
            source_agent: Optional source agent filter.
            source_kind: Optional source kind filter.

        Returns:
            List of scored result dicts, deduplicated.
        """
        if not query or not query.strip():
            return []

        all_sets: list[list[dict]] = []

        # Depth 0: full query as-is
        depth0 = self._fts_candidates(query, limit * 2, exclude_deleted, project,
                                      scope=scope, source_harness=source_harness,
                                      source_agent=source_agent, source_kind=source_kind)
        if depth0:
            scored0 = self._score_candidates(query, depth0)
            scored0.sort(key=lambda x: x.get("_score", 0), reverse=True)
            all_sets.append(scored0)

        # Check if we need expansion
        current_count = len(self._merge_dedup_max_score(all_sets))
        if current_count >= match_threshold or max_depth <= 1:
            merged = self._merge_dedup_max_score(all_sets)
            # Normalise score key
            for r in merged:
                if "_score" in r:
                    r.setdefault("score", r["_score"])
            return merged[:limit]

        # Depth 1: OR-joined content keywords
        keywords = _extract_keywords(query)
        if keywords:
            or_query = " OR ".join(keywords)
            depth1 = self._fts_candidates(or_query, limit * 2, exclude_deleted, project,
                                          scope=scope, source_harness=source_harness,
                                          source_agent=source_agent, source_kind=source_kind)
            if depth1:
                scored1 = self._score_candidates(or_query, depth1)
                scored1.sort(key=lambda x: x.get("_score", 0), reverse=True)
                all_sets.append(scored1)

        current_count = len(self._merge_dedup_max_score(all_sets))
        if current_count >= match_threshold or max_depth <= 2:
            merged = self._merge_dedup_max_score(all_sets)
            for r in merged:
                if "_score" in r:
                    r.setdefault("score", r["_score"])
            return merged[:limit]

        # Depth 2: single keyword searches, union
        if keywords:
            for kw in keywords:
                kw_set = self._fts_candidates(kw, limit * 2, exclude_deleted, project,
                                              scope=scope, source_harness=source_harness,
                                              source_agent=source_agent, source_kind=source_kind)
                if kw_set:
                    scored_kw = self._score_candidates(kw, kw_set)
                    scored_kw.sort(key=lambda x: x.get("_score", 0), reverse=True)
                    all_sets.append(scored_kw)

        merged = self._merge_dedup_max_score(all_sets)
        for r in merged:
            if "_score" in r:
                r.setdefault("score", r["_score"])
        return merged[:limit]

    @staticmethod
    def _generate_query_variations(query: str) -> list[str]:
        """Generate search query variations for HRR multi-query.

        Produces 2–3 variations:
        - **Original**: the query as-is.
        - **Keywords-only**: stopwords removed.
        - **Bigrams**: adjacent word pairs (only if query has 3+ tokens).

        Args:
            query: Original search query.

        Returns:
            List of query variation strings (minimum 2, maximum 3).
        """
        variations = [query]
        keywords = _extract_keywords(query)
        if keywords and " ".join(keywords) != query:
            variations.append(" ".join(keywords))
        tokens = _sanitize_fts5(query).split()
        if len(tokens) >= 3:
            bigrams = [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
            if bigrams:
                variations.append(" ".join(bigrams))
        return variations

    def _hrr_multi_query(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
        project: str = "",
        scope: str = "canonical",
        source_harness: str = "",
        source_agent: str = "",
        source_kind: str = "",
    ) -> list[dict]:
        """Multi-query HRR search with parallel query variations.

        Generates 2–3 query variations (original, keywords-only, bigrams),
        encodes each via HRR, scores FTS5 candidates for each variation,
        and merges results (dedup by ID, max score across variations).

        Uses ``ThreadPoolExecutor`` (max_workers=3) to parallelize encoding
        and scoring. Daemon threads with 1s timeout.

        Args:
            query: Search query.
            limit: Max results.
            exclude_deleted: Whether to exclude soft-deleted facts.
            project: Optional project filter.
            scope: Scope filter (default: ``'canonical'``).
            source_harness: Optional source harness filter.
            source_agent: Optional source agent filter.
            source_kind: Optional source kind filter.

        Returns:
            Merged list of scored result dicts.
        """
        if not query or not query.strip():
            return []

        variations = self._generate_query_variations(query)
        if not variations:
            return []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _search_variation(var: str) -> list[dict]:
            candidates = self._fts_candidates(var, limit * 2, exclude_deleted, project,
                                              scope=scope, source_harness=source_harness,
                                              source_agent=source_agent, source_kind=source_kind)
            if not candidates:
                return []
            scored = self._score_candidates(var, candidates)
            scored.sort(key=lambda x: x.get("_score", 0), reverse=True)
            return scored

        all_sets: list[list[dict]] = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_search_variation, var): var for var in variations}
            try:
                for future in as_completed(futures, timeout=1.0):
                    try:
                        result = future.result()
                        if result:
                            all_sets.append(result)
                    except Exception:
                        logger.exception("HRR multi-query variation failed")
            except Exception:
                logger.exception("HRR multi-query timed out")
                # Cancel remaining
                for f in futures:
                    f.cancel()

        if not all_sets:
            return []

        merged = self._merge_dedup_max_score(all_sets)
        for r in merged:
            if "_score" in r:
                r.setdefault("score", r["_score"])
        return merged[:limit]

    def _search_auto(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
        project: str = "",
        fallback_thresholds: Optional[list[int]] = None,
        scope: str = "canonical",
        source_harness: str = "",
        source_agent: str = "",
        source_kind: str = "",
    ) -> list[dict]:
        """RRF-fused search across ALL available streams.

        Collects results from **all** available strategies and merges them
        via RRF. Unlike the pure cascade approach, this never short-circuits
        — every active strategy contributes its top results to the ranking.

        Streams (in priority order, all contribute):
        1. FTS5 expanded (``search_expanded``)
        2. HRR multi-query (``_hrr_multi_query``)
        3. Embedding vector search (if configured)

        Args:
            query: Search query.
            limit: Max results.
            exclude_deleted: Whether to exclude soft-deleted facts.
            project: Optional project filter.
            fallback_thresholds: Ignored (kept for backward compat).
                All streams always contribute.
            scope: Scope filter (default: ``'canonical'``).
            source_harness: Optional source harness filter.
            source_agent: Optional source agent filter.
            source_kind: Optional source kind filter.

        Returns:
            List of scored result dicts.
        """
        import warnings

        streams: list[list[dict]] = []

        # Stream 1: FTS5 expanded
        try:
            level1 = self.search_expanded(
                query, limit, exclude_deleted=exclude_deleted, project=project,
                scope=scope, source_harness=source_harness,
                source_agent=source_agent, source_kind=source_kind,
            )
            if level1:
                streams.append(level1)
        except Exception as e:
            warnings.warn(f"FTS5 expanded search failed: {e}")

        # Stream 2: HRR multi-query
        try:
            level2 = self._hrr_multi_query(
                query, limit, exclude_deleted=exclude_deleted, project=project,
                scope=scope, source_harness=source_harness,
                source_agent=source_agent, source_kind=source_kind,
            )
            if level2:
                streams.append(level2)
        except Exception as e:
            warnings.warn(f"HRR multi-query failed: {e}")

        # Stream 3: Embedding vector search (if configured)
        if self._compute_embedding is not None:
            try:
                q_vec = self._compute_embedding(query)
                if q_vec:
                    import struct
                    vec_bytes = struct.pack(f"{len(q_vec)}f", *q_vec)
                    embedding_results = self._store.search_by_vector(
                        vec_bytes, limit=limit, project=project,
                        scope=scope, source_harness=source_harness,
                        source_agent=source_agent, source_kind=source_kind,
                    )
                    if embedding_results:
                        for r in embedding_results:
                            r["_score"] = r.get("score", 0.5)
                        streams.append(embedding_results)
            except Exception:
                logger.exception("Embedding search in auto-cascade failed")

        merged = self._merge_dedup_max_score(streams)
        for r in merged:
            if "_score" in r:
                r.setdefault("score", r["_score"])
        return merged[:limit]

    def probe(
        self,
        topic: str,
        limit: int = 10,
        project: str = "",
        scope: str = "canonical",
        source_harness: str = "",
        source_agent: str = "",
        source_kind: str = "",
    ) -> list[dict]:
        """Search by topic tag or content keyword.

        Matches facts where the tag or content contains *topic*.

        Args:
            topic: Keyword to search for in tags and content.
            limit: Max results (default: 10).
            project: Optional project filter.
            scope: Scope filter (default: ``'canonical'``).
            source_harness: Optional source harness filter.
            source_agent: Optional source agent filter.
            source_kind: Optional source kind filter.

        Returns:
            List of fact dicts with a ``_score`` key.
        """
        with self._store._lock:
            conditions: list[str] = ["(f.deleted IS NULL OR f.deleted = 0)"]
            params: list = []
            if project:
                conditions.append("f.project = ?")
                params.append(project)
            if scope:
                conditions.append("f.scope = ?")
                params.append(scope)
            if source_harness:
                conditions.append("f.source_harness = ?")
                params.append(source_harness)
            if source_agent:
                conditions.append("f.source_agent = ?")
                params.append(source_agent)
            if source_kind:
                conditions.append("f.source_kind = ?")
                params.append(source_kind)
            conditions.append("(f.tags LIKE ? OR f.content LIKE ?)")
            params.extend([f"%{topic}%", f"%{topic}%"])
            w = " AND ".join(conditions)
            rows = self._store._conn.execute(
                f"SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score, "
                f"f.project, f.created_at, f.session_id "
                f"FROM facts f WHERE {w} "
                f"ORDER BY f.trust_score DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        results = [dict(r) for r in rows]
        for r in results:
            r["_score"] = r.get("trust_score", 0.5)
        return results

    def related(
        self,
        topic: str,
        limit: int = 10,
        scope: str = "canonical",
        source_harness: str = "",
        source_agent: str = "",
        source_kind: str = "",
    ) -> list[dict]:
        """Find facts related to a topic via entities + FTS5.

        Searches for facts that share entities with the given topic,
        falling back to FTS5 content match.

        Args:
            topic: Topic keyword to search related facts for.
            limit: Max results (default: 10).
            scope: Scope filter (default: ``'canonical'``).
            source_harness: Optional source harness filter.
            source_agent: Optional source agent filter.
            source_kind: Optional source kind filter.

        Returns:
            List of fact dicts related to the topic.
        """
        # First: find entities matching the topic
        with self._store._lock:
            entity_rows = self._store._conn.execute(
                "SELECT entity_id FROM entities WHERE name LIKE ? LIMIT 5",
                (f"%{topic}%",),
            ).fetchall()
            entity_ids = [r["entity_id"] for r in entity_rows]

            if entity_ids:
                placeholders = ",".join("?" * len(entity_ids))
                base_conditions = ["(f.deleted IS NULL OR f.deleted = 0)"]
                base_params: list = []
                if scope:
                    base_conditions.append("f.scope = ?")
                    base_params.append(scope)
                if source_harness:
                    base_conditions.append("f.source_harness = ?")
                    base_params.append(source_harness)
                if source_agent:
                    base_conditions.append("f.source_agent = ?")
                    base_params.append(source_agent)
                if source_kind:
                    base_conditions.append("f.source_kind = ?")
                    base_params.append(source_kind)
                base_where = " AND ".join(base_conditions)
                rows = self._store._conn.execute(
                    f"""SELECT DISTINCT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                               f.project, f.created_at, f.session_id
                        FROM facts f
                        JOIN fact_entities fe ON fe.fact_id = f.fact_id
                        WHERE fe.entity_id IN ({placeholders})
                          AND {base_where}
                        ORDER BY f.trust_score DESC
                        LIMIT ?""",
                    base_params + entity_ids + [limit],
                ).fetchall()
            else:
                # Fallback: FTS5 search on topic
                try:
                    fts_params: list = [topic]
                    fts_conditions: list[str] = ["(f.deleted IS NULL OR f.deleted = 0)"]
                    if scope:
                        fts_conditions.append("f.scope = ?")
                        fts_params.append(scope)
                    if source_harness:
                        fts_conditions.append("f.source_harness = ?")
                        fts_params.append(source_harness)
                    if source_agent:
                        fts_conditions.append("f.source_agent = ?")
                        fts_params.append(source_agent)
                    if source_kind:
                        fts_conditions.append("f.source_kind = ?")
                        fts_params.append(source_kind)
                    fts_where = " AND ".join(fts_conditions)
                    rows = self._store._conn.execute(
                        f"""SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                                  f.project, f.created_at, f.session_id
                           FROM facts f
                           JOIN facts_fts fts ON fts.rowid = f.fact_id
                           WHERE facts_fts MATCH ?
                             AND {fts_where}
                           ORDER BY fts.rank
                           LIMIT ?""",
                        fts_params + [limit],
                    ).fetchall()
                except Exception:
                    rows = []
        results = [dict(r) for r in rows]
        for r in results:
            r["_score"] = r.get("trust_score", 0.5)
        return results

    # ------------------------------------------------------------------
    # Atlas retrieval methods
    # ------------------------------------------------------------------

    def explore_map(self, map_id: int, max_depth: int = 3) -> list[dict]:
        """Explore a map and return its linked facts, regions, and edges.

        Args:
            map_id: The map ID to explore.
            max_depth: Maximum depth for edge traversal (default: 3).

        Returns:
            List of dicts with node_type, node_id, name, and metadata.
        """
        results = []
        # Get the map itself
        m = self._store.get_map(map_id)
        if m:
            results.append({
                "node_type": "map",
                "node_id": m["map_id"],
                "name": m["name"],
                "description": m.get("description", ""),
            })
            # Get regions
            regions = self._store.list_regions(map_id)
            for r in regions:
                results.append({
                    "node_type": "region",
                    "node_id": r["region_id"],
                    "name": r["name"],
                    "parent_region_id": r.get("parent_region_id"),
                })
                # Get edges per region — linked facts
                edges = self._store.get_edges(node_type="region", node_id=r["region_id"])
                for e in edges:
                    if e["target_type"] == "fact":
                        fact = self._store.get_fact(e["target_id"])
                        if fact:
                            results.append({
                                "node_type": "fact",
                                "node_id": fact["fact_id"],
                                "name": fact.get("content", "")[:100],
                                "relation_type": e["relation_type"],
                                "weight": e["weight"],
                            })
            # Also get edges linked directly to the map
            edges = self._store.get_edges(node_type="map", node_id=map_id)
            for e in edges:
                if e["target_type"] == "fact":
                    fact = self._store.get_fact(e["target_id"])
                    if fact:
                        results.append({
                            "node_type": "fact",
                            "node_id": fact["fact_id"],
                            "name": fact.get("content", "")[:100],
                            "relation_type": e["relation_type"],
                            "weight": e["weight"],
                        })
        return results

    def traverse_path(
        self,
        start_map_id: int,
        end_map_id: int,
        max_depth: int = 5,
    ) -> list[dict]:
        """Find a path between two maps via atlas edges.

        Delegates to ``store.traverse_path``.

        Args:
            start_map_id: Starting map ID.
            end_map_id: Target map ID.
            max_depth: Maximum traversal depth.

        Returns:
            List of node dicts on the path, or empty list.
        """
        return self._store.traverse_path(start_map_id, end_map_id, max_depth)

    def search_map(
        self,
        query: str,
        limit: int = 20,
        project: str = "",
        scope: str = "canonical",
    ) -> list[dict]:
        """Full-text search across atlas maps.

        Delegates to ``store.search_map``.

        Args:
            query: FTS5 search query.
            limit: Max results (default 20).
            project: Optional project filter.
            scope: Ignored for atlas (kept for API compatibility).

        Returns:
            List of map summary dicts.
        """
        return self._store.search_map(query, limit=limit, project=project, scope=scope)

    def contradict(self, limit: int = 10) -> list[dict]:
        """Find contradictions — known (fact_relations) then algorithmic.

        First checks existing relations, then falls back to a heuristic
        scan of facts sharing the same category within a project.

        Args:
            limit: Max contradiction pairs to return (default: 10).

        Returns:
            List of contradictory fact pair dicts with ``fact_id_a``,
            ``content_a``, ``fact_id_b``, ``content_b``, and ``source``.
        """
        # 1. Known contradictions from fact_relations
        known = self._store.get_contradictions(limit)
        for r in known:
            r["source"] = "fact_relations"
        if known:
            return known[:limit]

        # 2. Algorithmic fallback: scan for contradictory content in same project
        with self._store._lock:
            rows = self._store._conn.execute(
                """SELECT f1.fact_id AS fact_id_a, f1.content AS content_a,
                          f2.fact_id AS fact_id_b, f2.content AS content_b,
                          0.5 AS confidence
                   FROM facts f1
                   JOIN facts f2 ON f1.project = f2.project AND f1.fact_id < f2.fact_id
                   WHERE (f1.deleted IS NULL OR f1.deleted = 0)
                     AND (f2.deleted IS NULL OR f2.deleted = 0)
                     AND f1.category = f2.category
                   ORDER BY RANDOM()
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
