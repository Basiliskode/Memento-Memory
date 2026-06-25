"""Tests for the MCP stdio server."""

import json
import os
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import StdioServerParameters, stdio_client

MCP_MODULE = "memento.mcp.__main__"


@pytest.fixture
def server_params(tmp_path) -> StdioServerParameters:
    """Create StdioServerParameters pointing to our MCP server."""
    db_path = str(tmp_path / "test_etch.db")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "memento.mcp"],
        env={**os.environ, "MEMENTO_DB_PATH": db_path},
    )


@pytest.fixture
def server_params_memory() -> StdioServerParameters:
    """Create StdioServerParameters using in-memory DB (for tests)."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "memento.mcp"],
        env={**os.environ, "MEMENTO_DB_PATH": ":memory:"},
    )


class TestMCPTools:
    """Integration tests for MCP tools via stdio transport."""

    @pytest.mark.asyncio
    async def test_list_tools(self, server_params_memory):
        """Server exposes the expected tools (9 base + 7 atlas + 6 mem_* = 22)."""
        async with stdio_client(server_params_memory) as (read, write):
            from mcp import ClientSession
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                tool_names = {t.name for t in result.tools}
                assert "add_fact" in tool_names
                assert "search_facts" in tool_names
                assert "get_fact" in tool_names
                assert "delete_fact" in tool_names
                assert "get_timeline" in tool_names
                assert "search_similar" in tool_names
                assert "list_inbox" in tool_names
                assert "promote_fact" in tool_names
                assert "reject_fact" in tool_names
                assert "get_version" in tool_names
                # Atlas tools
                assert "read_map" in tool_names
                assert "list_maps" in tool_names
                assert "create_map" in tool_names
                assert "link_fact" in tool_names
                assert "search_map" in tool_names
                assert "list_regions" in tool_names
                assert len(tool_names) == 22  # 9 base + 7 atlas + 6 mem_* aliases

    @pytest.mark.asyncio
    async def test_add_fact_and_search_roundtrip(self, server_params_memory):
        """add_fact creates a fact, search_facts finds it."""
        async with stdio_client(server_params_memory) as (read, write):
            from mcp import ClientSession
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Add a fact
                add_result = await session.call_tool(
                    "add_fact",
                    arguments={
                        "content": "Python is a programming language",
                        "project": "test",
                        "session_id": "s1",
                        "topic_key": "topic:python",
                    },
                )
                add_data = _parse_mcp_result(add_result)
                assert "id" in add_data
                assert add_data["id"] > 0
                assert add_data["status"] in ("created", "updated")

                # Search for it
                search_result = await session.call_tool(
                    "search_facts",
                    arguments={"query": "Python", "limit": 5, "project": "test"},
                )
                search_data = _parse_mcp_result(search_result)
                assert isinstance(search_data, list)
                assert len(search_data) >= 1
                assert search_data[0]["content"] == "Python is a programming language"

    @pytest.mark.asyncio
    async def test_get_fact(self, server_params_memory):
        """get_fact returns fact dict or not_found."""
        async with stdio_client(server_params_memory) as (read, write):
            from mcp import ClientSession
            async with ClientSession(read, write) as session:
                await session.initialize()

                add_result = await session.call_tool(
                    "add_fact", arguments={"content": "Test fact"}
                )
                add_data = _parse_mcp_result(add_result)
                fid = add_data["id"]

                # Get existing
                get_result = await session.call_tool(
                    "get_fact", arguments={"fact_id": fid}
                )
                get_data = _parse_mcp_result(get_result)
                assert isinstance(get_data, dict)
                assert get_data.get("fact_id") == fid
                assert get_data["content"] == "Test fact"

                # Get non-existent
                not_found = await session.call_tool(
                    "get_fact", arguments={"fact_id": 99999}
                )
                nf_data = _parse_mcp_result(not_found)
                assert nf_data == {"status": "not_found"}

    @pytest.mark.asyncio
    async def test_delete_fact(self, server_params_memory):
        """delete_fact removes a fact and returns status."""
        async with stdio_client(server_params_memory) as (read, write):
            from mcp import ClientSession
            async with ClientSession(read, write) as session:
                await session.initialize()

                add_result = await session.call_tool(
                    "add_fact", arguments={"content": "To delete"}
                )
                add_data = _parse_mcp_result(add_result)
                fid = add_data["id"]

                # Delete existing
                del_result = await session.call_tool(
                    "delete_fact", arguments={"fact_id": fid}
                )
                del_data = _parse_mcp_result(del_result)
                assert del_data == {"status": "deleted"}

                # Verify gone
                get_result = await session.call_tool(
                    "get_fact", arguments={"fact_id": fid}
                )
                get_data = _parse_mcp_result(get_result)
                assert get_data == {"status": "not_found"}

    @pytest.mark.asyncio
    async def test_delete_fact_nonexistent(self, server_params_memory):
        """delete_fact on non-existent returns not_found."""
        async with stdio_client(server_params_memory) as (read, write):
            from mcp import ClientSession
            async with ClientSession(read, write) as session:
                await session.initialize()

                del_result = await session.call_tool(
                    "delete_fact", arguments={"fact_id": 99999}
                )
                del_data = _parse_mcp_result(del_result)
                assert del_data == {"status": "not_found"}

    @pytest.mark.asyncio
    async def test_get_timeline(self, server_params_memory):
        """get_timeline returns a list of facts."""
        async with stdio_client(server_params_memory) as (read, write):
            from mcp import ClientSession
            async with ClientSession(read, write) as session:
                await session.initialize()

                _ = await session.call_tool(
                    "add_fact", arguments={"content": "Timeline fact 1"}
                )
                _ = await session.call_tool(
                    "add_fact", arguments={"content": "Timeline fact 2"}
                )

                tl_result = await session.call_tool(
                    "get_timeline", arguments={"limit": 10}
                )
                tl_data = _parse_mcp_result(tl_result)
                assert isinstance(tl_data, list)
                assert len(tl_data) >= 2

    @pytest.mark.asyncio
    async def test_search_similar(self, server_params_memory):
        """search_similar returns similar facts."""
        async with stdio_client(server_params_memory) as (read, write):
            from mcp import ClientSession
            async with ClientSession(read, write) as session:
                await session.initialize()

                await session.call_tool(
                    "add_fact", arguments={"content": "Python is great for data science"}
                )
                await session.call_tool(
                    "add_fact", arguments={"content": "Python has many libraries"}
                )

                sim_result = await session.call_tool(
                    "search_similar", arguments={"query": "Python", "limit": 5}
                )
                sim_data = _parse_mcp_result(sim_result)
                assert isinstance(sim_data, list)
                assert len(sim_data) >= 1

    @pytest.mark.asyncio
    async def test_memento_db_path_env_var(self, tmp_path):
        """MEMENTO_DB_PATH env var controls the database path."""
        db_path = str(tmp_path / "custom_etch.db")
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "memento.mcp"],
            env={**os.environ, "MEMENTO_DB_PATH": db_path},
        )
        async with stdio_client(params) as (read, write):
            from mcp import ClientSession
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "add_fact", arguments={"content": "Custom DB path fact"}
                )
                data = _parse_mcp_result(result)
                assert data["status"] in ("created", "updated")
                assert data["id"] > 0

        # Verify the db file was created at the custom path
        assert Path(db_path).exists()


def _parse_mcp_result(result):
    """Parse MCP CallToolResult content into a Python object."""
    if hasattr(result, "content") and result.content:
        text = result.content[0].text
        return json.loads(text)
    return result
