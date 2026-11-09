"""Unit tests for MCP server tool schema and dispatch."""

import asyncio
import json
from unittest.mock import patch

import pytest

from rag.mcp_server import call_tool, list_tools


def test_list_tools_exposes_search_knowledge_base_schema() -> None:
    tools = asyncio.run(list_tools())
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "search_knowledge_base"
    assert "personal knowledge base" in (tool.description or "")
    assert tool.inputSchema["type"] == "object"
    assert "q" in tool.inputSchema["properties"]
    assert tool.inputSchema["properties"]["k"]["default"] == 5
    assert tool.inputSchema["required"] == ["q"]


def test_call_tool_dispatches_to_rag_query_and_serializes_json() -> None:
    fake_result = {
        "answer": "pgvector HNSW is used for dense retrieval [S1].",
        "citations": [{"id": "S1", "source": "examples/architecture-decisions.md", "text": "..."}],
        "usage": {"total_tokens": 100},
        "latency_ms": 15,
    }
    with patch("rag.mcp_server.rag_query", return_value=fake_result) as mock_rag_query:
        content = asyncio.run(
            call_tool("search_knowledge_base", {"q": "Why pgvector?", "k": 4})
        )

    mock_rag_query.assert_called_once_with("Why pgvector?", k=4)
    assert len(content) == 1
    assert content[0].type == "text"
    parsed = json.loads(content[0].text)
    assert parsed == fake_result


def test_call_tool_uses_default_k_when_omitted() -> None:
    with patch("rag.mcp_server.rag_query", return_value={"answer": "ok"}) as mock_rag_query:
        asyncio.run(call_tool("search_knowledge_base", {"q": "default k test"}))

    mock_rag_query.assert_called_once_with("default k test", k=5)


def test_call_tool_raises_on_unknown_tool_name() -> None:
    with pytest.raises(ValueError, match="unknown tool nonexistent_tool"):
        asyncio.run(call_tool("nonexistent_tool", {"q": "test"}))
