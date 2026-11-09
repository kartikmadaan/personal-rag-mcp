import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .pipeline import query as rag_query

server = Server("personal-rag-mcp")


@server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge_base",
            description=(
                "Search the user's personal knowledge base (notes, docs, code) and "
                "return a grounded answer with citations. Use when the user asks "
                "about their own materials, prior decisions, or codebase context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Natural-language question"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["q"],
            },
        ),
    ]


@server.call_tool()  # type: ignore[no-untyped-call,untyped-decorator]
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name != "search_knowledge_base":
        raise ValueError(f"unknown tool {name}")
    out = rag_query(arguments["q"], k=arguments.get("k", 5))
    return [TextContent(type="text", text=json.dumps(out, indent=2))]


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()

