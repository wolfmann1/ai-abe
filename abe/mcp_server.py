"""Serve an agent package to MCP clients.

Tools exposed:

    search_knowledge(query, top_k)  ranked passages with source and score
    ask(question)                   the agent's answer, with cited sources
    list_sources()                  documents in the knowledge base

`search_knowledge` lets a client's own model reason over the passages; `ask`
uses the agent's configured provider and rules.
"""

from __future__ import annotations

from pathlib import Path

from .agent import Agent
from .providers import ProviderError

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server


def create_server(agent_dir: str | Path):
    agent = Agent.from_directory(agent_dir)
    spec = agent.spec
    server = _Server(
        name=f"abe-{spec.slug}",
        instructions=(
            f"{spec.name}. {spec.description} Use search_knowledge to read source passages, "
            "or ask for a cited answer."
        ),
    )

    @server.tool(description="Search the knowledge base. Returns the best-matching passages with source and score.")
    def search_knowledge(query: str, top_k: int = 4) -> list[dict]:
        hits = agent.index.search(query, max(1, min(top_k, 20)))
        return [{"source": c.source, "id": c.id, "score": s, "text": c.text} for c, s in hits]

    @server.tool(description="Ask the agent a question. Returns its answer and the sources it cited.")
    def ask(question: str) -> dict:
        try:
            answer = agent.ask(question)
        except ProviderError as exc:
            return {"answer": "", "sources": [], "declined": True, "error": str(exc)}
        return {"answer": answer.text, "sources": answer.sources, "declined": answer.declined}

    @server.tool(description="List the documents in the knowledge base.")
    def list_sources() -> list[str]:
        return agent.index.sources()

    return server


def run(agent_dir: str | Path, transport: str = "stdio") -> None:
    create_server(agent_dir).run(transport=transport)
