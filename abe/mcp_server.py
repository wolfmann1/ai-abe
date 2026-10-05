"""Serve an agent package to MCP clients.

Tools exposed:

    search_knowledge(query, top_k)  ranked passages with source and score
    ask(question)                   the agent's answer, with cited sources (needs the agent's model key)
    list_sources()                  documents in the knowledge base

Prompt exposed:

    agent_rules                     the agent's system prompt, for the client's model to follow

`search_knowledge` and `agent_rules` let a client's own model (Claude Desktop on
a Pro or Max plan, GitHub Copilot in VS Code) do the reasoning on the user's
existing subscription, with no API key configured in ABE. `ask` uses the agent's
configured provider instead. `retrieval_only=True` leaves `ask` out.
"""

from __future__ import annotations

from pathlib import Path

from .agent import Agent
from .providers import ProviderError

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server


def create_server(agent_dir: str | Path, retrieval_only: bool = False):
    agent = Agent.from_directory(agent_dir)
    spec = agent.spec
    server = _Server(
        name=f"abe-{spec.slug}",
        instructions=(
            f"{spec.name}. {spec.description} Call search_knowledge before answering any question in this "
            "domain, answer only from the passages it returns, and cite their sources. The agent_rules prompt "
            "holds the full rules."
        ),
    )

    @server.prompt(name="agent_rules", description=f"Rules and context for {spec.name}.")
    def agent_rules() -> str:
        return spec.system_prompt() + (
            "\n\nBefore answering, call the search_knowledge tool with the user's question. Number the "
            "returned passages in order and cite them as [1], [2]."
        )

    @server.tool(description="Search the knowledge base. Returns the best-matching passages with source and score.")
    def search_knowledge(query: str, top_k: int = 4) -> list[dict]:
        hits = agent.index.search(query, max(1, min(top_k, 20)))
        return [{"source": c.source, "id": c.id, "score": s, "text": c.text} for c, s in hits]

    if not retrieval_only:

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


def run(agent_dir: str | Path, transport: str = "stdio", retrieval_only: bool = False) -> None:
    create_server(agent_dir, retrieval_only).run(transport=transport)
