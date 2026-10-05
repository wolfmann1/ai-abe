"""Command line interface.

    abe serve                        run the intake form at http://127.0.0.1:8765
    abe build blueprint.yaml --docs DIR
    abe reindex AGENT
    abe ask AGENT "question"
    abe eval AGENT [--provider extractive] [--testset PATH]
    abe mcp AGENT [--transport stdio|streamable-http] [--retrieval-only]

AGENT is a path to an agent package, or the name of one inside --workspace (default ./agents).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_WORKSPACE = "agents"


def resolve_agent(value: str, workspace: str) -> Path:
    path = Path(value)
    if (path / "agent.yaml").exists():
        return path
    candidate = Path(workspace) / value
    if (candidate / "agent.yaml").exists():
        return candidate
    raise SystemExit(f"No agent package at {value} or {candidate}")


def cmd_serve(args) -> int:
    import uvicorn

    from .web import create_app

    print(f"ai-abe on http://{args.host}:{args.port}  (workspace: {Path(args.workspace).resolve()})")
    uvicorn.run(create_app(args.workspace), host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_build(args) -> int:
    from .builder import Blueprint, build_agent
    from .ingest import SUPPORTED_SUFFIXES

    blueprint = Blueprint.load(args.blueprint)
    docs = [p for p in Path(args.docs).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not docs:
        raise SystemExit(f"No supported documents found in {args.docs}")
    agent_dir = build_agent(blueprint, docs, args.workspace)
    print(f"Built {agent_dir} from {len(docs)} documents")
    return 0


def cmd_reindex(args) -> int:
    from .builder import build_index

    index = build_index(resolve_agent(args.agent, args.workspace))
    print(f"Indexed {len(index.chunks)} passages from {len(index.sources())} documents")
    return 0


def _agent(args):
    from .agent import Agent
    from .providers import make_provider
    from .spec import AgentSpec, ProviderConfig

    agent_dir = resolve_agent(args.agent, args.workspace)
    provider = None
    if getattr(args, "provider", None):
        spec = AgentSpec.load(agent_dir / "agent.yaml")
        provider = make_provider(ProviderConfig(**{**spec.provider.model_dump(), "kind": args.provider}))
    return agent_dir, Agent.from_directory(agent_dir, provider)


def cmd_ask(args) -> int:
    from .providers import ProviderError

    _dir, agent = _agent(args)
    try:
        answer = agent.ask(args.question)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(answer.text)
    if answer.sources:
        print(f"\nSources: {', '.join(answer.sources)}")
    return 0


def cmd_eval(args) -> int:
    from .evals import TestSet, run_eval, write_report
    from .providers import ProviderError

    agent_dir, agent = _agent(args)
    testset = TestSet.load(args.testset or agent_dir / "evals" / "testset.yaml")
    if args.threshold is not None:
        testset.threshold = args.threshold
    try:
        report = run_eval(agent, testset)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    md, _js = write_report(report, agent_dir / "evals")
    print(report.to_markdown())
    print(f"Report written to {md}")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    ok = report.passed
    if args.min_retrieval is not None:
        hit = report.retrieval_hit_rate
        if hit is not None and hit < args.min_retrieval:
            print(f"Retrieval hit rate {hit:.0%} is below the required {args.min_retrieval:.0%}")
            ok = False
    return 0 if ok else 1


def cmd_mcp(args) -> int:
    try:
        from .mcp_server import run
    except ImportError:
        raise SystemExit("The MCP server needs the mcp package: pip install 'ai-abe[mcp]'") from None
    run(resolve_agent(args.agent, args.workspace), transport=args.transport, retrieval_only=args.retrieval_only)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abe", description="ai-abe: Agent Builder Engine")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="folder holding agent packages")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the web intake form")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("build", help="build an agent package from a blueprint file")
    p.add_argument("blueprint")
    p.add_argument("--docs", required=True, help="folder of PDF, DOCX, TXT or MD files")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("reindex", help="rebuild the retrieval index after changing docs/")
    p.add_argument("agent")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("ask", help="ask an agent one question")
    p.add_argument("agent")
    p.add_argument("question")
    p.add_argument("--provider", help="override the provider kind, e.g. extractive")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="run the test set; exits 1 below the threshold")
    p.add_argument("agent")
    p.add_argument("--testset")
    p.add_argument("--provider", help="override the provider kind, e.g. extractive")
    p.add_argument("--threshold", type=float, help="override the test set's accuracy threshold (0-1)")
    p.add_argument("--min-retrieval", type=float, help="also fail if the retrieval hit rate is below this (0-1)")
    p.add_argument("--json", action="store_true", help="also print the JSON report")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("mcp", help="serve an agent over MCP")
    p.add_argument("agent")
    p.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help="expose search and rules only, so the client's own model and subscription do the answering",
    )
    p.set_defaults(func=cmd_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
