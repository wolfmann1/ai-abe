# ai-abe — Agent Builder Engine

ABE builds a document-grounded AI agent from a short intake form and a folder of documents. You describe the
problem, score whether it is worth building, upload PDF, Word or text files, and write a few questions you
already know the answers to. ABE produces an agent package with a retrieval index, a generated system prompt,
a starter test set, a cost model and an MCP server entry, then lets you ask it questions and evaluate it in
the browser.

It is meant to be cloned and adapted. The code is small, has few dependencies, and each part can be replaced
without touching the others.

## What you get

| Piece | What it does |
|---|---|
| Intake form | Walks through the problem, a six-criterion build/no-build screen, documents, rules, model choice, test questions and running cost. |
| Retrieval | Reads PDF, DOCX, TXT and Markdown; chunks on paragraph and sentence boundaries; indexes with BM25. Works offline. |
| Agent | Answers only from retrieved passages, cites them, and declines when nothing relevant is found. |
| Providers | Azure OpenAI (default), OpenRouter, OpenAI, any OpenAI-compatible server including Ollama for local models, Anthropic, and a no-model extractive baseline. |
| MCP server | Exposes `search_knowledge`, `ask`, `list_sources` and the agent's rules to Claude Desktop, VS Code or any MCP client. A retrieval-only mode lets the client's own model answer on an existing subscription. |
| Evaluation | Runs a YAML test set, scores answers and retrieval separately, writes Markdown and JSON reports, and fails CI below a threshold. |
| Intake screen | Data readiness, technical feasibility, business readiness, minimum viable accuracy, output verifiability, build versus buy. |
| Cost model | Monthly token volume from usage assumptions and retrieval settings; monthly cost from the rates you enter. |
| CI | GitHub Actions: lint, unit tests, example build, retrieval and baseline evaluation on every push; full model evaluation when secrets are configured. |

## Quick start

Python 3.10 or later.

```
git clone https://github.com/wolfmann1/ai-abe.git
cd ai-abe
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -e ".[mcp]"
abe serve
```

Open http://127.0.0.1:8765, fill in the form, upload documents and select **Build agent**.

To try it without a model, choose **No model: quote the best passages** as the provider. To use a local
model, install and start [Ollama](https://ollama.com) and choose **Local model (Ollama)**. The form lists the
models Ollama already has; pick one, or type the name of another and ABE downloads it when you build the agent.
If that download fails, run `ollama pull <model>` yourself. See
[docs/models-and-billing.md](docs/models-and-billing.md#local-models-with-ollama).

For Azure OpenAI, enter the resource URL and deployment name, and set the key in your shell:

```
$env:AZURE_OPENAI_API_KEY = "..."      # PowerShell
export AZURE_OPENAI_API_KEY=...        # bash / zsh
```

## Paying for the model

| Option | Billed to |
|---|---|
| An existing Claude Pro/Max or GitHub Copilot plan, through MCP | Your subscription; no key in ABE |
| Azure OpenAI | Your Azure subscription |
| OpenRouter | Prepaid OpenRouter credits, one balance across many models |
| OpenAI or Anthropic API | Pay-as-you-go API account |
| Ollama | Nothing; runs locally |

To use OpenRouter credits, create a key at openrouter.ai, set it as `OPENROUTER_API_KEY`, choose **OpenRouter** in
the form and enter a model ID such as `openai/gpt-5.2`. Chat subscriptions can't be called as an API, so ABE plugs
into the app instead, as a retrieval-only MCP server. [docs/models-and-billing.md](docs/models-and-billing.md) has
step-by-step instructions for each option and the trade-offs of the subscription route.

## The example

`examples/ops-diagnostics` is an on-call assistant for a fictional artifact cache service called Stash: three
short runbooks, a blueprint, and five test questions.

```
abe build examples/ops-diagnostics/blueprint.yaml --docs examples/ops-diagnostics/docs
abe eval stash-ops-assistant --provider extractive
abe ask stash-ops-assistant "Can I delete old files from /srv/stash with rm?" --provider extractive
```

With no model, the extractive baseline retrieves the right document for every question and passes 3 of the 6
test cases. It finds the right passage and quotes sentences that share words with the question, which
is often not the sentence holding the answer. That 50% is the floor a language model has to beat to justify its
cost; run the same evaluation with a configured provider to see by how much.

## Command line

```
abe serve [--host 127.0.0.1] [--port 8765]
abe build BLUEPRINT --docs DIR
abe reindex AGENT
abe ask AGENT "question" [--provider KIND]
abe eval AGENT [--provider KIND] [--threshold 0.8] [--min-retrieval 1.0] [--json]
abe mcp AGENT [--transport stdio|streamable-http] [--retrieval-only]
```

`AGENT` is a path to an agent package or its name inside the workspace (`--workspace`, default `./agents`).
`abe eval` exits with status 1 when accuracy falls below the threshold, so it can gate a pipeline.

## Agent package

```
agents/<slug>/
  blueprint.yaml      answers from the form; rebuild with abe build
  agent.yaml          model, retrieval settings and rules
  system_prompt.md    generated prompt, for review
  docs/               source documents
  index.json          retrieval index
  evals/testset.yaml  test cases; evals/report.md after a run
  intake.md           build recommendation
  cost.md             cost model
  mcp.json            MCP client entry
  README.md           how to run this agent
```

## Connecting an MCP client

Each package's `mcp.json` holds an entry like this, with the full path filled in:

```json
{
  "mcpServers": {
    "abe-stash-ops-assistant": {
      "command": "python",
      "args": ["-m", "abe", "mcp", "C:/path/to/agents/stash-ops-assistant"]
    }
  }
}
```

Paste it into Claude Desktop's `claude_desktop_config.json` or VS Code's MCP settings. `search_knowledge` returns
passages for the client's own model to reason over; `ask` returns the agent's own cited answer. `mcp.json` also
holds a `subscriptionAlternative` entry that adds `--retrieval-only`: it drops `ask`, so the client's model answers
from the passages under the `agent_rules` prompt and no API key is needed.

## Writing a good test set

Start with the questions users actually ask, and add a case every time you find a wrong answer.
`expect_contains` lists phrases a correct answer must include; `expect_source` names the document it should
cite; `expect_decline: true` marks questions the agent must refuse. The report scores retrieval and answers
separately, so a failure points to the right fix: documents and chunking when retrieval misses, prompt and
model when retrieval hits and the answer is still wrong.

## Design

See [docs/architecture.md](docs/architecture.md) for the module map, the design decisions and how to extend it, and
[docs/models-and-billing.md](docs/models-and-billing.md) for model and billing options.

## Development

```
pip install -e ".[mcp,dev]"
ruff check .
python -m unittest discover -s tests -t .
```

## Author

Christian Lesemann. Released under the [MIT License](LICENSE).
