# ai-abe — Agent Builder Engine

ABE builds a document-grounded AI agent from a short intake form and a folder of documents. You describe the
problem, upload PDF, Word or text files, and write a few questions you already know the answers to. ABE produces
an agent package with a retrieval index, a generated system prompt, a starter test set, a cost model and an MCP
server entry, then lets you ask it questions and evaluate it in the browser.

It is meant to be cloned and adapted. The code is small, has few dependencies, and each part can be replaced
without touching the others.

## What you get

| Piece | What it does |
|---|---|
| Intake form | Walks through the problem, documents, rules, model choice, test questions and running cost, with an optional build/no-build screen. |
| Retrieval | Reads PDF, DOCX, TXT and Markdown; chunks on paragraph and sentence boundaries; indexes with BM25. Works offline. |
| Agent | Answers only from retrieved passages, cites them, and declines when nothing relevant is found. Effort level and answer-style options adjust the system prompt. |
| Providers | Azure OpenAI (default), OpenRouter, OpenAI, any OpenAI-compatible server including Ollama for local models, Anthropic, and a no-model extractive baseline. |
| MCP server | Exposes `search_knowledge`, `ask`, `list_sources` and the agent's rules to Claude Desktop, VS Code or any MCP client. A retrieval-only mode lets the client's own model answer on an existing subscription. |
| Evaluation | Runs a YAML test set, scores answers and retrieval separately, writes Markdown and JSON reports, and fails CI below a threshold. |
| Intake screen (optional) | For making the case to a sponsor: data readiness, technical feasibility, business readiness, minimum viable accuracy, output verifiability, build versus buy. Leave it blank to skip it. |
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

Activate the virtual environment in every new terminal before running `abe`; otherwise PowerShell reports that
`abe` isn't recognised. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process Bypass`
first, which applies to that window only.

To try it without a model, choose **No model: quote the best passages** as the provider. To use a local
model, install and start [Ollama](https://ollama.com) and choose **Local model (Ollama)**. The form lists the
models Ollama already has; pick one, or type the name of another and ABE downloads it when you build the agent.
If that download fails, run `ollama pull <model>` yourself. See
[docs/models-and-billing.md](docs/models-and-billing.md#local-models-with-ollama).

A local model has to fit in your graphics card's memory (VRAM): its download size plus about 2 GB should be no more
than your VRAM. On an 8 GB card, `gemma4:e4b-it-qat` or `llama3.1:8b` fit; a 26B model needs about 20 GB. If the
model is too big, asking a question fails with an "out of memory" error. [docs/choosing-a-local-model.md](docs/choosing-a-local-model.md)
explains how to check your VRAM, what the tag names mean, and how to confirm the model is running on the GPU.

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
abe set-model AGENT MODEL [--provider ollama|openrouter|openai|azure_openai|anthropic|custom|extractive]
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
  intake.md           build recommendation, if the screen was filled in
  cost.md             cost model
  mcp.json            MCP client entry
  README.md           how to run this agent
```

## Changing an agent

Open the agent's page and select **Edit agent**. From there you can:

- change the model or switch provider, for example from Azure OpenAI to a local Ollama model
- edit the description, audience, out-of-scope list, tone and citation rules
- set the effort level (Quick, Standard or Thorough) and answer-style options such as numbered steps, commands
  shown as code, version-aware answers, flagging conflicting passages, or allowing labelled general knowledge
- add or remove documents
- adjust how many passages are sent per question, the minimum match score and the passage size
- edit the test set

The agent keeps its folder name, so MCP client entries and links keep working. Changing documents or passage
size rebuilds the search index; nothing else does. If you pick an Ollama model that isn't installed, ABE starts
downloading it. Run the test set after a change to see what it did.

To change only the model from the command line:

```
abe set-model powershell-guide gemma4:e4b-it-qat --provider ollama
```

Leave out `--provider` to keep the current provider and change only the model name. Every setting also lives in the
agent's `agent.yaml`, which you can edit by hand; run `abe reindex AGENT` afterwards if you changed passage size.

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

## Running it safely

ABE is a single-user tool for your own machine.

- `abe serve` listens on `127.0.0.1` only and has no login. Don't run it with `--host 0.0.0.0` on a network you
  don't control: anyone who can reach the port can upload documents and read every agent's answers.
- Uploaded documents are copied into `agents/<slug>/docs/`. The `agents/` folder is in `.gitignore` so they aren't
  committed by accident.
- API keys are read from environment variables named in each agent's `agent.yaml`. They are never written to disk.
- With a cloud provider, the passages retrieved for each question are sent to that provider. Use Ollama when the
  documents must stay on your machine.

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
