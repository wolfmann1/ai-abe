# Roadmap

## v0.1 (this release)

- Web intake form, agent package builder, BM25 retrieval over PDF, DOCX, TXT and Markdown
- Providers: Azure OpenAI, OpenRouter, OpenAI-compatible (including Ollama), Anthropic, extractive baseline
- Ollama integration: lists installed models in the form and downloads missing ones in the background
- MCP server, including a retrieval-only mode for clients that use an existing chat subscription
- Edit an existing agent: model, rules, documents, search settings and test set, from the web or `abe set-model`
- Effort levels (Quick, Standard, Thorough, Custom) and answer-style options that adjust the system prompt, plus
  your own rules and style lines or a hand-written system prompt
- Evaluation harness with a CI gate, optional intake screen, cost model
- Guides: models and billing, choosing a local model for your GPU

## Next

| Item | Why |
|---|---|
| Embedding retrieval (Azure OpenAI or local) as an option beside BM25 | Paraphrased questions that share few words with the source |
| Measured cost: feed average tokens from the last eval run into `cost.md` | Replace the estimate with real usage |
| Second example: lease and property document Q&A | Shows the same pipeline on business documents |
| Evaluation history: keep each run's JSON and chart accuracy over time | See whether a change helped |
| Tool calls beyond retrieval (declared HTTP APIs exposed through MCP) | Agents that act, not only answer |
| Answer feedback in the workbench (correct / wrong, saved as new test cases) | Grow the test set from real use |
| Model reasoning switch: OpenAI `reasoning_effort`, Anthropic extended thinking, Ollama `think` | Better answers to multi-part questions on models that support it |
| Container image | One-command deployment |
