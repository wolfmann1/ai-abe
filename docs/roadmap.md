# Roadmap

## v0.1 (this release)

- Web intake form, agent package builder, BM25 retrieval over PDF, DOCX, TXT and Markdown
- Providers: Azure OpenAI, OpenAI-compatible (including Ollama), Anthropic, extractive baseline
- MCP server, evaluation harness with CI gate, intake screen, cost model

## Next

| Item | Why |
|---|---|
| Embedding retrieval (Azure OpenAI or local) as an option beside BM25 | Paraphrased questions that share few words with the source |
| Measured cost: feed average tokens from the last eval run into `cost.md` | Replace the estimate with real usage |
| Second example: lease and property document Q&A | Shows the same pipeline on business documents |
| Evaluation history: keep each run's JSON and chart accuracy over time | See whether a change helped |
| Tool calls beyond retrieval (declared HTTP APIs exposed through MCP) | Agents that act, not only answer |
| Answer feedback in the workbench (correct / wrong, saved as new test cases) | Grow the test set from real use |
| Container image | One-command deployment |
