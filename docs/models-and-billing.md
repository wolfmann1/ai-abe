# Models and billing

ABE needs a model only for the `ask` path: the web workbench, `abe ask`, `abe eval`, and the MCP `ask` tool.
Retrieval, the optional intake screen, the cost model and the MCP `search_knowledge` tool run without one. That leaves
five ways to pay for the model.

| Option | Billed to | Key needed in ABE | Best for |
|---|---|---|---|
| Existing chat subscription, through MCP | Your Claude Pro/Max plan or GitHub Copilot plan | No | Individuals who already pay for an assistant |
| Azure OpenAI | Your Azure subscription | Yes, `AZURE_OPENAI_API_KEY` | Organizations already on Azure and Microsoft 365 |
| OpenRouter | Prepaid OpenRouter credits | Yes, `OPENROUTER_API_KEY` | Trying many models on one balance |
| OpenAI or Anthropic API | Pay-as-you-go API account | Yes | Direct vendor billing |
| Ollama, local | Nothing; runs on your hardware | No | Offline use, sensitive documents |

## Using an existing subscription

Consumer chat subscriptions don't include API access. ChatGPT Plus and Pro are billed separately from the
OpenAI API, and Anthropic does not allow Claude Free, Pro or Max sign-ins to be used by third-party tools;
third-party tools have to use an API key from the Claude Console or a supported cloud provider. ABE therefore
can't send requests on a subscription's behalf, and it doesn't try.

What it can do is the reverse: run as an MCP server inside an app the subscription already covers. The app's
model does the reasoning, and ABE supplies the documents and the rules.

1. Build the agent as usual. In step 5 of the form, choose **No model** if you won't use the workbench.
2. Open the agent's `mcp.json` and copy the entry under `subscriptionAlternative`. It runs
   `abe mcp <agent> --retrieval-only`, which exposes:
   - `search_knowledge`: the best-matching passages, with sources and scores
   - `list_sources`: the documents in the knowledge base
   - the `agent_rules` prompt: the agent's system prompt, with an instruction to search before answering
3. Add the entry to your client:
   - **Claude Desktop** (Pro or Max plan): Settings → Developer → Edit Config opens `claude_desktop_config.json`.
     Paste the entry inside `"mcpServers"` and restart Claude Desktop.
   - **VS Code with GitHub Copilot**: add the entry to `.vscode/mcp.json` in your workspace (VS Code uses
     `"servers"` as the top-level key in that file) and start it from the MCP view. Use it from Copilot Chat in
     agent mode.
4. Start a conversation by attaching the `agent_rules` prompt (Claude Desktop lists MCP prompts under the
   attachment menu), then ask questions as normal.

Trade-offs to know about:

- **No evaluation.** `abe eval` needs a model ABE can call directly, so a subscription-only agent can't be
  scored by the harness. Keep a small API balance, or use Ollama, for evaluation runs.
- **Rules are advisory.** The client's model receives the rules as a prompt and may not follow them as strictly
  as the `ask` path, which enforces the decline threshold in code before any model is called.
- **Plan limits apply.** Usage counts against your subscription's limits.

GitHub Models, which once offered a free API with a GitHub token, was retired on 30 July 2026 and is not an
option.

## Using OpenRouter credits

OpenRouter sells prepaid credits that work across models from many providers through one OpenAI-compatible
API. ABE talks to it with the same code it uses for OpenAI and Ollama.

1. Create an account at [openrouter.ai](https://openrouter.ai) and buy credits.
2. Create an API key under **Keys**. Setting a credit limit on the key caps what ABE can spend.
3. Set the key in your shell:

   ```
   $env:OPENROUTER_API_KEY = "sk-or-..."      # PowerShell
   export OPENROUTER_API_KEY=sk-or-...        # bash / zsh
   ```

4. In step 5 of the form, choose **OpenRouter** and enter the model ID exactly as OpenRouter's model page shows
   it, including the provider prefix, for example `openai/gpt-5.2`. The endpoint and key variable fill in
   automatically.

To configure it by hand, the provider block in `agent.yaml` or a blueprint looks like this:

```yaml
provider:
  kind: openai_compatible
  endpoint: https://openrouter.ai/api/v1
  model: openai/gpt-5.2
  api_key_env: OPENROUTER_API_KEY
  extra_headers:
    HTTP-Referer: https://github.com/wolfmann1/ai-abe
    X-OpenRouter-Title: ai-abe
```

The two `extra_headers` are optional. OpenRouter uses them to show which app made the requests in your
activity log; change them to your own app's URL and name.

Comparing models is where OpenRouter earns its place. Change `model`, run `abe eval`, and compare the reports:
accuracy, which cases failed, and the token counts that drive `cost.md`. OpenRouter's activity page shows what
each run actually cost. Some models are listed with a `:free` suffix; they cost nothing but have tight rate
limits, which an eval run can hit.

## Local models with Ollama

Ollama runs models on your own computer, so nothing leaves it and there is nothing to pay per question.

1. Install Ollama from [ollama.com](https://ollama.com) and start it. ABE looks for it at
   `http://localhost:11434`, or at the address in the `OLLAMA_HOST` environment variable if you've set one.
2. In step 5 of the form, choose **Local model (Ollama)**. A panel lists every model Ollama has installed;
   click one to use it. The model box also offers them as suggestions.
3. If the model you want isn't listed, either:
   - install it yourself with `ollama pull <model>` and reload the form, or
   - type its name in the model box. When you build the agent, ABE asks Ollama to download it, and the agent's
     page shows the progress and refreshes until it finishes. Models are several gigabytes, so expect minutes
     rather than seconds.
4. If the download fails, usually because the name is misspelt or the network dropped, the agent's page shows
   Ollama's error and the command to run instead: `ollama pull <model>`. Run it in a terminal and reload the
   page. A **Try the download again** button is there too.

Model names come from [ollama.com/library](https://ollama.com/library). A name without a tag, such as
`llama3.1`, means `llama3.1:latest`; add a tag such as `qwen2.5:7b` to choose a specific size. Smaller models
answer faster and fit on more graphics cards; larger ones usually score better in `abe eval`. To switch models
later, change `model` in the agent's `agent.yaml`; the index doesn't need rebuilding.

## Azure OpenAI on an existing Azure subscription

For an organization already on Azure, this keeps model spend on the existing Azure bill and the data inside
the tenant. Create an Azure OpenAI resource, deploy a model, and enter the resource URL and deployment name in
step 5. The key is on the resource's **Keys and Endpoint** page; set it as `AZURE_OPENAI_API_KEY`.
