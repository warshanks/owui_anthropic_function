# Anthropic Manifold Pipe for Open WebUI

![Version](https://img.shields.io/badge/version-0.16.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This pipe provides seamless integration with Anthropic's Claude models for Open WebUI, enabling advanced capabilities like web search, secure code execution, and extended thinking.

## Features

- **Web Search**: Enable Claude to search the web for real-time information. On Claude 4.6+ models the pipe uses the dynamic-filtering tool version, so Claude filters results with code before they reach the context window (fewer input tokens on search-heavy turns).
- **Web Fetch**: Fetch and process content from specific URLs for deeper analysis, with the same dynamic filtering on Claude 4.6+ models.
- **Code Execution**: Run Python code in Anthropic's secure sandbox environment for calculations, data analysis, and more.
- **Extended Thinking**: Leverage Claude's extended thinking for complex problem-solving. Newer models (Opus 5, Fable 5, Sonnet 5, Opus 4.8/4.6, Sonnet 4.6) use **adaptive thinking** (Claude decides when and how much to think); older models use a configurable token budget. On Opus 5 thinking is on by default at the API level, so `ENABLE_THINKING=false` is sent as an explicit disabled config.
- **Effort Control**: Guide how much adaptive-thinking models reason via the `EFFORT` valve (`low`, `medium`, `high`, `xhigh`, `max`). `xhigh` is available everywhere except Opus 4.6 / Sonnet 4.6.
- **Thinking Display**: Adaptive-thinking reasoning defaults to `summarized` (shown in `<think>` blocks) via the `THINKING_DISPLAY` valve; set to `omitted` for lower latency. This overrides the API-level `omitted` default on Opus 5, Fable 5, Sonnet 5 and Opus 4.8 so it's visible the model thought.
- **Refusal Handling**: Requests declined by Anthropic's safety classifiers (`stop_reason: "refusal"`, returned as a normal HTTP 200) surface a notice with the refusal category instead of an empty reply.
- **Prompt Caching**: Automatic caching of the conversation prefix (`ENABLE_PROMPT_CACHING` valve, on by default). Follow-up turns re-read prior history at ~10% of the input price instead of reprocessing it in full. The `CACHE_TTL` valve selects a `5m` (default) or `1h` cache lifetime.
- **Image Processing**: Analyze images with support for both URL and base64 inputs (up to 5MB).
- **Streaming Support**: Real-time streaming of responses, including thinking blocks and code execution outputs.
- **Cost Tracking**: Track and display the cost of requests in real-time.
- **Flexible Configuration**: Manage capabilities globally or per-user via Valves.

## Supported Models

The pipe automatically handles capabilities for various Claude models, including:

- **Claude Opus** (`claude-opus-5`, `claude-opus-4-8`, `claude-opus-4-6`, `claude-opus-4-5-20251101`)
- **Claude Fable** (`claude-fable-5`)
- **Claude Sonnet** (`claude-sonnet-5`, `claude-sonnet-4-6`, `claude-sonnet-4-5-20250929`)
- **Claude Haiku** (`claude-haiku-4-5-20251001`)

Claude Opus 5 (`claude-opus-5`) is the default flagship: a 1M-token context window and 128K max output, **adaptive thinking on by default**, the full `low`-`max` effort range, and Opus 4.8 pricing ($5/$25 per MTok). See [What's new in Claude Opus 5](docs/claude_opus_5.md).

Claude Fable 5 (`claude-fable-5`) is the highest-capability tier ($10/$50 per MTok) and always thinks — the API rejects disabling it. Claude Opus 4.8 remains available; see [What's new in Claude 4.8](docs/claude_4-8.md).

## Configuration

You can configure the pipe using **Valves**. These can be set globally by the admin or individually by users (if allowed).

## Installation

1. Open your Open WebUI instance.
2. Go to **Admin Panel** -> **Functions**.
3. Click the **+ New Function** button.
4. Import the `anthropic_manifold.py` file or paste its content.
5. Save and activate the function.

## Usage

### Extended Thinking
When using a supported model, the model may utilize "thinking" blocks to reason through complex problems before answering. These blocks are displayed as:
```
<think>
... reasoning process ...
</think>
```
*Note: On budget-based models (e.g. Opus 4.5, Sonnet 4.5), thinking requires a minimum budget of 1,024 tokens and the budget must be less than `MAX_TOKENS`. On adaptive-thinking models (Opus 5, Fable 5, Sonnet 5, Opus 4.8/4.6, Sonnet 4.6), the budget is ignored — Claude allocates thinking automatically, optionally guided by the `EFFORT` valve. On Opus 5, Fable 5, Sonnet 5 and Opus 4.8, adaptive is the only thinking mode (manual budgets are rejected by the API). Because Opus 5 thinks by default, its `MAX_TOKENS` has to cover thinking plus the answer.*

### Code Execution
If enabled, Claude can write and execute Python code. The code and its output (stdout/stderr) will be displayed in the chat:
```python
print("Hello World")
```
**Output:**
```
Hello World
```

### Prompt Caching
Enabled by default. Each request caches the conversation up to the newest message; the next turn in the same chat reads that prefix from cache at ~10% of the input price (cache writes cost 1.25x, so caching pays for itself from the second turn onward). Cache hits require the prefix to be byte-identical, so toggling web search / code execution / thinking or switching models mid-chat starts a fresh cache. Very short conversations below the model's minimum cacheable length (~0.5-4K tokens depending on model; 512 tokens on Opus 5) are processed normally without caching. Cache read/write token counts appear in the usage stats and are included in the cost calculation.

### Web Search
Claude can perform web searches to fetch up-to-date information when enabled via the UI's web search toggle.

On Claude 4.6 and later models (Opus 5, Fable 5, Sonnet 5, Opus 4.8/4.6, Sonnet 4.6) the pipe sends `web_search_20260318` / `web_fetch_20260318`, which filter results with code before they enter the context window. That filtering runs in a sandbox the API provisions itself, so it can't be combined with an explicitly declared code execution tool: if you also turn on the code interpreter, the request falls back to the basic `web_search_20250305` / `web_fetch_20250910` versions. Older models always use the basic versions. The filtering code is never rendered into the chat — code execution blocks are only shown when you enabled code execution yourself.

### Web Fetch
When enabled via the URL context toggle in the UI, Claude can fetch and analyze content from specific URLs. This is useful for reading documentation, analyzing web pages, or extracting information from specific websites.

Uses [suurt8ll's gemini_url_context_toggle filter](https://github.com/suurt8ll/open_webui_functions/blob/master/plugins/filters/gemini_url_context_toggle.py) to enable in the UI.

## License

MIT License - see the [LICENSE](LICENSE) file for details.
