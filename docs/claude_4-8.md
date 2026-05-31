# What's new in Claude 4.8

Overview of new features and capabilities in Claude Opus 4.8.

---

Claude Opus 4.8 (released May 28, 2026) is Anthropic's most intelligent model for building agents and coding. This page summarizes what's new and how this pipe handles it.

## New model

| Model | API model ID | Description |
|:------|:-------------|:------------|
| Claude Opus 4.8 | `claude-opus-4-8` | Most intelligent model for agents and coding |

The model ID `claude-opus-4-8` is a dateless pinned snapshot (the alias and the ID are the same — there is no dated `YYYYMMDD` form). On Amazon Bedrock the ID is `anthropic.claude-opus-4-8`; on Vertex AI it is `claude-opus-4-8`.

Claude Opus 4.8 supports a **1M token context window by default** (200K on Microsoft Foundry), **128K max output tokens** on the synchronous Messages API, extended thinking via adaptive mode, vision, and all existing tools (web search, web fetch, code execution).

For complete pricing and specs, see the [models overview](https://platform.claude.com/docs/en/about-claude/models/overview).

## Pricing

Pricing is identical to Opus 4.6/4.7:

| | Input | Output | 5m cache write | 1h cache write | Cache hit | Batch input | Batch output |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Claude Opus 4.8 | $5 / MTok | $25 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok | $2.50 / MTok | $12.50 / MTok |

Web search is billed at $10 per 1,000 searches. Code execution is **free when used with web search or web fetch** in the same request.

## Adaptive thinking is the only thinking mode

On Opus 4.8, [adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking) (`thinking: {type: "adaptive"}`) is the **only** supported thinking mode. Thinking is off unless explicitly enabled. Manual `thinking: {type: "enabled", budget_tokens: N}` is **rejected with a 400 error**.

This pipe automatically sends adaptive thinking for `claude-opus-4-8` (alongside `claude-opus-4-6` and `claude-sonnet-4-6`) when thinking is enabled, instead of the deprecated budget-based mode.

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    thinking={"type": "adaptive", "display": "summarized"},
    messages=[{"role": "user", "content": "Solve this complex problem..."}],
)
```

### `thinking.display` defaults to `"omitted"`

This is a **silent change from Opus 4.6**. On Opus 4.8/4.7, `thinking.display` defaults to `"omitted"`: thinking blocks are returned with an **empty** `thinking` field (the `signature` field still carries the encrypted thinking for multi-turn continuity). To stream/show summarized thinking text — which this pipe renders inside `<think>` blocks — you must set `display: "summarized"` explicitly.

This pipe defaults `display` to `"summarized"` for all adaptive models (via the `THINKING_DISPLAY` valve) so the model's reasoning is visible in `<think>` blocks and it's clear the model thought. Set the valve to `"omitted"` to hide the reasoning text for lower streaming latency — the `<think>` block still appears (empty) and you are still billed for thinking tokens. (On 4.6, `"summarized"` is already the API default, so this is harmless.)

### Effort parameter

The [effort parameter](https://platform.claude.com/docs/en/build-with-claude/effort) (`output_config.effort`) guides how much adaptive-thinking models reason. It defaults to `high` (≈ omitting it). Levels: `low`, `medium`, `high`, `xhigh`, `max`. **`xhigh` is available only on Opus 4.8 and 4.7.**

This pipe exposes effort via the `EFFORT` valve (empty = API default `high`). It is sent through `extra_body.output_config.effort`, so it works regardless of the installed Anthropic SDK version, and only on adaptive-thinking models. `xhigh` is ignored (with a log) on non-4.8/4.7 models.

```python
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    output_config={"effort": "xhigh"},
    messages=[{"role": "user", "content": "..."}],
)
```

## Tools

All tools are generally available and require **no beta header** on Opus 4.8:

- **Web search** — `web_search_20250305`
- **Web fetch** — `web_fetch_20250910`
- **Code execution** — `code_execution_20250825` (Bash + file operations)

This pipe keeps these GA tool versions. Newer `web_search_20260209` / `web_fetch_20260209` versions add dynamic filtering but **require the `code_execution` tool to be enabled in the same request**, so they are intentionally not adopted by default. Code execution is free when bundled with a web tool.

## Sampling parameters

On Opus 4.8 (Messages API), `temperature`, `top_p`, and `top_k` must be left at default — any non-default value returns a 400 error. This pipe does not set them, so no action is required.

## Other launch features

- **1M token context window** by default (Claude API / Bedrock / Vertex).
- **128K max output tokens** (synchronous Messages API). The pipe's `MAX_TOKENS` valve can be raised accordingly; requests with `max_tokens` > 21,333 are automatically forced to stream.
- **Minimum cacheable prompt length** lowered to 1,024 tokens.
- **Mid-conversation system messages** and a **refusal `stop_details` object** (no beta header).
- **Fast mode** (research preview, `speed: "fast"`, premium pricing) — not enabled by this pipe.

## Deprecations / breaking changes

- Manual extended thinking (`type: "enabled"` + `budget_tokens`) is **rejected** on Opus 4.8/4.7 (still functional but deprecated on 4.6 / Sonnet 4.6).
- Prefilling assistant messages returns a 400 error on Opus 4.8 (inherited from 4.6/4.7). This pipe does not prefill.
- `thinking.display` default changed to `"omitted"` (see above) — handled by this pipe.

## Migration guide

For step-by-step migration instructions, see [Migrating to Claude 4.8](https://platform.claude.com/docs/en/about-claude/models/migration-guide).
