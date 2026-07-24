# What's new in Claude Opus 5

Overview of new features and capabilities in Claude Opus 5, and how this pipe handles them.

---

Claude Opus 5 is Anthropic's Opus-tier model for complex agentic coding and enterprise work, and a step-change over Opus 4.8 on deep reasoning, agentic and long-horizon work. It ships at Opus 4.8's pricing.

## New model

| Model | API model ID | Description |
|:------|:-------------|:------------|
| Claude Opus 5 | `claude-opus-5` | Complex agentic coding, long-horizon and enterprise work |

`claude-opus-5` is a dateless pinned snapshot (there is no dated `YYYYMMDD` form). On Amazon Bedrock the ID is `anthropic.claude-opus-5`; on Vertex AI it is `claude-opus-5`.

Claude Opus 5 supports a **1M token context window** (the default *and* the maximum), **128K max output tokens**, adaptive thinking, vision, and all the tools this pipe uses (web search, web fetch, code execution).

## Pricing

Identical to Opus 4.8 — half of Fable 5:

| | Input | Output | 5m cache write | 1h cache write | Cache hit |
|:--|:--|:--|:--|:--|:--|
| Claude Opus 5 | $5 / MTok | $25 / MTok | $6.25 / MTok | $10 / MTok | $0.50 / MTok |

Web search is billed at $10 per 1,000 searches. Code execution is free when used with web search or web fetch in the same request.

> **Rate limits are a separate bucket.** Opus 4.8/4.7/4.6/4.5 share one combined Opus limit; Claude Opus 5 does not draw from it. Check your tier's Claude Opus 5 limits before shifting traffic over.

## Thinking is on by default

This is the first Claude model where **omitting the `thinking` parameter still thinks**. On Opus 4.8 and 4.7, a request with no `thinking` field ran without thinking; on Opus 5 the same request runs adaptive thinking (`{"type": "adaptive"}` is equivalent to omitting it).

Two consequences the pipe handles:

- **`ENABLE_THINKING = false` must be sent explicitly.** The pipe now sends `thinking: {"type": "disabled"}` for `claude-opus-5` when thinking is turned off, instead of just leaving the parameter out (which would silently keep thinking on and bill for it).
- **`MAX_TOKENS` caps thinking *plus* response text.** A budget sized tightly around the answer can truncate mid-response now that thinking is on. Raise `MAX_TOKENS` (requests above 21,333 are automatically forced to stream) or turn thinking off.

Manual thinking (`{"type": "enabled", "budget_tokens": N}`) is **rejected with a 400**, as on Fable 5 / Sonnet 5 / Opus 4.8. The pipe routes every adaptive-thinking model through `{"type": "adaptive"}`.

```python
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    thinking={"type": "adaptive", "display": "summarized"},
    messages=[{"role": "user", "content": "..."}],
)
```

### Disabling thinking is capped at `high` effort

`thinking: {"type": "disabled"}` is accepted **only at effort `high` or below** — pairing it with `xhigh` or `max` returns a 400, and the check runs per request. The pipe only sends `output_config.effort` alongside adaptive thinking, so a disabled-thinking request always runs at the API default (`high`) and can't hit this.

(Fable 5 differs: it rejects `{"type": "disabled"}` at *any* effort, so the pipe leaves the parameter out entirely there and logs that thinking cannot be turned off.)

### `thinking.display` defaults to `"omitted"`

As on Fable 5 / Sonnet 5 / Opus 4.8, thinking blocks come back with an **empty** `thinking` field unless `display: "summarized"` is set. The pipe defaults `display` to `"summarized"` via the `THINKING_DISPLAY` valve so reasoning is visible inside `<think>` blocks; set the valve to `"omitted"` for lower streaming latency (you are billed for thinking tokens either way). The raw chain of thought is never returned on Opus 5 — only the summary.

## Effort

Claude Opus 5 supports the full ladder — `low`, `medium`, `high`, `xhigh`, `max` — with no beta header. The API default is `high`.

Anthropic's guidance: start at `xhigh` for coding and agentic work and `high` elsewhere, then sweep downward — `low` and `medium` are unusually strong on this model and are the primary cost/latency lever. Effort defaults carried over from an older model rarely transfer.

The pipe exposes this via the `EFFORT` valve (empty = API default). It is sent through `extra_body.output_config.effort` so it works regardless of the installed Anthropic SDK version, and only on adaptive-thinking models. `xhigh` is ignored (with a log) on Opus 4.6 / Sonnet 4.6, which support `low`/`medium`/`high`/`max` only.

## Refusals are a normal response, not an error

Claude Opus 5 ships with elevated cybersecurity safeguards, and its classifiers can decline a request. That comes back as a **successful HTTP 200** with `stop_reason: "refusal"`, a `stop_details` category, and an empty (or partial) content list — not an exception.

The pipe checks `stop_reason` in both the streaming and non-streaming paths and appends a notice naming the refusal category, so a declined request no longer renders as a blank reply. Any partial output that was already streamed is kept.

Server-side refusal fallbacks (`fallbacks`, beta `server-side-fallback-2026-07-01`) are not used by this pipe — pick a different model from the model list if a request is declined.

## Prompt caching

The minimum cacheable prompt length drops to **512 tokens** on Opus 5 (1,024 on Opus 4.8), so shorter conversations start benefiting from the pipe's automatic prompt caching without any configuration change.

## Tools

The pipe keeps the GA tool versions it already uses on every model:

- **Web search** — `web_search_20250305`
- **Web fetch** — `web_fetch_20250910`
- **Code execution** — `code_execution_20250825`

Opus 5 also supports the newer `web_search_20260209` / `web_fetch_20260209` variants (dynamic filtering, where the model filters results with code before they reach the context window). Those are not adopted here because the pipe sends one tool version to every model; adopting them would require per-model tool selection.

## Sampling parameters

`temperature`, `top_p`, and `top_k` are rejected on Opus 5 (as on Fable 5 / Sonnet 5 / Opus 4.8). The pipe never sets them, so no action is required.

## Not enabled by this pipe

- **Fast mode** (`speed: "fast"`, beta `fast-mode-2026-02-01`) — research preview at $10/$50 per MTok, Claude API only.
- **Task budgets** (`output_config.task_budget`, beta `task-budgets-2026-03-13`).
- **Mid-conversation tool changes** (beta `mid-conversation-tool-changes-2026-07-01`).

## Deprecations / breaking changes

- Thinking is **on by default** when `thinking` is omitted (changed from Opus 4.8/4.7) — handled by the pipe.
- `thinking: {"type": "disabled"}` returns a 400 at effort `xhigh` / `max`.
- Manual `budget_tokens` thinking is rejected (inherited from Opus 4.8/4.7).
- Prefilling assistant messages returns a 400. This pipe does not prefill.

## Migration guide

For step-by-step migration instructions, see [Anthropic's model migration guide](https://platform.claude.com/docs/en/about-claude/models/migration-guide).
