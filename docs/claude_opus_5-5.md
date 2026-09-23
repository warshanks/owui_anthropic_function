# What's new in Claude Opus 5.5

Overview of new features and capabilities in Claude Opus 5.5, and how this pipe handles them.

---

Claude Opus 5.5 (released 22 September 2026) succeeds Claude Opus 5 in the Opus line for long-running agentic coding and knowledge work, **at a lower price**. Anthropic now recommends it as the starting point for most workloads; Claude Fable 5.1 remains the highest-capability tier for demanding reasoning and long-horizon agentic work.

## New model

| Model | API model ID | Description |
|:------|:-------------|:------------|
| Claude Opus 5.5 | `claude-opus-5-5` | Long-running agentic coding and knowledge work |

`claude-opus-5-5` is a dateless pinned snapshot. On Amazon Bedrock the ID is `anthropic.claude-opus-5-5`; on Google Cloud, Microsoft Foundry, and Claude Platform on AWS it is `claude-opus-5-5`.

Claude Opus 5.5 supports a **1M token context window** (the default *and* the maximum), **128K max output tokens**, adaptive thinking, vision, and all the tools this pipe uses (web search, web fetch, code execution). It uses the same tokenizer as Claude Opus 5, so token counts carry over unchanged. Claude Opus 5 stays available.

## Pricing

20% cheaper than Opus 5 per token, and 60% cheaper on cache reads:

| | Input | Output | 5m cache write | 1h cache write | Cache hit |
|:--|:--|:--|:--|:--|:--|
| Claude Opus 5.5 | $4 / MTok | $20 / MTok | $5 / MTok | $8 / MTok | $0.20 / MTok |

Cache hits and refreshes cost **0.05x** the base input price on this model, against 0.1x on most models (0.025x on Fable 5.1). The pipe's cost tracking applies the reduced multiplier for `claude-opus-5-5`. Batch pricing is $2 / $10 per MTok, and fast mode (not used by this pipe) is $8 / $40.

Web search is billed at $10 per 1,000 searches. Code execution is free when used with web search or web fetch in the same request.

## Thinking is always on, and effort defaults to `medium`

Unlike Opus 5, thinking cannot be disabled at any effort level: both `{"type": "disabled"}` and `{"type": "enabled", "budget_tokens": N}` return a 400. The pipe treats `claude-opus-5-5` like the Fable models: `ENABLE_THINKING = false` logs that thinking cannot be disabled and leaves it on. To make it cheaper or faster, lower the `EFFORT` valve instead.

The API default effort is **`medium`** (one level below Opus 5's `high`). Leaving the `EFFORT` valve empty uses that default; set it to `high` or `xhigh` for harder work.

`thinking.display` defaults to `"omitted"` at the API level, so the pipe sets `"summarized"` (via the `THINKING_DISPLAY` valve) to keep reasoning visible in `<think>` blocks. Because thinking always runs, `MAX_TOKENS` has to cover thinking plus the answer.

## Thinking blocks are bound to the conversation prefix

Opus 5.5 uses the same "preserved thinking" mechanism as Fable 5.1: editing anything before a thinking block (an earlier message, the system prompt, or the tool list) invalidates it, and replaying an invalidated block returns a 400 on accounts created on or after 31 August 2026. Since Open WebUI lets users edit, branch, and regenerate earlier messages, the pipe adds `claude-opus-5-5` to `PREFIX_BINDING_MODELS`, sending the `thinking-binding-controls-2026-08-01` beta header with:

```python
params["thinking"]["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
```

The API then drops invalidated blocks (unbilled) and answers the request instead of rejecting it.

Model binding: Opus 5.5 reads thinking blocks from Opus 5 and earlier Opus / Sonnet / Haiku models, but not from Fable or Mythos models. Going the other way, only Fable 5.1 (on the Claude API) reads Opus 5.5's blocks, so switching a chat from Opus 5.5 to another model continues without the earlier reasoning. The API drops what the target model can't read, and the request still succeeds.

## Forced tool use is rejected

`tool_choice` of `{"type": "any"}` or `{"type": "tool", ...}` returns a 400. The pipe never sets `tool_choice`, so nothing changes.

## Refusals

Opus 5.5 runs cybersecurity **and biology** safety classifiers, and can also decline requests that try to get the model to reproduce its internal reasoning (`reasoning_extraction`). Declines come back as `stop_reason: "refusal"` on an HTTP 200; the pipe's existing refusal handling shows a notice with the category instead of a blank reply.

## Progress updates between tool calls

On longer agentic turns, Opus 5.5 (like Fable 5.1) returns the notes it writes between tool calls as `thinking` blocks instead of `text`. With the pipe's default `THINKING_DISPLAY = summarized` those notes are shown in the `<think>` blocks.

## References

- [Claude Opus 5.5 overview](https://platform.claude.com/docs/en/models/opus-5-5/overview)
- [Migrating to Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)
- [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/thinking#preserved-thinking)
- [Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
