# What's new in Claude Sonnet 5.5

Overview of new features and capabilities in Claude Sonnet 5.5, and how this pipe handles them.

---

Claude Sonnet 5.5 (released 28 September 2026) succeeds Claude Sonnet 5 as the best combination of speed and intelligence, **at the same price**. It is the fast, low-cost tier below Claude Opus 5.5 and Claude Fable 5.1.

## New model

| Model | API model ID | Description |
|:------|:-------------|:------------|
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | The best combination of speed and intelligence |

On Amazon Bedrock the ID is `anthropic.claude-sonnet-5-5`; on Google Cloud, Microsoft Foundry, and Claude Platform on AWS it is `claude-sonnet-5-5`.

Claude Sonnet 5.5 supports a **1M token context window** (the default *and* the maximum), **128K max output tokens**, adaptive thinking, vision, and the server tools this pipe uses (web search, web fetch, code execution, with the same tool versions as Sonnet 5). It uses the same tokenizer as Claude Sonnet 5, so token counts carry over unchanged. Claude Sonnet 5 stays available.

## Pricing

Unchanged from Sonnet 5:

| | Input | Output | 5m cache write | 1h cache write | Cache hit |
|:--|:--|:--|:--|:--|:--|
| Claude Sonnet 5.5 | $2 / MTok | $10 / MTok | $2.50 / MTok | $4 / MTok | $0.20 / MTok |

Cache hits and refreshes cost the standard **0.1x** the base input price, so the pipe's cost tracking needs no reduced multiplier for this model (unlike Opus 5.5 and Fable 5.1). Batch pricing is $1 / $5 per MTok. Sonnet 5.5 is not offered in fast mode, which this pipe doesn't use anyway.

The minimum cacheable prompt is 512 tokens (1,024 on Sonnet 5). Prompts below it are processed normally without caching.

Web search is billed at $10 per 1,000 searches. Code execution is free when used with web search or web fetch in the same request.

## Thinking is on by default, and `disabled` is replaced by `between_tools`

Adaptive thinking is on by default and the API default effort is **`high`**. All five levels (`low` to `max`) are available, but they're recalibrated against Sonnet 5, so an effort you tuned there won't produce the same amount of thinking here.

Unlike Sonnet 5, `{"type": "disabled"}` and `{"type": "enabled", "budget_tokens": N}` both return a 400. The lowest thinking setting is `{"type": "between_tools"}`, which turns off *up-front* thinking: the model only thinks between tool calls, and a request without tools gets a text-only reply. With `ENABLE_THINKING = false` the pipe sends:

```python
params["thinking"] = {"type": "between_tools"}
```

`between_tools` takes no other field, and it is rejected at `xhigh` and `max` effort. In this mode the pipe therefore sends no `display`, no `block_binding` (so no `thinking-binding-controls` beta header), and no `output_config.effort`, so the `EFFORT` valve has no effect and the API default (`high`) applies. With `ENABLE_THINKING = true` the pipe uses adaptive thinking as on other models, and the `EFFORT` valve works as usual. With thinking on, `MAX_TOKENS` has to cover thinking plus the answer.

## Progress updates between tool calls

Notes longer than a sentence or two that Sonnet 5.5 writes between tool calls come back as `thinking` blocks instead of `text` (shorter remarks stay `text`). At the API default `display: "omitted"` those blocks are empty, so an app that shows them goes quiet between tool calls.

With the pipe's default `THINKING_DISPLAY = summarized` the notes are streamed into the Thoughts section. With `between_tools` they come back without any `display` setting, so they are shown regardless of the valve. Setting `THINKING_DISPLAY = omitted` with adaptive thinking hides them along with the reasoning, and the Thoughts section shows a short note instead.

## Thinking blocks are bound to the conversation prefix

Sonnet 5.5 uses the same "preserved thinking" mechanism as Opus 5.5 and Fable 5.1: editing anything before a thinking block (an earlier message, the system prompt, or the tool list) invalidates it, and replaying an invalidated block returns a 400 on accounts created on or after 31 August 2026. Since Open WebUI lets users edit, branch, and regenerate earlier messages, the pipe adds `claude-sonnet-5-5` to `PREFIX_BINDING_MODELS`, sending the `thinking-binding-controls-2026-08-01` beta header with:

```python
params["thinking"]["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
```

The API then drops invalidated blocks (unbilled) and answers the request instead of rejecting it.

`block_binding` works only with adaptive thinking, and `between_tools` rejects it. So in `between_tools` mode the pipe doesn't replay thinking blocks at all: it strips them from the history it sends, following the API docs' advice to "strip the thinking blocks from the edited turn on". The pipe can't tell where an edit happened, so it strips them everywhere. An edited or regenerated chat can then never trip the prefix check, at the cost of the model not seeing its own earlier progress notes on later requests (they stay visible in Open WebUI's Thoughts section).

Model binding: Sonnet 5.5 reads thinking blocks from Sonnet 5, Opus 4.8, Haiku 4.5 and earlier models, but not from Opus 5, Opus 5.5, or any Fable or Mythos model. No other model reads Sonnet 5.5's blocks, so switching a chat from Sonnet 5.5 to another model continues without the earlier reasoning. The API drops what the target model can't read, and the request still succeeds. Blocks are also tied to the account that produced them.

## Forced tool use is rejected

`tool_choice` of `{"type": "any"}` or `{"type": "tool", ...}` returns a 400, as does a non-default `temperature`, `top_p`, or `top_k`. The pipe sets none of these, so nothing changes.

## Refusals

Sonnet 5.5's safeguards can decline a request in five categories: `cyber`, `bio`, `frontier_llm`, `reasoning_extraction`, and `general_harms`. Declines come back as `stop_reason: "refusal"` on an HTTP 200; the pipe's existing refusal handling shows a notice with the category instead of a blank reply.

## References

- [Claude Sonnet 5.5 overview](https://platform.claude.com/docs/en/models/sonnet-5-5/overview)
- [What's new in Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/whats-new-sonnet-5-5)
- [Migrating to Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide)
- [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)
- [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)
- [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
