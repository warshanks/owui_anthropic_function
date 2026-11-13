# What's new in Claude Haiku 5.5

Overview of new features and capabilities in Claude Haiku 5.5, and how this pipe handles them.

---

Claude Haiku 5.5 (released 7 October 2026) succeeds Claude Haiku 4.5 as the fastest, lowest-cost tier, built for high-volume, latency-sensitive work such as classification, routing, extraction, and subagent tasks. It is the first Haiku model with adaptive thinking and the effort parameter.

## New model

| Model | API model ID | Description |
|:------|:-------------|:------------|
| Claude Haiku 5.5 | `claude-haiku-5-5` | High-volume, latency-sensitive tasks |

`claude-haiku-5-5` is a dateless pinned snapshot. On Amazon Bedrock the ID is `anthropic.claude-haiku-5-5`; on Google Cloud, Microsoft Foundry, and Claude Platform on AWS it is `claude-haiku-5-5`.

Claude Haiku 5.5 supports a **1M token context window** and **128K max output tokens** (up from 200K and 64K on Haiku 4.5), adaptive thinking, vision, and all the tools this pipe uses (web search, web fetch, code execution), including the dynamic-filtering web tools that Haiku 4.5 doesn't support. It uses the tokenizer introduced with Claude Opus 4.7, so the same text counts as about 30% more tokens than on Haiku 4.5. Claude Haiku 4.5 stays available.

## Pricing: two rate cards

Haiku 5.5 is priced by prompt length. A request whose prompt is **over 100,000 tokens** pays the higher prices for the whole request, output included:

| Prompt length | Input | Output | 5m cache write | 1h cache write | Cache hit |
|:--|:--|:--|:--|:--|:--|
| Up to 100,000 tokens | $0.10 / MTok | $0.50 / MTok | $0.125 / MTok | $0.20 / MTok | $0.01 / MTok |
| Over 100,000 tokens | $0.50 / MTok | $2.50 / MTok | $0.625 / MTok | $1 / MTok | $0.05 / MTok |

The prompt length counts all input tokens, **including cache reads and cache writes**, so a long chat crosses the threshold even when most of its prompt is a cache hit. Each request is priced on its own. Cache hits are the standard 0.1x the input price, and cache writes the standard 1.25x (5 minutes) or 2x (1 hour).

The pipe's cost tracking adds the cached prefix to the uncached input to get the prompt length, and picks the rate card from it (`LONG_PROMPT_PRICING`). Batch pricing is half of each card. The minimum cacheable prompt is 512 tokens, down from 4,096 on Haiku 4.5, so short chats now cache too.

Web search is billed at $10 per 1,000 searches. Code execution is free when used with web search or web fetch in the same request.

## Thinking is on by default

Haiku 4.5 only thinks with a manual budget. Haiku 5.5 runs **adaptive thinking when `thinking` is omitted**, and the manual `{"type": "enabled", "budget_tokens": N}` form returns a 400. Depth is steered by `output_config.effort`, which supports `low` through `max` and defaults to **`medium`**, as on Opus 5.5. Thinking text is `"omitted"` by default at the API level, so the pipe sets `"summarized"` (via the `THINKING_DISPLAY` valve) to keep reasoning visible in Open WebUI's Thoughts section.

With `ENABLE_THINKING = true` the pipe sends adaptive thinking and the `EFFORT` valve, as on other adaptive models. With `ENABLE_THINKING = false` it sends:

```python
params["thinking"] = {"type": "disabled"}
```

Haiku 5.5 accepts `disabled` only at effort `high` or below. The pipe sends effort only alongside adaptive thinking, so the API default (`medium`) applies here.

Because thinking is on by default, `MAX_TOKENS` has to cover thinking plus the answer.

## Thinking blocks are bound to the conversation prefix

Haiku 5.5 runs the same prefix check as Fable 5.2, Fable 5.1, Opus 5.5, and Sonnet 5.5: editing anything before a thinking block (an earlier message, the system prompt, or the tool list) invalidates it, and replaying an invalidated block returns a 400 on accounts created on or after 31 August 2026. The pipe adds `claude-haiku-5-5` to `PREFIX_BINDING_MODELS`, sending the `thinking-binding-controls-2026-08-01` beta header with:

```python
params["thinking"]["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
```

The API then drops invalidated blocks (unbilled) and answers the request instead of rejecting it.

As on Sonnet 5.5, `block_binding` works only with adaptive thinking: sending it with `{"type": "disabled"}` is a 400. So with `ENABLE_THINKING = false` the pipe sends no `block_binding` and doesn't replay thinking blocks at all, stripping them from the history it sends. An edited or regenerated chat then can't trip the prefix check, at the cost of the model not seeing its earlier reasoning on later requests (it stays visible in Open WebUI's Thoughts section).

Model binding: Haiku 5.5 reads thinking blocks from Sonnet 5, Opus 4.8, Haiku 4.5, and earlier models, but not from Opus 5, Opus 5.5, Sonnet 5.5, or any Fable or Mythos model. Opus 5.5, Sonnet 5.5, and Fable 5.2 read Haiku 5.5's blocks; no other model does. Blocks are also tied to the account that produced them.

## Other API changes from Haiku 4.5

- **Sampling parameters**: any non-default `temperature`, `top_p`, or `top_k` returns a 400. The pipe sends none.
- **Assistant prefill** returns a 400. The pipe never sends a trailing assistant turn of its own.
- **Forced tool use** (`tool_choice` `any` / `tool`) is still accepted, unlike on the other current models, but skips thinking. The pipe never sets `tool_choice`.
- **Earlier thinking blocks stay in context**: Haiku 5.5 keeps thinking blocks from every earlier turn in context (Haiku 4.5 kept only the latest), so multi-turn chats carry more input tokens.

## Refusals

Haiku 5.5 runs safety classifiers that can decline a request (new compared with Haiku 4.5), in four categories: `cyber`, `bio`, `frontier_llm`, and `general_harms`. Declines come back as `stop_reason: "refusal"` on an HTTP 200; the pipe's existing refusal handling shows a notice with the category instead of a blank reply. There is no server-side fallback for this model.

## Behavior differences worth knowing

From Anthropic's [Prompting Claude Haiku 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-haiku-5-5) guide:

- **With a search tool, give today's date** in the system prompt. Without it, Haiku 5.5 searches less often for facts that may have changed since its training data.
- **Empty replies at `xhigh`**: in multi-turn chats at `xhigh` effort, the model sometimes puts its whole answer in its thinking and returns no visible text.
- **Lower effort, not "answer directly" prompts**, is the way to get less thinking.

## References

- [Claude Haiku 5.5 overview](https://platform.claude.com/docs/en/models/haiku-5-5/overview)
- [What's new in Claude Haiku 5.5](https://platform.claude.com/docs/en/models/haiku-5-5/whats-new-haiku-5-5)
- [Migrating to Claude Haiku 5.5](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide)
- [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)
- [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
