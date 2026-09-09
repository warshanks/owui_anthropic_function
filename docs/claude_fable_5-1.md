# What's new in Claude Fable 5.1

Overview of new features and capabilities in Claude Fable 5.1, and how this pipe handles them.

---

Claude Fable 5.1 (released 1 September 2026) is Anthropic's most capable widely released model, for demanding reasoning and long-horizon agentic work. It succeeds Claude Fable 5 at the same input and output prices, with cache reads at a quarter of the cost. Claude Opus 5 remains the recommended starting point for most workloads; reach for Fable 5.1 when Opus 5 at higher effort still falls short.

## New model

| Model | API model ID | Description |
|:------|:-------------|:------------|
| Claude Fable 5.1 | `claude-fable-5-1` | Demanding reasoning and long-horizon agentic work |

`claude-fable-5-1` is a dateless pinned snapshot. On Amazon Bedrock the ID is `anthropic.claude-fable-5-1`; on Google Cloud and Microsoft Foundry it is `claude-fable-5-1`.

Claude Fable 5.1 supports a **1M token context window** (the default *and* the maximum), **128K max output tokens**, adaptive thinking, vision, and all the tools this pipe uses (web search, web fetch, code execution). It shares the Claude Opus 4.7 tokenizer with Fable 5, so token counts carry over unchanged from Fable 5.

Claude Mythos 5.1 (`claude-mythos-5-1`) has the same capabilities and pricing but is limited to Project Glasswing participants, so the pipe does not list it.

## Pricing

Identical to Fable 5 except for cache reads:

| | Input | Output | 5m cache write | 1h cache write | Cache hit |
|:--|:--|:--|:--|:--|:--|
| Claude Fable 5.1 | $10 / MTok | $50 / MTok | $12.50 / MTok | $20 / MTok | $0.25 / MTok |

Cache hits and refreshes cost **0.025x** the base input price on this model, against 0.1x everywhere else. Long chats that re-read a cached prefix pay a quarter of the Fable 5 rate. The pipe's cost tracking applies the reduced multiplier for `claude-fable-5-1` only; cache writes and the 512-token minimum cacheable prompt length are unchanged.

Web search is billed at $10 per 1,000 searches. Code execution is free when used with web search or web fetch in the same request.

## Thinking is always on

As on Fable 5, adaptive thinking cannot be turned off. Both `{"type": "disabled"}` and the manual `{"type": "enabled", "budget_tokens": N}` form return a 400 — omit the parameter or send `{"type": "adaptive"}`. Depth is steered by `output_config.effort`, which defaults to `high` and accepts the full `low`-`max` range including `xhigh`.

The pipe treats `claude-fable-5-1` the same way it treats `claude-fable-5`: `ENABLE_THINKING = false` logs that thinking cannot be disabled and leaves it on. `thinking.display` defaults to `"omitted"` at the API level, so the pipe sets `"summarized"` (via the `THINKING_DISPLAY` valve) to keep reasoning visible in `<think>` blocks.

Because thinking always runs, `MAX_TOKENS` has to cover thinking plus the answer. Requests above 21,333 tokens are automatically forced to stream.

## Thinking blocks are bound to the conversation prefix

This is the change that matters most for a chat front end. Every Fable 5.1 thinking block records the conversation prefix that produced it. Editing anything *before* a thinking block — an earlier message, the system prompt, or the tool list — invalidates that block and every block after it. Replaying an invalidated block returns a 400 whose message reads `The block is bound to a different conversation`. The check is enforced for accounts created on or after 31 August 2026; older accounts record the mismatch but act on it only when asked to.

Open WebUI rebuilds the message array on every turn and lets users edit, branch, and regenerate earlier messages, all of which change the prefix. So for models in `PREFIX_BINDING_MODELS` the pipe sends the `thinking-binding-controls-2026-08-01` beta header with:

```python
params["thinking"]["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
```

The API then drops invalidated blocks and answers the request instead of rejecting it. A dropped block costs nothing — it is not billed and does not count toward `input_tokens`. Reasoning continuity is preserved whenever the history is untouched, which is the common case.

Thinking blocks are also preserved in one direction only: Fable 5.1 reads earlier models' thinking blocks, but no earlier model reads Fable 5.1's. Switching a chat from Fable 5.1 back to Opus 5 silently drops the reasoning for turns that run on the older model.

## Forced tool use is rejected

`tool_choice` of `{"type": "any"}` or `{"type": "tool", "name": "..."}` returns a 400 on Fable 5.1. `{"type": "auto"}` (the default) and `{"type": "none"}` are unaffected. The pipe never sets `tool_choice`, so nothing changes here — but it rules out adding forced tool use later without a per-model guard.

## Server tools

Unchanged from Fable 5, and all supported:

- **Web search** — `web_search_20260318` with dynamic filtering
- **Web fetch** — `web_fetch_20260318` with dynamic filtering
- **Code execution** — `code_execution_20260521`

Dynamic filtering runs inside a code execution sandbox the API provisions itself, so the pipe still falls back to the basic `web_search_20250305` / `web_fetch_20250910` versions when the user turns on Open WebUI's code interpreter.

## Refusals

Fable 5.1 runs the same safety classifiers as Fable 5 and can return `stop_reason: "refusal"` on an HTTP 200 with an empty or partial content list. The pipe's existing refusal handling surfaces a notice with the category instead of a blank reply. A refusal that arrives before any output is not billed.

## Behaviour differences worth knowing

These show up without any code change, and each has a prompting fix in Anthropic's [Prompting Claude Fable 5.1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1) guide:

- **Fewer progress updates during long tool runs**, especially at higher effort.
- **More variable parallel tool calling** — one call per turn where Fable 5 batched several.
- **Answers from memory more often at `low` effort**, calling search or retrieval less.
- **Denser prose and less chat formatting** than earlier Claude models, so anti-formatting system prompts written for older models can suppress structure the content needs.

## Availability

Claude Fable 5.1 carries 30-day data retention and is not available under zero data retention unless expressly authorised by Anthropic. Requests from an organisation whose retention configuration does not meet the requirement return a 400.

## References

- [Claude Fable 5.1 overview](https://platform.claude.com/docs/en/models/fable-5-1/overview)
- [What's new in Claude Fable 5.1](https://platform.claude.com/docs/en/models/fable-5-1/whats-new-fable-5-1)
- [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/thinking#preserved-thinking)
- [Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
