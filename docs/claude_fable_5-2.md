# What's new in Claude Fable 5.2

Overview of new features and capabilities in Claude Fable 5.2, and how this pipe handles them.

---

Claude Fable 5.2 (released 9 November 2026) is Anthropic's most capable widely released model, for demanding reasoning and long-horizon agentic work. It succeeds Claude Fable 5.1 **at a lower price**. Claude Opus 5.5 stays the recommended starting point for most workloads; reach for Fable 5.2 when Opus 5.5 at higher effort still falls short.

## New model

| Model | API model ID | Description |
|:------|:-------------|:------------|
| Claude Fable 5.2 | `claude-fable-5-2` | Demanding reasoning and long-horizon agentic work |

`claude-fable-5-2` is a dateless pinned snapshot. On Amazon Bedrock the ID is `anthropic.claude-fable-5-2`; on Google Cloud, Microsoft Foundry (Anthropic-hosted), and Claude Platform on AWS it is `claude-fable-5-2`.

Claude Fable 5.2 supports a **1M token context window** (the default *and* the maximum), **128K max output tokens**, always-on adaptive thinking, vision, and all the tools this pipe uses (web search, web fetch, code execution). It uses the same tokenizer as Fable 5.1 and Opus 4.7 and later, so token counts carry over unchanged. Claude Fable 5.1 stays available.

Claude Mythos 5.2 (`claude-mythos-5-2`) has the same capabilities and pricing but is limited to organizations verified through Anthropic's verification programs, so the pipe does not list it.

## Pricing

Lower than Fable 5.1 on every line:

| | Input | Output | 5m cache write | 1h cache write | Cache hit |
|:--|:--|:--|:--|:--|:--|
| Claude Fable 5.2 | $8 / MTok | $40 / MTok | $10 / MTok | $16 / MTok | $0.20 / MTok |

Cache hits and refreshes cost **0.025x** the base input price, the same multiplier as Fable 5.1. The pipe's cost tracking applies it for `claude-fable-5-2`. Batch pricing is $4 / $20 per MTok. The minimum cacheable prompt is 512 tokens.

Web search is billed at $10 per 1,000 searches. Code execution is free when used with web search or web fetch in the same request.

## Fast mode

Fable 5.2 supports [fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode): the same model with up to 2.5x higher output tokens per second, at $16 / $80 per MTok (twice the standard price; cache multipliers apply on top). No beta header or access request is needed on the Claude API, though the organization has to be in the Start, Build, or Scale usage tier.

The pipe sends `speed: "fast"` on Fable 5.2 and Opus 5.5 requests when the `FAST_MODE` valve is on (off by default). Fast mode has its own rate limit. When it is exhausted the API returns a 429 (or 529), and after the SDK's own retries the pipe resends the request at standard speed rather than failing the reply. Requests at different speeds don't share a prompt cache, so the fallback request starts a fresh cache. Cost tracking reads `usage.speed` and prices fast requests at the fast mode rates.

## Thinking is always on

As on Fable 5.1, adaptive thinking can't be turned off. Both `{"type": "disabled"}` and the manual `{"type": "enabled", "budget_tokens": N}` form return a 400. Depth is steered by `output_config.effort`, which defaults to **`high`** and accepts the full `low`-`max` range. Anthropic recommends starting at `high`; most of the capability gain is already there. Prompt lines that ask the model to think less don't stop it from thinking at `medium` and above, so use a lower `EFFORT` for easy turns instead.

The pipe treats `claude-fable-5-2` like `claude-fable-5-1`: `ENABLE_THINKING = false` logs that thinking can't be disabled and leaves it on. `thinking.display` defaults to `"omitted"` at the API level, so the pipe sets `"summarized"` (via the `THINKING_DISPLAY` valve) to keep reasoning visible in Open WebUI's Thoughts section.

Because thinking always runs, `MAX_TOKENS` has to cover thinking plus the answer.

## Thinking blocks are bound to the conversation prefix, on every account

Fable 5.2 runs the same prefix check as Fable 5.1, Opus 5.5, Sonnet 5.5, and Haiku 5.5: editing anything before a thinking block (an earlier message, the system prompt, or the tool list) invalidates it. What changes is who is enforced. On Fable 5.1 a request that replays an invalid block fails only on accounts created on or after 31 August 2026; **on Fable 5.2 it fails on every account**, whatever its age.

Open WebUI lets users edit, branch, and regenerate earlier messages, and toggling web search or code execution changes the tool list. So the pipe adds `claude-fable-5-2` to `PREFIX_BINDING_MODELS` and sends the `thinking-binding-controls-2026-08-01` beta header with:

```python
params["thinking"]["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
```

The API then drops invalidated blocks (unbilled) and answers the request instead of rejecting it. A dropped block means the model thinks again where it was dropped, which can raise output tokens on edited chats. Because Fable 5.2's thinking is always adaptive, the opt-in is sent even when `ENABLE_THINKING` is false.

Model binding: Fable 5.2 reads thinking blocks from every earlier Opus, Sonnet, Haiku, Fable, and Mythos model, but **no earlier model reads Fable 5.2's blocks**, not even Fable 5.1 or Opus 5.5. Moving a chat from Fable 5.2 to another model continues without the earlier reasoning; the API drops the blocks and the request succeeds. Blocks are also tied to the account that produced them.

## Forced tool use is rejected

`tool_choice` of `{"type": "any"}` or `{"type": "tool", "name": "..."}` returns a 400, as does any `temperature`, `top_p`, or `top_k`. The pipe sets none of these, so nothing changes.

## Server tools

All supported, with the same versions as Fable 5.1:

- **Web search**: `web_search_20260318` with dynamic filtering
- **Web fetch**: `web_fetch_20260318` with dynamic filtering
- **Code execution**: `code_execution_20260521`

Since v0.21.2 the pipe keeps the dynamic-filtering versions when the user also turns on Open WebUI's code interpreter: the API accepts `code_execution_20260120` or later alongside them and runs the filtering in that tool's container.

## Refusals

Fable 5.2 runs the same safety classifiers as Fable 5.1, which decline in five categories: `cyber`, `bio`, `frontier_llm`, `reasoning_extraction`, and `general_harms`. Declines come back as `stop_reason: "refusal"` on an HTTP 200; the pipe's existing refusal handling shows a notice with the category instead of a blank reply. Server-side fallback (retrying a declined request on Opus 4.8 or Opus 5) is available in beta, but the pipe doesn't use it.

## Availability

Claude Fable 5.2 requires 30-day data retention and isn't available under zero data retention unless expressly authorized by Anthropic. Requests from an organization whose retention configuration doesn't meet the requirement return a 400. Fable 5.2 has its own rate limit, separate from the one Fable 5.1 and Fable 5 share.

## References

- [Claude Fable 5.2 overview](https://platform.claude.com/docs/en/models/fable-5-2/overview)
- [What's new in Claude Fable 5.2](https://platform.claude.com/docs/en/models/fable-5-2/whats-new-fable-5-2)
- [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)
- [Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode)
- [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
