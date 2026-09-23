"""
title: Anthropic Manifold Pipe
authors: warshanks
author_url: https://github.com/warshanks
funding_url: https://github.com/warshanks
version: 0.19.0
license: MIT

This pipe provides access to Anthropic's Claude models with support for:
- Open WebUI tools (workspace tools, MCP servers, tool servers and builtins)
  through native function calling
- Web search capabilities (dynamic result filtering on Claude 4.6+ models)
- Web fetch capabilities (dynamic content filtering on Claude 4.6+ models)
- Code execution in Anthropic's secure sandbox environment
- Extended thinking capabilities with proper validation
- Thinking shown in Open WebUI's Thoughts section, with signatures carried
  between turns out of band instead of in the reply text
- Automatic prompt caching for multi-turn conversations (large input cost savings)
- Image processing and analysis
- Centralized model capability management
- Proper handling of redacted thinking and streaming requirements
- Safety-classifier refusals surfaced instead of returning an empty response
- Preserved-thinking prefix binding handled on Claude Fable 5.1 and Claude Opus 5.5
"""

import json
import os
import re
import time
import requests
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
import anthropic
import asyncio
from loguru import logger
from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Optional,
    Callable,
    Awaitable,
    Literal,
    Any,
    AsyncIterator,
)

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)


class EventEmitter:
    """A helper class to abstract web-socket event emissions to the front-end."""

    def __init__(
        self,
        event_emitter: Callable[[dict], Awaitable[None]] | None,
    ):
        self.event_emitter = event_emitter

    async def emit_toast(
        self,
        msg: str,
        toastType: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        """Emits a toast notification to the front-end. This is a fire-and-forget operation."""
        if not self.event_emitter:
            return

        event = {
            "type": "notification",
            "data": {"type": toastType, "content": msg},
        }

        async def send_toast():
            try:
                # Re-check in case the event loop runs this later and state has changed.
                if self.event_emitter:
                    await self.event_emitter(event)
            except Exception:
                pass

        asyncio.create_task(send_toast())

    async def emit_usage(self, usage_data: dict[str, Any]) -> None:
        """A wrapper around emit_completion to specifically emit usage data."""
        await self.emit_completion(usage=usage_data)

    async def emit_completion(
        self,
        content: str | None = None,
        done: bool = False,
        error: str | None = None,
        sources: list[dict] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Constructs and emits completion event."""
        if not self.event_emitter:
            return

        emission = {
            "type": "chat:completion",
            "data": {"done": done},
        }
        if content is not None:
            emission["data"]["content"] = content
        if error is not None:
            emission["data"]["error"] = {"detail": error}
        if sources is not None:
            emission["data"]["sources"] = sources
        if usage is not None:
            emission["data"]["usage"] = usage

        try:
            await self.event_emitter(emission)
        except Exception:
            pass


class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")
        REQUIRE_USER_API_KEY: bool = Field(
            default=False,
            description="Whether to require user's own API key (applies to admins too).",
        )
        THINKING_BUDGET: int = Field(
            default=8192,
            description="Token budget for Claude's extended thinking capability (max tokens to use for thinking).",
        )
        MAX_TOKENS: int = Field(
            default=10240,
            description="Default maximum number of tokens to generate in the response.",
        )
        ENABLE_THINKING: bool = Field(
            default=True,
            description=(
                "Enable Claude's extended thinking capability for supported models. "
                "On Opus 5 thinking is on by default at the API level, so turning "
                "this off sends an explicit disabled config. Opus 5.5, Fable 5.1 "
                "and Fable 5 always think (the API rejects disabling it), so this "
                "valve has no effect there; lower EFFORT instead."
            ),
        )
        EFFORT: str = Field(
            default="",
            description=(
                "Effort level for adaptive-thinking models (Opus 5.5, Fable 5.1, "
                "Opus 5, Fable 5, Sonnet 5, Opus 4.8, Opus 4.6, Sonnet 4.6). "
                "One of: low, medium, high, xhigh, max. Empty = API default "
                "(medium on Opus 5.5, high elsewhere). "
                "'xhigh' is not supported on Opus 4.6 / Sonnet 4.6."
            ),
        )
        THINKING_DISPLAY: str = Field(
            default="summarized",
            description=(
                "How adaptive-thinking reasoning is returned: 'summarized' shows the "
                "model's reasoning in Open WebUI's Thoughts section; 'omitted' hides "
                "the reasoning text for lower streaming latency (the Thoughts section "
                "still appears, with a short note in place of the reasoning). "
                "Defaults to 'summarized' so it's clear the model thought. "
                "Note: Opus 5.5, Fable 5.1, Opus 5, Fable 5, Sonnet 5 and Opus 4.8 "
                "default to 'omitted' at the API level; this valve overrides that. You are billed for thinking "
                "tokens either way."
            ),
        )
        ENABLE_PROMPT_CACHING: bool = Field(
            default=True,
            description=(
                "Enable automatic prompt caching. The API caches the conversation "
                "prefix and re-reads it on follow-up turns at ~10% of the input "
                "price instead of reprocessing the full history every request. "
                "Has no effect on response content."
            ),
        )
        CACHE_TTL: str = Field(
            default="5m",
            description=(
                "Prompt cache lifetime: '5m' (default; refreshed at no cost each "
                "time it's used) or '1h' (2x write cost; use when users typically "
                "take more than 5 minutes between replies)."
            ),
        )

    class UserValves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")
        THINKING_BUDGET: int = Field(default=8192)
        MAX_TOKENS: int = Field(default=10240)
        ENABLE_THINKING: bool = Field(default=True)
        EFFORT: str = Field(default="")
        THINKING_DISPLAY: str = Field(default="")
        ENABLE_PROMPT_CACHING: bool = Field(default=True)
        CACHE_TTL: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"

        # Try to get valves from Functions first
        try:
            from open_webui.models.functions import Functions

            valves = Functions.get_function_valves_by_id("anthropic")
            self.valves = self.Valves(**(valves if valves else {}))
        except (ImportError, Exception):
            # Fallback to environment variables if Functions is not available
            self.valves = self.Valves(
                **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "")}
            )

        self.MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
        self.client = None
        self.is_thinking = False
        self.is_code_execution = False
        self.code_execution_block_index = None
        # Whether code execution blocks are rendered into the chat (only when
        # the user asked for code execution — see _code_execution_declared).
        self.show_code_execution = True
        self.event_emitter = None

        # Centralized model capability configuration
        self.MODEL_CAPABILITIES = {
            # Web Search: According to https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
            "web_search": {
                "claude-fable-5-1",
                "claude-opus-5-5",
                "claude-opus-5",
                "claude-fable-5",
                "claude-sonnet-5",
                "claude-opus-4-8",
                "claude-opus-4-7",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-opus-4-5-20251101",
                "claude-opus-4-1-20250805",
                "claude-opus-4-20250514",
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-20250514",
                "claude-haiku-4-5-20251001",
                "claude-3-7-sonnet-20250219",
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
            },
            # Web Fetch: According to https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
            "web_fetch": {
                "claude-fable-5-1",
                "claude-opus-5-5",
                "claude-opus-5",
                "claude-fable-5",
                "claude-sonnet-5",
                "claude-opus-4-8",
                "claude-opus-4-7",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-20250514",
                "claude-3-7-sonnet-20250219",
                "claude-haiku-4-5-20251001",
                "claude-3-5-haiku-latest",
                "claude-opus-4-5-20251101",
                "claude-opus-4-1-20250805",
                "claude-opus-4-20250514",
            },
            # Code Execution: According to https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
            "code_execution": {
                "claude-fable-5-1",
                "claude-opus-5-5",
                "claude-opus-5",
                "claude-fable-5",
                "claude-sonnet-5",
                "claude-opus-4-8",
                "claude-opus-4-7",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-opus-4-5-20251101",
                "claude-opus-4-1-20250805",
                "claude-opus-4-20250514",
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-20250514",
                "claude-haiku-4-5-20251001",
                "claude-3-7-sonnet-20250219",
                "claude-3-7-sonnet-latest",
                "claude-3-5-haiku-latest",
            },
            # Dynamic filtering for the web tools (web_search_20260318 /
            # web_fetch_20260318): Claude writes and runs code that filters
            # results before they reach the context window, cutting tokens on
            # search-heavy turns. Available on Claude 4.6 and later models.
            # https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
            "dynamic_web_tools": {
                "claude-fable-5-1",
                "claude-opus-5-5",
                "claude-opus-5",
                "claude-fable-5",
                "claude-sonnet-5",
                "claude-opus-4-8",
                "claude-opus-4-7",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
            },
            # Extended Thinking: According to https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
            "thinking": {
                "claude-fable-5-1",
                "claude-opus-5-5",
                "claude-opus-5",
                "claude-fable-5",
                "claude-sonnet-5",
                "claude-opus-4-8",
                "claude-opus-4-7",
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-opus-4-5-20251101",
                "claude-opus-4-1-20250805",
                "claude-opus-4-20250514",
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-20250514",
                "claude-haiku-4-5-20251001",
                "claude-3-7-sonnet-20250219",
                "claude-3-7-sonnet-latest",
            },
        }

        # Models where adaptive thinking is the only supported mode. Manual
        # thinking (`{"type": "enabled", "budget_tokens": N}`) is rejected with
        # a 400 on all of these, so they must never take the budget path.
        self.ADAPTIVE_THINKING_MODELS = {
            "claude-fable-5-1",
            "claude-opus-5-5",
            "claude-opus-5",
            "claude-fable-5",
            "claude-sonnet-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        }

        # Adaptive-thinking models that do not accept effort "xhigh"
        # (Opus 4.6 / Sonnet 4.6 support low/medium/high/max only).
        self.NO_XHIGH_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}

        # Models that think when the `thinking` parameter is omitted, so
        # honoring ENABLE_THINKING=False requires sending an explicit
        # `{"type": "disabled"}` config rather than leaving it out.
        self.THINKING_ON_BY_DEFAULT_MODELS = {
            "claude-fable-5-1",
            "claude-opus-5-5",
            "claude-opus-5",
            "claude-fable-5",
        }

        # Models that reject `{"type": "disabled"}` outright — thinking is
        # always on and the parameter has to be omitted (or sent as
        # `{"type": "adaptive"}`).
        self.THINKING_ALWAYS_ON_MODELS = {
            "claude-fable-5-1",
            "claude-opus-5-5",
            "claude-fable-5",
        }

        # Models that bind each thinking block to the conversation prefix that
        # produced it. Editing anything before a thinking block (an earlier
        # message, the system prompt, the tool list) invalidates every later
        # block, and replaying one is rejected with a 400 ("The block is bound
        # to a different conversation") on accounts created on or after
        # 2026-08-31. Open WebUI lets users edit and regenerate earlier turns,
        # so the pipe opts into dropping invalidated blocks instead.
        self.PREFIX_BINDING_MODELS = {"claude-fable-5-1", "claude-opus-5-5"}

        # Open WebUI builtin tools that duplicate an Anthropic server tool.
        # With native function calling, Open WebUI offers search_web and
        # fetch_url when web search is toggled on and execute_code for the code
        # interpreter; each is dropped while the pipe declares the Anthropic
        # version, so Claude isn't handed two tools for the same job.
        self.OWUI_TOOLS_REPLACED_BY = {
            "web_search": {"search_web"},
            "web_fetch": {"fetch_url"},
            "code_execution": {"execute_code"},
        }

        # Thoughts-section text for a thinking block with no readable text
        # (display "omitted", or no summary returned). Open WebUI only opens a
        # Thoughts item for reasoning text, and the block's signature has to
        # attach to one to be replayed on the next turn.
        self.THINKING_HIDDEN_NOTE = "*Reasoning hidden.*"
        self.REDACTED_THINKING_NOTE = "*[Some reasoning has been encrypted for safety]*"

        # Pricing per million tokens (Input / Output)
        # NOTE: _get_pricing matches by substring, so longer model IDs must
        # come first — "claude-fable-5" is a prefix of "claude-fable-5-1" and
        # "claude-opus-5" is a prefix of "claude-opus-5-5".
        self.PRICING = {
            # Claude 5 family
            "claude-fable-5-1": {"input": 10.00, "output": 50.00},
            "claude-opus-5-5": {"input": 4.00, "output": 20.00},
            "claude-opus-5": {"input": 5.00, "output": 25.00},
            "claude-fable-5": {"input": 10.00, "output": 50.00},
            "claude-sonnet-5": {"input": 2.00, "output": 10.00},
            # Claude 4.8 / 4.7 family
            "claude-opus-4-8": {"input": 5.00, "output": 25.00},
            "claude-opus-4-7": {"input": 5.00, "output": 25.00},
            # Claude 4.6 family
            "claude-opus-4-6": {"input": 5.00, "output": 25.00},
            "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
            # Claude 4.5 family
            "claude-opus-4-5": {"input": 5.00, "output": 25.00},
            "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
            "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
            # Claude 4 family
            "claude-opus-4": {"input": 15.00, "output": 75.00},
            "claude-sonnet-4": {"input": 3.00, "output": 15.00},
            # Claude 3.7
            "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},
            # Claude 3.5 family
            "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
            # Claude 3 family
            "claude-3-opus": {"input": 15.00, "output": 75.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
        }

        # Cache reads cost 0.10x the base input price on every model except
        # Claude Fable 5.1 (0.025x: $0.25 per MTok against a $10 input price)
        # and Claude Opus 5.5 (0.05x: $0.20 per MTok against a $4 input price).
        self.CACHE_READ_MULTIPLIER = 0.10
        self.REDUCED_CACHE_READ_MODELS = {
            "claude-fable-5-1": 0.025,
            "claude-opus-5-5": 0.05,
        }

        # Cost per web search request ($10 per 1,000 searches)
        self.WEB_SEARCH_COST = 0.01

        # Cache write premium over base input price (1.25x for 5m TTL,
        # 2x for 1h TTL); updated per-request from the CACHE_TTL valve.
        self.cache_write_multiplier = 1.25

    def get_anthropic_models(self):
        return [
            {"id": "claude-opus-5-5", "name": "claude-opus-5-5"},
            #{"id": "claude-fable-5-1", "name": "claude-fable-5-1"},
            #{"id": "claude-opus-5", "name": "claude-opus-5"},
            # {"id": "claude-fable-5", "name": "claude-fable-5"},
            #{"id": "claude-sonnet-5", "name": "claude-sonnet-5"},
            # {"id": "claude-opus-4-8", "name": "claude-opus-4-8"},
            # {"id": "claude-opus-4-6", "name": "claude-opus-4-6"},
            # {"id": "claude-sonnet-4-6", "name": "claude-sonnet-4-6"},
            #{"id": "claude-haiku-4-5-20251001", "name": "claude-haiku-4-5"},
        ]

    def pipes(self) -> List[dict]:
        return self.get_anthropic_models()

    def supports_capability(self, model_name: str, capability: str) -> bool:
        """Check if a model supports a specific capability."""
        return model_name in self.MODEL_CAPABILITIES.get(capability, set())

    def get_model_capabilities(self, model_name: str) -> List[str]:
        """Get all capabilities supported by a model."""
        capabilities = []
        for capability, models in self.MODEL_CAPABILITIES.items():
            if model_name in models:
                capabilities.append(capability)
        return capabilities

    def _get_pricing(self, model_name: str) -> dict[str, float]:
        """Get pricing for a specific model."""
        # Try exact match first
        for key, pricing in self.PRICING.items():
            if key in model_name:
                return pricing

        # Default or fallback (return 0s if unknown)
        print(f"Warning: No pricing found for model {model_name}")
        return {"input": 0.0, "output": 0.0}

    def _cache_read_multiplier(self, model_name: str) -> float:
        """Cache-hit price as a fraction of the base input price."""
        for key, multiplier in self.REDUCED_CACHE_READ_MODELS.items():
            if key in model_name:
                return multiplier
        return self.CACHE_READ_MULTIPLIER

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        web_search_count: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Calculate total cost for the usage."""
        pricing = self._get_pricing(model_name)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        cache_write_cost = (
            (cache_creation_tokens / 1_000_000)
            * pricing["input"]
            * self.cache_write_multiplier
        )
        cache_read_cost = (
            (cache_read_tokens / 1_000_000)
            * pricing["input"]
            * self._cache_read_multiplier(model_name)
        )
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        web_search_cost = web_search_count * self.WEB_SEARCH_COST
        return round(
            input_cost + cache_write_cost + cache_read_cost + output_cost
            + web_search_cost,
            6,
        )

    def _code_execution_declared(self, params) -> bool:
        """Whether this request declares the code execution tool itself.

        The dynamic-filtering web tools run their filtering code in a sandbox
        the API provisions on its own, so bash and file-operation blocks come
        back even when the user only asked for web search. Those are filtering
        internals rather than the user's code execution, so they are not
        rendered into the chat.
        """
        return any(
            isinstance(tool, dict) and tool.get("name") == "code_execution"
            for tool in (params.get("tools") or [])
        )

    def _refusal_notice(self, stop_details=None) -> str:
        """Build a user-facing notice for a `stop_reason: "refusal"` response.

        Opus 5.5 / Opus 5 (like Fable 5.1 / Fable 5) run safety classifiers that
        can decline a request. That comes back as a successful HTTP 200 with an empty or
        partial content list, so without this the user would just see a blank
        reply with no explanation.
        """
        category = getattr(stop_details, "category", None)
        explanation = getattr(stop_details, "explanation", None)
        detail = f" (category: {category})" if category else ""
        notice = (
            f"\n\n*[Anthropic's safety system declined this request{detail}. "
            "Try rephrasing it, or select a different model.]*\n"
        )
        if explanation:
            notice += f"\n*{explanation}*\n"
        return notice

    def _delta(self, **fields) -> dict:
        """An OpenAI-format stream chunk, for output that isn't reply text.

        Open WebUI reads `reasoning_content` into the Thoughts section,
        `reasoning_details` onto the Thoughts item, and `tool_calls` into its
        native tool-call loop. Plain strings yielded from the stream are still
        treated as reply text.
        """
        return {"choices": [{"index": 0, "delta": fields, "finish_reason": None}]}

    def _thinking_detail(self, thinking: str, signature: str) -> dict:
        """A thinking block as an Open WebUI `reasoning_details` entry.

        Open WebUI stores these on the reply's Thoughts item and hands them
        back on the assistant message in later requests to the same model, so
        thinking blocks keep their signatures between turns and through tool
        calls without being written into the reply text. The shape is
        OpenRouter's, which Open WebUI already understands.
        """
        return {
            "type": "reasoning.text",
            "text": thinking,
            "signature": signature,
            "format": "anthropic-claude-v1",
        }

    def _redacted_thinking_detail(self, data: str) -> dict:
        # No "format" key: Open WebUI drops anthropic-claude-v1 entries that
        # lack a signature, and a redacted block carries `data` instead.
        return {"type": "reasoning.encrypted", "data": data}

    def _thinking_blocks_from_details(self, details) -> list:
        """Rebuild thinking blocks from the `reasoning_details` Open WebUI replays."""
        blocks = []
        for detail in details if isinstance(details, list) else [details]:
            if not isinstance(detail, dict):
                continue
            if detail.get("type") == "reasoning.text" and detail.get("signature"):
                blocks.append(
                    {
                        "type": "thinking",
                        "thinking": detail.get("text") or "",
                        "signature": detail["signature"],
                    }
                )
            elif detail.get("type") == "reasoning.encrypted" and detail.get("data"):
                blocks.append({"type": "redacted_thinking", "data": detail["data"]})
        return blocks

    def _split_legacy_thinking(self, text: str):
        """Pull a signed thinking block out of reply text from older versions.

        Before 0.19.0 the pipe wrote thinking into the reply as
        <think>...</think> with the signature in a comment, in one of these
        formats (newest first):
          <think>...</think>\\n[//]: # (signature: ...)
          <think>...</think>\\n<!-- signature: ... -->
          <think>...\\n<!-- signature: ... --></think>
        Returns (thinking block or None, remaining text). Stray signature
        comments are stripped either way so they don't reach the model as text.
        """
        block = None
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            inner, after = match.group(1), text[match.end() :]
            signature_match = (
                re.search(r"<!-- signature: (.*?) -->", inner)
                or re.match(r"\s*<!-- signature: (.*?) -->", after)
                or re.match(r"\s*\[//\]: # \(signature: (.*?)\)", after)
            )
            if signature_match:
                block = {
                    "type": "thinking",
                    "thinking": re.sub(r"<!-- signature: .*? -->", "", inner).strip(),
                    "signature": signature_match.group(1),
                }
            text = text[: match.start()] + after
        text = re.sub(
            r"\[//\]: # \(signature: [^)]*\)|<!-- signature: .*? -->", "", text
        )
        return block, text.strip()

    def _content_blocks(self, content) -> list:
        """Text and image blocks for a user message or a tool result."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []

        blocks = []
        for item in content or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("text", "input_text"):
                if item.get("text"):  # Only add non-empty text blocks
                    blocks.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image_url":
                blocks.append(self.process_image(item))
            elif item.get("type") == "input_image" and item.get("image_url"):
                blocks.append(
                    self.process_image({"image_url": {"url": item["image_url"]}})
                )
        return blocks

    def _api_tool_name(self, name: str) -> str:
        """Tool name the API accepts (`^[a-zA-Z0-9_-]{1,128}$`).

        MCP tool names in particular can contain other characters. The mapping
        is deterministic, so a tool call replayed from history converts to the
        same name as the tool's current definition.
        """
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name or "")[:128]

    def _tool_use_block(self, tool_call: dict) -> dict:
        """An OpenAI-format tool call from Open WebUI as a `tool_use` block."""
        function = tool_call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        try:
            tool_input = (
                json.loads(arguments) if isinstance(arguments, str) else arguments
            )
        except ValueError:
            tool_input = {}
        return {
            "type": "tool_use",
            "id": tool_call.get("id", ""),
            "name": self._api_tool_name(function.get("name", "")),
            "input": tool_input if isinstance(tool_input, dict) else {},
        }

    def _convert_messages(self, messages: list) -> list:
        """Convert Open WebUI's OpenAI-format history to Anthropic messages.

        Assistant turns may carry `tool_calls` and `reasoning_details` (the
        thinking blocks this pipe streamed, which Open WebUI replays for the
        same model), and each tool result arrives as its own `role: "tool"`
        message. Consecutive messages with the same role are merged, which
        also keeps a turn's tool results together in one user message ahead
        of any follow-up content Open WebUI adds (e.g. images from tools).
        """
        converted = []
        total_image_size = 0

        for message in messages:
            role = message.get("role")
            content = message.get("content")
            input_blocks = []  # text and images from the user or a tool

            if role == "tool":
                input_blocks = self._content_blocks(content)
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id", ""),
                }
                if input_blocks:
                    tool_result["content"] = input_blocks
                role, blocks = "user", [tool_result]
            elif role == "assistant":
                blocks = self._thinking_blocks_from_details(
                    message.get("reasoning_details")
                )
                if isinstance(content, list):
                    content = "".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                legacy_thinking, text = self._split_legacy_thinking(content or "")
                if legacy_thinking and not blocks:
                    blocks.append(legacy_thinking)
                if text:
                    blocks.append({"type": "text", "text": text})
                blocks.extend(
                    self._tool_use_block(tool_call)
                    for tool_call in message.get("tool_calls") or []
                )
            else:
                blocks = input_blocks = self._content_blocks(content)

            # Track total size for base64 images
            for block in input_blocks:
                if block["type"] == "image" and block["source"]["type"] == "base64":
                    total_image_size += len(block["source"]["data"]) * 3 / 4
                    if total_image_size > 100 * 1024 * 1024:  # 100MB total limit
                        raise ValueError("Total size of images exceeds 100 MB limit")

            if not blocks:
                continue
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": blocks})

        return converted

    def _convert_owui_tools(self, body_tools, skip_names, reserved_names):
        """Convert the tools Open WebUI offers the model to Anthropic tools.

        With native function calling (Open WebUI's default), every tool
        enabled for the chat — workspace tools, MCP servers, tool servers and
        Open WebUI's builtins — arrives in `body["tools"]` in OpenAI format.
        Open WebUI runs the calls itself when the stream ends in tool calls,
        then calls the pipe again with the results. In legacy function calling
        Open WebUI resolves tools before the pipe runs and sends none here.

        Returns the Anthropic tool definitions and a map from each API tool
        name back to the Open WebUI name (they differ only when the original
        has characters the API rejects).
        """
        tools, owui_names = [], {}
        for tool in body_tools or []:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            spec = tool.get("function") or {}
            name = spec.get("name")
            if not name or name in skip_names:
                continue
            api_name = self._api_tool_name(name)
            if api_name in reserved_names or api_name in owui_names:
                print(f"Skipping tool '{name}': its name is already in use.")
                continue

            input_schema = dict(spec.get("parameters") or {})
            input_schema.setdefault("type", "object")
            definition = {"name": api_name, "input_schema": input_schema}
            if spec.get("description"):
                definition["description"] = spec["description"]

            tools.append(definition)
            owui_names[api_name] = name
        return tools, owui_names

    def process_image(self, image_data):
        """Process image data with size validation."""
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            # Check base64 image size
            image_size = len(base64_data) * 3 / 4  # Convert base64 size to bytes
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            # For URL images, perform size check after fetching
            url = image_data["image_url"]["url"]
            response = requests.head(url, allow_redirects=True)
            content_length = int(response.headers.get("content-length", 0))

            if content_length > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image at URL exceeds 5MB limit: {content_length / (1024 * 1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {"type": "url", "url": url},
            }

    def init_client(self, user_valves=None, event_emitter=None):
        """Initialize the Anthropic client with the appropriate API key.

        Beta headers are not set here. Which betas a request needs depends on
        the model and the tools enabled for that request, so they are sent per
        request via `extra_headers` (see `_beta_headers`) rather than baked
        into the client's default headers.
        """
        if event_emitter:
            self.event_emitter = EventEmitter(event_emitter)

        if not self.client:
            # Use user's API key if provided and allowed
            api_key = None
            if (
                user_valves
                and hasattr(user_valves, "ANTHROPIC_API_KEY")
                and user_valves.ANTHROPIC_API_KEY
            ):
                api_key = user_valves.ANTHROPIC_API_KEY
            else:
                api_key = self.valves.ANTHROPIC_API_KEY

            if not api_key and self.valves.REQUIRE_USER_API_KEY:
                raise ValueError("API key is required but not provided")

            self.client = anthropic.Anthropic(api_key=api_key)

    def _beta_headers(
        self,
        model_name,
        web_fetch_enabled=False,
        dynamic_web_tools=False,
    ) -> List[str]:
        """Beta opt-ins this request needs, as `anthropic-beta` values.

        Code execution no longer needs one: every current tool version
        (`code_execution_20250825` through `code_execution_20260521`) is GA.
        """
        betas = []

        if model_name in self.PREFIX_BINDING_MODELS:
            # Lets the API drop thinking blocks invalidated by an edited
            # history instead of rejecting the whole request.
            betas.append("thinking-binding-controls-2026-08-01")

        if (
            web_fetch_enabled
            and not dynamic_web_tools
            and self.supports_capability(model_name, "web_fetch")
        ):
            # Only the basic web_fetch_20250910 version needs this beta
            # header; the dynamic-filtering versions are GA.
            betas.append("web-fetch-2025-09-10")

        return betas

    async def pipe(
        self,
        body: dict,
        __user__=None,
        __metadata__=None,
        __event_emitter__=None,
        **kwargs,
    ) -> Union[str, Generator, Iterator, AsyncIterator]:
        # Get user valves if user info is provided
        user_valves = None
        if __user__ and hasattr(__user__, "valves") and __user__.valves:
            # Try to parse user valves
            try:
                user_valves = self.UserValves(**__user__.valves.get("anthropic", {}))
            except Exception as e:
                print(f"Error parsing user valves: {e}")

        # Prepare common parameters
        model_name = body["model"][body["model"].find(".") + 1 :]

        # Reset client to ensure correct headers for the model
        self.client = None

        # UI toggles. Open WebUI pops `features` off the request body before
        # calling pipes and passes them in __metadata__; the body is only a
        # fallback for older versions. They're read, never switched off here:
        # Open WebUI has already acted on them by the time the pipe runs, and
        # its tool-call loop calls the pipe again with the same metadata, so
        # clearing a flag would drop that server tool partway through the loop.
        metadata = __metadata__ or {}
        features = (
            metadata.get("features")
            or body.get("features")
            or body.get("metadata", {}).get("features")
        )
        if not isinstance(features, dict):
            features = {}

        # Check if code execution is enabled in the UI
        code_execution_enabled = bool(features.get("code_interpreter"))

        # Check if web search is enabled in the UI
        web_search_enabled = bool(features.get("web_search"))

        # Check if web fetch is enabled via the url_context toggle filter, which
        # shows up in filter_ids and sets "url_context" in features
        web_fetch_enabled = "gemini_url_context_toggle" in (
            metadata.get("filter_ids") or []
        ) or bool(features.get("url_context"))

        # Check if thinking is enabled in valves
        thinking_enabled = (
            user_valves.ENABLE_THINKING
            if user_valves and hasattr(user_valves, "ENABLE_THINKING")
            else self.valves.ENABLE_THINKING
        )

        # Pick the web tool versions for this request. The dynamic-filtering
        # versions run their filtering code inside a code execution sandbox that
        # the API provisions itself, so the code execution tool must not also be
        # declared in the same request — a second execution environment confuses
        # the model, and the pipe declares an older code execution version than
        # dynamic filtering calls into. When the user has the code interpreter
        # toggled on, their code execution wins and the basic web tools are used.
        code_execution_requested = code_execution_enabled and self.supports_capability(
            model_name, "code_execution"
        )
        use_dynamic_web_tools = (
            self.supports_capability(model_name, "dynamic_web_tools")
            and not code_execution_requested
        )

        self.init_client(user_valves, __event_emitter__)

        # Log model capabilities for debugging
        capabilities = self.get_model_capabilities(model_name)
        enabled_tools = []
        if self.supports_capability(model_name, "web_search") and web_search_enabled:
            enabled_tools.append("web_search")
        if (
            self.supports_capability(model_name, "code_execution")
            and code_execution_enabled
        ):
            enabled_tools.append("code_execution")
        if self.supports_capability(model_name, "thinking") and thinking_enabled:
            enabled_tools.append("thinking")
        if self.supports_capability(model_name, "web_fetch") and web_fetch_enabled:
            enabled_tools.append("web_fetch")

        if capabilities:
            print(f"Model {model_name} supports: {', '.join(capabilities)}")
        if enabled_tools:
            print(f"Enabled tools for {model_name}: {', '.join(enabled_tools)}")
        if use_dynamic_web_tools and (web_search_enabled or web_fetch_enabled):
            print(f"Using dynamic-filtering web tools for {model_name}")

        system_message, messages = pop_system_message(body["messages"])

        processed_messages = self._convert_messages(messages)

        # Add tools for supported models
        tools = []

        # Add web search tool if supported by model and enabled
        if self.supports_capability(model_name, "web_search") and web_search_enabled:
            tools.append(
                {
                    "type": (
                        "web_search_20260318"
                        if use_dynamic_web_tools
                        else "web_search_20250305"
                    ),
                    "name": "web_search",
                    "max_uses": 5,
                }
            )

        # Add code execution tool if supported by model and enabled
        if (
            self.supports_capability(model_name, "code_execution")
            and code_execution_enabled
        ):
            # code_execution_20260521: same runtime as 20260120, and the
            # version the current dynamic-filtering web tools call into.
            # Every supported model accepts it and none of the current
            # versions needs a beta header.
            tools.append({"type": "code_execution_20260521", "name": "code_execution"})

        # Add web fetch tool if supported by model and enabled
        if self.supports_capability(model_name, "web_fetch") and web_fetch_enabled:
            tools.append(
                {
                    "type": (
                        "web_fetch_20260318"
                        if use_dynamic_web_tools
                        else "web_fetch_20250910"
                    ),
                    "name": "web_fetch",
                    "max_uses": 5,
                    "citations": {"enabled": True},
                }
            )

        # Add the tools enabled in Open WebUI, minus any builtin that
        # duplicates a server tool declared above
        server_tool_names = {tool["name"] for tool in tools}
        replaced_tools = set().union(
            *(self.OWUI_TOOLS_REPLACED_BY.get(name, set()) for name in server_tool_names)
        )
        owui_tools, tool_names = self._convert_owui_tools(
            body.get("tools"), replaced_tools, server_tool_names
        )
        if owui_tools:
            tools.extend(owui_tools)
            print(f"Open WebUI tools for {model_name}: {', '.join(tool_names.values())}")

        # Convert to None if no tools
        tools = tools if tools else None

        # Create the parameters dict for the API call
        # Get max_tokens from user valves, system valves, or body, in that order of preference
        max_tokens = body.get("max_tokens", None)
        if max_tokens is None:
            if user_valves and hasattr(user_valves, "MAX_TOKENS"):
                max_tokens = user_valves.MAX_TOKENS
            else:
                max_tokens = self.valves.MAX_TOKENS

        params = {
            "model": model_name,
            "messages": processed_messages,
            "max_tokens": max_tokens,
        }

        # Add extended thinking capability for supported models
        if self.supports_capability(model_name, "thinking") and thinking_enabled:
            # Adaptive thinking models. On Opus 5 / Fable 5 / Sonnet 5 / Opus 4.8
            # this is the ONLY thinking mode (manual budget_tokens returns 400).
            # display="summarized" is required on those models, whose default is
            # "omitted" (empty thinking text); on Opus 4.6 / Sonnet 4.6 it is the
            # existing default and harmless.
            if model_name in self.ADAPTIVE_THINKING_MODELS:
                # Thinking display. Default "summarized" so the model's reasoning is
                # visible in the Thoughts section. Opus 5 / Fable 5 / Sonnet 5 / Opus 4.8
                # default to "omitted" (empty thinking text) at the API level, so we
                # set this explicitly.
                display = (
                    user_valves.THINKING_DISPLAY
                    if user_valves and getattr(user_valves, "THINKING_DISPLAY", "")
                    else self.valves.THINKING_DISPLAY
                )
                display = (display or "summarized").strip().lower()
                if display not in ("summarized", "omitted"):
                    print(f"Invalid THINKING_DISPLAY '{display}'; using 'summarized'.")
                    display = "summarized"
                params["thinking"] = {
                    "type": "adaptive",
                    "display": display,
                }
                print(
                    f"Enabling adaptive thinking for {model_name} (display={display})"
                )

                # Optional effort guidance (output_config.effort). Sent via
                # extra_body so it works regardless of installed SDK version.
                effort = (
                    user_valves.EFFORT
                    if user_valves and getattr(user_valves, "EFFORT", "")
                    else self.valves.EFFORT
                )
                effort = (effort or "").strip().lower()
                if effort:
                    valid_efforts = {"low", "medium", "high", "xhigh", "max"}
                    if effort not in valid_efforts:
                        print(
                            f"Invalid effort '{effort}'; valid values: {sorted(valid_efforts)}. Ignoring."
                        )
                    elif effort == "xhigh" and model_name in self.NO_XHIGH_MODELS:
                        print(
                            f"Effort 'xhigh' is not supported on {model_name} "
                            "(Opus 4.6 / Sonnet 4.6); ignoring."
                        )
                    else:
                        extra_body = params.setdefault("extra_body", {})
                        extra_body["output_config"] = {"effort": effort}
                        print(f"Setting effort={effort} for {model_name}")

                # Check if streaming is required for large max_tokens (>21,333 per documentation)
                if max_tokens > 21333 and not body.get("stream", False):
                    print(
                        f"Warning: max_tokens ({max_tokens}) > 21,333 requires streaming. Forcing streaming mode."
                    )
                    body["stream"] = True
            else:
                # Get thinking budget from user valves or default to system valve
                thinking_budget = (
                    user_valves.THINKING_BUDGET
                    if user_valves and hasattr(user_valves, "THINKING_BUDGET")
                    else self.valves.THINKING_BUDGET
                )

                # Validate thinking budget according to documentation requirements
                if thinking_budget > 0:
                    # Minimum budget is 1,024 tokens per documentation
                    if thinking_budget < 1024:
                        print(
                            f"Thinking budget {thinking_budget} is below minimum of 1,024 tokens. Setting to minimum."
                        )
                        thinking_budget = 1024

                    # Budget must be less than max_tokens per documentation
                    if thinking_budget >= max_tokens:
                        print(
                            f"Thinking budget {thinking_budget} must be less than max_tokens {max_tokens}. Disabling thinking."
                        )
                        thinking_budget = 0

                    if thinking_budget > 0:
                        # Set the thinking budget with the required type field
                        params["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": thinking_budget,
                        }
                        print(
                            f"Enabling extended thinking with budget: {thinking_budget} tokens, max_tokens: {max_tokens}"
                        )

                        # Check if streaming is required for large max_tokens (>21,333 per documentation)
                        if max_tokens > 21333 and not body.get("stream", False):
                            print(
                                f"Warning: max_tokens ({max_tokens}) > 21,333 requires streaming. Forcing streaming mode."
                            )
                            body["stream"] = True
        elif (
            model_name in self.THINKING_ON_BY_DEFAULT_MODELS
            and model_name not in self.THINKING_ALWAYS_ON_MODELS
        ):
            # Opus 5 thinks when `thinking` is omitted, so ENABLE_THINKING=False
            # has to be sent explicitly. The API only accepts a disabled config at
            # effort "high" or below, and effort is only sent alongside adaptive
            # thinking above, so the API default (high) applies here.
            params["thinking"] = {"type": "disabled"}
            print(f"Disabling thinking for {model_name} (thinking is on by default)")
        elif model_name in self.THINKING_ALWAYS_ON_MODELS:
            # Opus 5.5 / Fable 5.1 / Fable 5 reject `{"type": "disabled"}` at any
            # effort — thinking is always on and the parameter must be omitted.
            print(
                f"Thinking cannot be disabled on {model_name}; leaving it on (API default)."
            )

        # Preserved thinking: on models that bind thinking blocks to the
        # conversation prefix, replaying a block after the history changed is a
        # 400. Open WebUI rebuilds the message list every turn and lets users
        # edit, branch and regenerate earlier messages, so ask the API to drop
        # invalidated blocks and answer anyway. `{"type": "adaptive"}` is the
        # always-on default on these models, so adding it when thinking is off
        # in the valves changes nothing except giving block_binding a home.
        if model_name in self.PREFIX_BINDING_MODELS:
            thinking_params = params.setdefault("thinking", {"type": "adaptive"})
            thinking_params["block_binding"] = {
                "prefix_mismatch_behavior": "drop_block"
            }

        # Automatic prompt caching: a top-level cache_control makes the API
        # place a cache breakpoint on the last cacheable block and move it
        # forward each turn, so the growing conversation prefix is served
        # from cache (~10% of input price) instead of reprocessed in full on
        # every request. Prompts below the model's minimum cacheable length
        # are silently processed without caching, so this is always safe to
        # send. Passed via extra_body to work on any installed SDK version.
        caching_enabled = (
            user_valves.ENABLE_PROMPT_CACHING
            if user_valves and hasattr(user_valves, "ENABLE_PROMPT_CACHING")
            else self.valves.ENABLE_PROMPT_CACHING
        )
        if caching_enabled:
            cache_ttl = (
                user_valves.CACHE_TTL
                if user_valves and getattr(user_valves, "CACHE_TTL", "")
                else self.valves.CACHE_TTL
            )
            cache_ttl = (cache_ttl or "5m").strip().lower()
            if cache_ttl not in ("5m", "1h"):
                print(f"Invalid CACHE_TTL '{cache_ttl}'; using '5m'.")
                cache_ttl = "5m"
            self.cache_write_multiplier = 2.0 if cache_ttl == "1h" else 1.25
            cache_control = {"type": "ephemeral"}
            if cache_ttl == "1h":
                cache_control["ttl"] = "1h"
            params.setdefault("extra_body", {})["cache_control"] = cache_control

        # Add optional parameters
        if system_message:
            # pop_system_message returns the whole message dict; send only its
            # text rather than the dict's repr.
            system_content = system_message.get("content")
            if isinstance(system_content, list):
                system_content = "\n".join(
                    part.get("text", "")
                    for part in system_content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            if system_content:
                params["system"] = str(system_content)

        if body.get("stop"):
            params["stop_sequences"] = body.get("stop")

        if tools:
            params["tools"] = tools

        beta_headers = self._beta_headers(
            model_name,
            web_fetch_enabled=web_fetch_enabled,
            dynamic_web_tools=use_dynamic_web_tools,
        )
        if beta_headers:
            params["extra_headers"] = {"anthropic-beta": ",".join(beta_headers)}

        try:
            if body.get("stream", False):
                return self.stream_response_sdk(params, tool_names)
            else:
                # Awaited here: Open WebUI awaits pipe() itself but not a
                # coroutine it returns, which would leave the reply empty
                return await self.non_stream_response_sdk(params, tool_names)
        except anthropic.AuthenticationError as e:
            error_msg = f"Authentication error with Anthropic API: {e}. Please check your API key."
            print(error_msg)
            return error_msg
        except anthropic.RateLimitError as e:
            error_msg = f"Rate limit exceeded: {e}. Please try again later."
            print(error_msg)
            return error_msg
        except anthropic.APIStatusError as e:
            error_msg = f"API status error: {e}. Status: {e.status_code}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error in pipe method: {e}"
            print(error_msg)
            return error_msg

    async def stream_response_sdk(self, params, tool_names=None):
        """Stream a response to Open WebUI.

        `tool_names` maps API tool names back to Open WebUI tool names.
        """
        tool_names = tool_names or {}
        try:
            self.show_code_execution = self._code_execution_declared(params)
            # Whether thinking blocks will arrive without readable text:
            # display "omitted" is the API default wherever the pipe doesn't set
            # display itself, and manual (budget) thinking is always summarized.
            thinking_config = params.get("thinking") or {}
            hide_thinking_text = (
                thinking_config.get("type") != "enabled"
                and thinking_config.get("display", "omitted") == "omitted"
            )
            with self.client.messages.stream(**params) as stream:
                # For extended thinking: handle thinking blocks in the stream
                self.is_thinking = False
                self.thinking_signature = None
                thinking_text = ""
                thinking_shown = False
                reasoning_blocks = 0
                # Client tool calls by content block index. They are passed to
                # Open WebUI only once the stop reason is known, so a call cut
                # off by max_tokens or a refusal is never run.
                tool_calls = {}
                stop_reason = None
                # Track code execution state
                self.is_code_execution = False
                self.code_execution_block_index = None

                self.is_code_execution = False
                self.code_execution_block_index = None

                # Handle streaming events - use a hybrid approach for best compatibility
                processed_text_via_events = False

                # Usage tracking
                stream_start_time = time.time()
                input_tokens = 0
                output_tokens = 0
                cache_creation_tokens = 0
                cache_read_tokens = 0
                web_search_count = 0

                # Track when we're in tool usage to add proper spacing after tools
                in_tool_usage = False

                # Safety-classifier refusal tracking (stop_reason "refusal")
                refused = False
                refusal_stop_details = None

                current_text_block_index = None
                current_citations = []

                for event in stream:
                    # Handle different event types
                    if hasattr(event, "type"):
                        # Usage tracking
                        if event.type == "message_start" and hasattr(event, "message"):
                            if hasattr(event.message, "usage"):
                                usage = event.message.usage
                                input_tokens = usage.input_tokens
                                cache_creation_tokens = (
                                    getattr(usage, "cache_creation_input_tokens", 0)
                                    or 0
                                )
                                cache_read_tokens = (
                                    getattr(usage, "cache_read_input_tokens", 0) or 0
                                )

                        elif event.type == "message_delta":
                            if hasattr(event, "usage"):
                                output_tokens = event.usage.output_tokens
                            # A safety classifier can decline the request
                            # mid-stream (or before any output) — the stream
                            # ends normally, so surface it explicitly.
                            delta = getattr(event, "delta", None)
                            stop_reason = getattr(delta, "stop_reason", None) or stop_reason
                            if stop_reason == "refusal":
                                refused = True
                                refusal_stop_details = getattr(
                                    delta, "stop_details", None
                                )

                        # Handle thinking content block start. Reasoning goes
                        # to Open WebUI as reasoning_content, which it shows in
                        # the Thoughts section, never as reply text.
                        if (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "thinking"
                        ):
                            self.is_thinking = True
                            self.thinking_signature = None
                            thinking_text = ""
                            thinking_shown = hide_thinking_text
                            if reasoning_blocks:
                                # Open WebUI adds every thinking block in a
                                # response to one Thoughts section
                                yield self._delta(reasoning_content="\n\n")
                            reasoning_blocks += 1
                            if hide_thinking_text:
                                # Open the Thoughts section now so its timer
                                # covers the time spent thinking
                                yield self._delta(
                                    reasoning_content=self.THINKING_HIDDEN_NOTE
                                )

                        # Handle redacted thinking content block start
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "redacted_thinking"
                        ):
                            # Show user-friendly message for redacted thinking,
                            # and keep the encrypted block for the next request
                            separator = "\n\n" if reasoning_blocks else ""
                            reasoning_blocks += 1
                            yield self._delta(
                                reasoning_content=separator + self.REDACTED_THINKING_NOTE,
                                reasoning_details=[
                                    self._redacted_thinking_detail(
                                        event.content_block.data
                                    )
                                ],
                            )

                        # Handle client tool call start (Open WebUI tools)
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "tool_use"
                        ):
                            tool_calls[event.index] = {
                                "index": len(tool_calls),
                                "id": event.content_block.id,
                                "type": "function",
                                "function": {
                                    "name": tool_names.get(
                                        event.content_block.name,
                                        event.content_block.name,
                                    ),
                                    "arguments": "",
                                },
                            }

                        # Handle client tool call input deltas
                        elif (
                            event.type == "content_block_delta"
                            and getattr(event, "index", None) in tool_calls
                            and event.delta.type == "input_json_delta"
                        ):
                            tool_calls[event.index]["function"][
                                "arguments"
                            ] += event.delta.partial_json

                        # Handle thinking content deltas
                        elif (
                            self.is_thinking
                            and event.type == "content_block_delta"
                            and hasattr(event, "delta")
                            and hasattr(event.delta, "type")
                        ):
                            if event.delta.type == "thinking_delta" and hasattr(
                                event.delta, "thinking"
                            ):
                                # Yield thinking content as it arrives
                                thinking_text += event.delta.thinking
                                if event.delta.thinking:
                                    thinking_shown = True
                                    yield self._delta(
                                        reasoning_content=event.delta.thinking
                                    )
                            elif event.delta.type == "signature_delta" and hasattr(
                                event.delta, "signature"
                            ):
                                # Capture signature
                                if self.thinking_signature is None:
                                    self.thinking_signature = ""
                                self.thinking_signature += event.delta.signature

                        # Handle thinking content block end
                        elif self.is_thinking and event.type == "content_block_stop":
                            self.is_thinking = False
                            if not thinking_shown:
                                yield self._delta(
                                    reasoning_content=self.THINKING_HIDDEN_NOTE
                                )

                            # Attach the block and its signature to the
                            # Thoughts item. Open WebUI replays it on the
                            # assistant message in later requests, which keeps
                            # the signature out of the reply text entirely.
                            if self.thinking_signature:
                                yield self._delta(
                                    reasoning_details=[
                                        self._thinking_detail(
                                            thinking_text, self.thinking_signature
                                        )
                                    ]
                                )
                                self.thinking_signature = None

                        # Handle any tool use start (web search, code execution, etc.)
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "server_tool_use"
                        ):
                            tool_name = getattr(event.content_block, "name", None)
                            # Suppressed blocks render nothing, so they also
                            # don't need the trailing spacing newline emitted
                            # when a visible tool block ends.
                            is_code_exec_block = tool_name in (
                                "bash_code_execution",
                                "text_editor_code_execution",
                            )
                            in_tool_usage = (
                                self.show_code_execution or not is_code_exec_block
                            )
                            # Handle code execution specifically
                            if tool_name == "bash_code_execution":
                                self.is_code_execution = True
                                self.code_execution_block_index = getattr(
                                    event, "index", None
                                )
                                if self.show_code_execution:
                                    yield "\n**Bash Command:**\n```bash\n"

                            elif tool_name == "text_editor_code_execution":
                                self.is_code_execution = True
                                self.code_execution_block_index = getattr(
                                    event, "index", None
                                )
                                if self.show_code_execution:
                                    yield "\n**File Operation:**\n```\n"

                            elif tool_name == "web_fetch":
                                yield "\n**Using web fetch tool...**\n"

                            elif tool_name == "web_search":
                                web_search_count += 1
                                yield "\n**Searching the web...**\n"

                        # Handle code execution input deltas
                        elif (
                            self.is_code_execution
                            and event.type == "content_block_delta"
                            and hasattr(event, "delta")
                            and event.delta.type == "input_json_delta"
                            and hasattr(event.delta, "partial_json")
                        ):
                            # Extract and yield content from the partial JSON
                            pass

                        # Handle server tool use end (close code block)
                        elif (
                            self.is_code_execution
                            and event.type == "content_block_stop"
                            and (
                                self.code_execution_block_index is None
                                or getattr(event, "index", None)
                                == self.code_execution_block_index
                            )
                        ):
                            self.is_code_execution = False
                            self.code_execution_block_index = None
                            if self.show_code_execution:
                                yield "\n```\n"

                        # Handle code execution results
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and (
                                event.content_block.type
                                == "bash_code_execution_tool_result"
                                or event.content_block.type
                                == "text_editor_code_execution_tool_result"
                            )
                        ):
                            if self.show_code_execution:
                                yield "\n**Output:**\n```\n"
                                if hasattr(event.content_block, "content"):
                                    content = event.content_block.content

                                    # Handle Bash results
                                    if (
                                        event.content_block.type
                                        == "bash_code_execution_tool_result"
                                    ):
                                        stdout = getattr(content, "stdout", None)
                                        stderr = getattr(content, "stderr", None)
                                        if stdout:
                                            yield stdout
                                        if stderr:
                                            yield f"\n**Error:**\n{stderr}"

                                    # Handle Text Editor results
                                    elif (
                                        event.content_block.type
                                        == "text_editor_code_execution_tool_result"
                                    ):
                                        if (
                                            hasattr(content, "content")
                                            and content.content
                                        ):
                                            yield content.content
                                        if hasattr(content, "lines") and content.lines:
                                            # Diff format
                                            yield "\n".join(content.lines)
                                        if hasattr(content, "is_file_update"):
                                            yield (
                                                "File created."
                                                if not content.is_file_update
                                                else "File updated."
                                            )

                                yield "\n```\n\n"

                        # Handle web fetch results
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "web_fetch_tool_result"
                        ):
                            yield "Fetched.\n\n"

                        # Handle tool completion - add spacing when tool finishes
                        elif event.type == "content_block_stop" and in_tool_usage:
                            # Add a newline when tool block ends to ensure proper spacing
                            in_tool_usage = False
                            yield "\n"

                        # Handle content block start for text to track index
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "text"
                        ):
                            current_text_block_index = event.index
                            current_citations = []

                        # Handle regular text content deltas with citations support
                        elif event.type == "content_block_delta" and hasattr(
                            event, "delta"
                        ):
                            if event.delta.type == "text_delta" and hasattr(
                                event.delta, "text"
                            ):
                                # Only yield text if we're not in thinking or code execution mode
                                if not self.is_thinking and not self.is_code_execution:
                                    yield event.delta.text
                                    processed_text_via_events = True

                            elif event.delta.type == "citations_delta" and hasattr(
                                event.delta, "citation"
                            ):
                                # Handle citation delta
                                citation = event.delta.citation
                                current_citations.append(citation)

                        # Handle content block stop for text to emit citations
                        elif event.type == "content_block_stop":
                            # If we have collected citations for this block, emit them
                            if current_citations and self.event_emitter:
                                # Process citations into Open WebUI Source format
                                sources = []
                                for citation in current_citations:
                                    source = {"source": {}}

                                    # Handle different citation types
                                    if citation.type == "char_location":
                                        source["source"]["type"] = "document"
                                        # Map to document source if possible, or generic
                                        # The citation object has cited_text, document_index, etc.
                                        if hasattr(citation, "cited_text"):
                                            source["source"][
                                                "content"
                                            ] = citation.cited_text

                                        # Add metadata
                                        source["source"]["metadata"] = {
                                            "document_index": getattr(
                                                citation, "document_index", None
                                            ),
                                            "start_char_index": getattr(
                                                citation, "start_char_index", None
                                            ),
                                            "end_char_index": getattr(
                                                citation, "end_char_index", None
                                            ),
                                        }

                                    elif citation.type == "page_location":
                                        source["source"]["type"] = "document"
                                        if hasattr(citation, "cited_text"):
                                            source["source"][
                                                "content"
                                            ] = citation.cited_text

                                        source["source"]["metadata"] = {
                                            "document_index": getattr(
                                                citation, "document_index", None
                                            ),
                                            "start_page_number": getattr(
                                                citation, "start_page_number", None
                                            ),
                                            "end_page_number": getattr(
                                                citation, "end_page_number", None
                                            ),
                                        }

                                    elif citation.type == "content_block_location":
                                        source["source"]["type"] = "document"
                                        if hasattr(citation, "cited_text"):
                                            source["source"][
                                                "content"
                                            ] = citation.cited_text

                                        source["source"]["metadata"] = {
                                            "document_index": getattr(
                                                citation, "document_index", None
                                            ),
                                            "start_block_index": getattr(
                                                citation, "start_block_index", None
                                            ),
                                            "end_block_index": getattr(
                                                citation, "end_block_index", None
                                            ),
                                        }

                                    elif hasattr(
                                        citation, "url"
                                    ):  # Web search/fetch citation (inferred structure)
                                        source["source"]["type"] = "web_search_result"
                                        source["source"]["url"] = getattr(
                                            citation, "url", ""
                                        )
                                        source["source"]["title"] = getattr(
                                            citation, "title", "Web Source"
                                        )
                                        if hasattr(citation, "cited_text"):
                                            source["source"][
                                                "content"
                                            ] = citation.cited_text

                                        if hasattr(citation, "encrypted_index"):
                                            source["source"]["metadata"] = {
                                                "encrypted_index": citation.encrypted_index
                                            }

                                    # Fallback/General handling
                                    if "name" not in source["source"] and hasattr(
                                        citation, "title"
                                    ):
                                        source["source"]["name"] = citation.title
                                    elif "url" in source["source"]:
                                        source["source"]["name"] = source["source"][
                                            "url"
                                        ]
                                    else:
                                        source["source"][
                                            "name"
                                        ] = "Citation"  # Default name

                                    sources.append(source["source"])

                                if sources:
                                    await self.event_emitter.emit_completion(
                                        sources=sources
                                    )

                                # Reset for next block
                                current_citations = []

                # Fallback to text_stream if no text was processed via events
                # This ensures compatibility if the SDK behavior changes
                if not processed_text_via_events and hasattr(stream, "text_stream"):
                    for text in stream.text_stream:
                        if not self.is_thinking and not self.is_code_execution:
                            yield text

                if refused:
                    print(
                        "Request declined by Anthropic's safety classifiers "
                        f"(model: {params['model']})."
                    )
                    yield self._refusal_notice(refusal_stop_details)
                elif tool_calls and stop_reason == "tool_use":
                    # Hand the calls to Open WebUI, which runs them and calls
                    # the pipe again with the results
                    for tool_call in tool_calls.values():
                        if not tool_call["function"]["arguments"]:
                            tool_call["function"]["arguments"] = "{}"
                    yield self._delta(tool_calls=list(tool_calls.values()))

            # Calculate and emit usage. With prompt caching, input_tokens only
            # counts tokens after the cache breakpoint — the cached prefix is
            # reported separately in the cache_* fields, so sum all three for
            # the true prompt size.
            total_prompt_tokens = (
                input_tokens + cache_creation_tokens + cache_read_tokens
            )
            if self.event_emitter and total_prompt_tokens > 0:
                total_cost = self._calculate_cost(
                    input_tokens,
                    output_tokens,
                    params["model"],
                    web_search_count,
                    cache_creation_tokens,
                    cache_read_tokens,
                )
                completion_time = time.time() - stream_start_time

                usage_data = {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_prompt_tokens + output_tokens,
                    "total_cost": total_cost,
                    "completion_time": round(completion_time, 2),
                }
                if cache_creation_tokens or cache_read_tokens:
                    usage_data["cache_creation_input_tokens"] = cache_creation_tokens
                    usage_data["cache_read_input_tokens"] = cache_read_tokens

                await self.event_emitter.emit_usage(usage_data)

        except anthropic.AuthenticationError as e:
            error_msg = f"Authentication error with Anthropic API: {e}. Please check your API key."
            print(error_msg)
            yield error_msg
        except anthropic.RateLimitError as e:
            error_msg = f"Rate limit exceeded: {e}. Please try again later."
            print(error_msg)
            yield error_msg
        except anthropic.APIStatusError as e:
            error_msg = f"API status error: {e}. Status: {e.status_code}"
            print(error_msg)
            yield error_msg
        except Exception as e:
            error_msg = f"Error in stream_response_sdk: {e}"
            print(error_msg)
            yield error_msg

    async def non_stream_response_sdk(self, params, tool_names=None):
        """Return a complete response as an OpenAI-format chat completion.

        Thinking goes in `reasoning_content` / `reasoning_details` and client
        tool calls in `tool_calls`, as in the streamed version.
        """
        tool_names = tool_names or {}
        try:
            start_time = time.time()
            self.show_code_execution = self._code_execution_declared(params)
            response = self.client.messages.create(**params)

            # A safety classifier may decline the request: HTTP 200 with
            # stop_reason "refusal" and empty (or partial) content.
            refusal_notice = ""
            if getattr(response, "stop_reason", None) == "refusal":
                refusal_notice = self._refusal_notice(
                    getattr(response, "stop_details", None)
                )
                print(
                    "Request declined by Anthropic's safety classifiers "
                    f"(model: {params['model']})."
                )

            # Handle different content types in the response
            if hasattr(response, "content") and response.content:
                result_parts = []
                reasoning_parts = []
                reasoning_details = []
                tool_calls = []
                all_citations = []

                for content_block in response.content:
                    # Handle thinking content blocks
                    if content_block.type == "thinking":
                        if getattr(content_block, "thinking", None):
                            reasoning_parts.append(content_block.thinking)
                        if getattr(content_block, "signature", None):
                            reasoning_details.append(
                                self._thinking_detail(
                                    content_block.thinking or "",
                                    content_block.signature,
                                )
                            )

                    # Handle redacted thinking content blocks
                    elif content_block.type == "redacted_thinking":
                        reasoning_parts.append(self.REDACTED_THINKING_NOTE)
                        reasoning_details.append(
                            self._redacted_thinking_detail(content_block.data)
                        )

                    # Handle client tool calls (Open WebUI tools)
                    elif content_block.type == "tool_use":
                        tool_calls.append(
                            {
                                "index": len(tool_calls),
                                "id": content_block.id,
                                "type": "function",
                                "function": {
                                    "name": tool_names.get(
                                        content_block.name, content_block.name
                                    ),
                                    "arguments": json.dumps(content_block.input),
                                },
                            }
                        )

                    # Handle text content
                    elif content_block.type == "text":
                        result_parts.append(content_block.text)

                        # Collect and process citations if present
                        if (
                            hasattr(content_block, "citations")
                            and content_block.citations
                        ):
                            for citation in content_block.citations:
                                source = {"source": {}}

                                # Handle different citation types
                                if citation.type == "char_location":
                                    source["source"]["type"] = "document"
                                    if hasattr(citation, "cited_text"):
                                        source["source"][
                                            "content"
                                        ] = citation.cited_text

                                    source["source"]["metadata"] = {
                                        "document_index": getattr(
                                            citation, "document_index", None
                                        ),
                                        "start_char_index": getattr(
                                            citation, "start_char_index", None
                                        ),
                                        "end_char_index": getattr(
                                            citation, "end_char_index", None
                                        ),
                                    }

                                elif citation.type == "page_location":
                                    source["source"]["type"] = "document"
                                    if hasattr(citation, "cited_text"):
                                        source["source"][
                                            "content"
                                        ] = citation.cited_text

                                    source["source"]["metadata"] = {
                                        "document_index": getattr(
                                            citation, "document_index", None
                                        ),
                                        "start_page_number": getattr(
                                            citation, "start_page_number", None
                                        ),
                                        "end_page_number": getattr(
                                            citation, "end_page_number", None
                                        ),
                                    }

                                elif citation.type == "content_block_location":
                                    source["source"]["type"] = "document"
                                    if hasattr(citation, "cited_text"):
                                        source["source"][
                                            "content"
                                        ] = citation.cited_text

                                    source["source"]["metadata"] = {
                                        "document_index": getattr(
                                            citation, "document_index", None
                                        ),
                                        "start_block_index": getattr(
                                            citation, "start_block_index", None
                                        ),
                                        "end_block_index": getattr(
                                            citation, "end_block_index", None
                                        ),
                                    }

                                elif hasattr(
                                    citation, "url"
                                ):  # Web search/fetch citation (inferred structure)
                                    source["source"]["type"] = "web_search_result"
                                    source["source"]["url"] = getattr(
                                        citation, "url", ""
                                    )
                                    source["source"]["title"] = getattr(
                                        citation, "title", "Web Source"
                                    )
                                    if hasattr(citation, "cited_text"):
                                        source["source"][
                                            "content"
                                        ] = citation.cited_text

                                    if hasattr(citation, "encrypted_index"):
                                        source["source"]["metadata"] = {
                                            "encrypted_index": citation.encrypted_index
                                        }

                                # Fallback/General handling
                                if "name" not in source["source"] and hasattr(
                                    citation, "title"
                                ):
                                    source["source"]["name"] = citation.title
                                elif "url" in source["source"]:
                                    source["source"]["name"] = source["source"]["url"]
                                else:
                                    source["source"][
                                        "name"
                                    ] = "Citation"  # Default name

                                all_citations.append(source["source"])

                    # Handle code execution tool use
                    elif content_block.type == "server_tool_use":
                        if content_block.name == "bash_code_execution":
                            if (
                                self.show_code_execution
                                and hasattr(content_block, "input")
                                and "command" in content_block.input
                            ):
                                command = content_block.input["command"]
                                result_parts.append(
                                    f"\n**Bash Command:**\n```bash\n{command}\n```\n"
                                )

                        elif (
                            content_block.name == "text_editor_code_execution"
                            and self.show_code_execution
                        ):
                            if hasattr(content_block, "input"):
                                cmd = content_block.input.get("command", "")
                                path = content_block.input.get("path", "")
                                result_parts.append(
                                    f"\n**File Operation ({cmd}):** {path}\n"
                                )
                                if "file_text" in content_block.input:
                                    result_parts.append(
                                        f"```\n{content_block.input['file_text']}\n```\n"
                                    )

                        elif content_block.name == "web_fetch":
                            if (
                                hasattr(content_block, "input")
                                and "url" in content_block.input
                            ):
                                url = content_block.input["url"]
                                result_parts.append(f"\n**Web Fetch:** {url}\n")

                        elif content_block.name == "web_search":
                            result_parts.append(f"\n**Web Search:**\n")

                    # Handle code execution results
                    elif (
                        content_block.type == "bash_code_execution_tool_result"
                        and self.show_code_execution
                    ):
                        if hasattr(content_block, "content"):
                            content = content_block.content
                            result_parts.append("\n**Output:**\n```\n")
                            if hasattr(content, "stdout") and content.stdout:
                                result_parts.append(content.stdout)
                            if hasattr(content, "stderr") and content.stderr:
                                result_parts.append(f"\n**Error:**\n{content.stderr}")
                            result_parts.append("\n```\n")

                    elif (
                        content_block.type == "text_editor_code_execution_tool_result"
                        and self.show_code_execution
                    ):
                        if hasattr(content_block, "content"):
                            content = content_block.content
                            result_parts.append("\n**Output:**\n```\n")
                            if hasattr(content, "content") and content.content:
                                result_parts.append(content.content)
                            if hasattr(content, "lines") and content.lines:
                                result_parts.append("\n".join(content.lines))
                            if hasattr(content, "is_file_update"):
                                result_parts.append(
                                    "File created."
                                    if not content.is_file_update
                                    else "File updated."
                                )
                            result_parts.append("\n```\n")

                    elif content_block.type == "web_fetch_tool_result":
                        result_parts.append("Fetched.\n\n")

                if all_citations and self.event_emitter:
                    await self.event_emitter.emit_completion(sources=all_citations)

                # Emit usage
                if self.event_emitter and hasattr(response, "usage"):
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    cache_creation_tokens = (
                        getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                    )
                    cache_read_tokens = (
                        getattr(response.usage, "cache_read_input_tokens", 0) or 0
                    )

                    # Try to get web search count from usage if available, otherwise count properties could be used
                    # but simpler to rely on parsing content blocks if we tracked them.
                    # As a fallback, let's recount from content blocks for consistency or check usage
                    web_search_count = 0
                    if hasattr(response, "usage") and isinstance(response.usage, dict):
                        # check dict
                        pass

                    # Manual count from content blocks is reliable for intent
                    for block in response.content:
                        if (
                            block.type == "server_tool_use"
                            and block.name == "web_search"
                        ):
                            web_search_count += 1

                    total_cost = self._calculate_cost(
                        input_tokens,
                        output_tokens,
                        params["model"],
                        web_search_count,
                        cache_creation_tokens,
                        cache_read_tokens,
                    )
                    completion_time = time.time() - start_time

                    # input_tokens only counts tokens after the cache
                    # breakpoint; add the cached prefix for the true total.
                    total_prompt_tokens = (
                        input_tokens + cache_creation_tokens + cache_read_tokens
                    )
                    usage_data = {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": total_prompt_tokens + output_tokens,
                        "total_cost": total_cost,
                        "completion_time": round(completion_time, 2),
                    }
                    if cache_creation_tokens or cache_read_tokens:
                        usage_data["cache_creation_input_tokens"] = (
                            cache_creation_tokens
                        )
                        usage_data["cache_read_input_tokens"] = cache_read_tokens

                    await self.event_emitter.emit_usage(usage_data)

                if refusal_notice:
                    result_parts.append(refusal_notice)

                message = {"role": "assistant", "content": "".join(result_parts)}
                if reasoning_details and not reasoning_parts:
                    reasoning_parts.append(self.THINKING_HIDDEN_NOTE)
                if reasoning_parts:
                    message["reasoning_content"] = "\n\n".join(reasoning_parts)
                if reasoning_details:
                    message["reasoning_details"] = reasoning_details
                # Only a turn that ended to call tools gets them run; one cut
                # off by max_tokens or a refusal may hold incomplete calls
                if tool_calls and response.stop_reason == "tool_use":
                    message["tool_calls"] = tool_calls

                return {
                    "id": response.id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": params["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": (
                                "tool_calls" if "tool_calls" in message else "stop"
                            ),
                        }
                    ],
                }

            return refusal_notice
        except anthropic.AuthenticationError as e:
            error_msg = f"Authentication error with Anthropic API: {e}. Please check your API key."
            print(error_msg)
            return error_msg
        except anthropic.RateLimitError as e:
            error_msg = f"Rate limit exceeded: {e}. Please try again later."
            print(error_msg)
            return error_msg
        except anthropic.APIStatusError as e:
            error_msg = f"API status error: {e}. Status: {e.status_code}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error in non_stream_response_sdk: {e}"
            print(error_msg)
            return error_msg
