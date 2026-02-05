"""
title: Anthropic Manifold Pipe
authors: warshanks
author_url: https://github.com/warshanks
funding_url: https://github.com/warshanks
version: 0.11.0
license: MIT

This pipe provides access to Anthropic's Claude models with support for:
- Web search capabilities
- Web fetch capabilities
- Code execution in Anthropic's secure sandbox environment
- Extended thinking capabilities with proper validation
- Image processing and analysis
- Centralized model capability management
- Proper handling of redacted thinking and streaming requirements
"""

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
from typing import List, Union, Generator, Iterator, Optional, Callable, Awaitable, Literal, Any, AsyncIterator


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
            description="Enable Claude's extended thinking capability for supported models.",
        )

    class UserValves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")
        THINKING_BUDGET: int = Field(default=8192)
        MAX_TOKENS: int = Field(default=10240)
        ENABLE_THINKING: bool = Field(default=True)

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
        self.thinking_start_time = None
        self.is_code_execution = False
        self.code_execution_block_index = None
        self.event_emitter = None

        # Centralized model capability configuration
        self.MODEL_CAPABILITIES = {
            # Web Search: According to https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
            "web_search": {
                "claude-opus-4-6",
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
                "claude-opus-4-6",
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
                "claude-opus-4-6",
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
            # Extended Thinking: According to https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
            "thinking": {
                "claude-opus-4-6",
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

        # Pricing per million tokens (Input / Output)
        self.PRICING = {
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

        # Cost per web search request
        self.WEB_SEARCH_COST = 0.01

    def get_anthropic_models(self):
        return [
            {"id": "claude-opus-4-6", "name": "claude-opus-4-6"},
            {"id": "claude-sonnet-4-5-20250929", "name": "claude-sonnet-4-5"},
            {"id": "claude-haiku-4-5-20251001", "name": "claude-haiku-4-5"},
            {"id": "claude-opus-4-5-20251101", "name": "claude-opus-4-5"},
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

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str, web_search_count: int = 0) -> float:
        """Calculate total cost for the usage."""
        pricing = self._get_pricing(model_name)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        web_search_cost = web_search_count * self.WEB_SEARCH_COST
        return round(input_cost + output_cost + web_search_cost, 6)


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

    def init_client(
        self,
        user_valves=None,
        model_name=None,
        code_execution_enabled=True,
        web_fetch_enabled=False,
        event_emitter=None,
    ):
        """Initialize Anthropic client with appropriate API key and beta headers."""
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

            # Determine required beta headers based on model capabilities
            beta_headers = []

            if model_name:
                if code_execution_enabled:
                    # Check if the model supports code execution
                    if self.supports_capability(model_name, "code_execution"):
                        beta_headers.append("code-execution-2025-08-25")

                if web_fetch_enabled:
                    # Check if the model supports web fetch
                    if self.supports_capability(model_name, "web_fetch"):
                        beta_headers.append("web-fetch-2025-09-10")

            # Create default headers
            default_headers = {}
            if beta_headers:
                default_headers["anthropic-beta"] = ",".join(beta_headers)

            self.client = anthropic.Anthropic(
                api_key=api_key,
                default_headers=default_headers if default_headers else None,
            )

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

        features = body.get("features")
        if not features:
            features = body.get("metadata", {}).get("features")

        # Check if code execution is enabled in the UI
        code_execution_enabled = False
        if features and isinstance(features, dict):
            code_execution_enabled = features.get("code_interpreter", False)
            # Disable OWUI code execution
            features["code_interpreter"] = False

        # Check if web search is enabled in the UI
        web_search_enabled = False
        if features and isinstance(features, dict):
            web_search_enabled = features.get("web_search", False)
            # Disable OWUI web search
            features["web_search"] = False

        # Check if web fetch is enabled via url_context toggle
        web_fetch_enabled = False

        # Check using __metadata__ if available (preferred method for toggle filters)
        if __metadata__:
            user_toggled_ids = __metadata__.get("filter_ids", [])
            if "gemini_url_context_toggle" in user_toggled_ids:
                web_fetch_enabled = True

        # Fallback to checking features dict if not enabled via metadata
        if not web_fetch_enabled:
            if features and isinstance(features, dict):
                # The filter sets "url_context" to True
                web_fetch_enabled = features.get("url_context", False)

        # Check if thinking is enabled in valves
        thinking_enabled = (
            user_valves.ENABLE_THINKING
            if user_valves and hasattr(user_valves, "ENABLE_THINKING")
            else self.valves.ENABLE_THINKING
        )

        self.init_client(
            user_valves,
            model_name,
            code_execution_enabled,
            web_fetch_enabled,
            __event_emitter__,
        )


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

        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []
        total_image_size = 0

        for message in messages:
            processed_content = []
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item["type"] == "text":
                        if item["text"]:  # Only add non-empty text blocks
                            processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        processed_image = self.process_image(item)
                        processed_content.append(processed_image)

                        # Track total size for base64 images
                        if processed_image["source"]["type"] == "base64":
                            image_size = len(processed_image["source"]["data"]) * 3 / 4
                            total_image_size += image_size
                            if (
                                total_image_size > 100 * 1024 * 1024
                            ):  # 100MB total limit
                                raise ValueError(
                                    "Total size of images exceeds 100 MB limit"
                                )
            else:
                content_text = message.get("content", "")

                # Check for thinking blocks with signatures in the text
                # Format: <think>\n...\n<!-- signature: ... -->\n</think>
                # OR: <think>...</think>\n[//]: # (signature: ...) (New Markdown)
                # OR: <think>...</think>\n<!-- signature: ... --> (Old HTML)
                # OR: <think>...\n<!-- signature: ... --></think> (Oldest)
                thinking_match = re.search(
                    r"<think>(.*?)</think>",
                    content_text,
                    re.DOTALL
                )

                if thinking_match and message["role"] == "assistant":
                    thinking_content = thinking_match.group(1)
                    signature = None

                    # Check for signature inside (oldest format)
                    inside_match = re.search(r"<!-- signature: (.*?) -->", thinking_content)
                    if inside_match:
                        signature = inside_match.group(1)
                        thinking_content = thinking_content.replace(inside_match.group(0), "")
                    else:
                        # Check for signature after (HTML format)
                        after_html_match = re.search(
                            r"</think>\s*<!-- signature: (.*?) -->",
                            content_text,
                            re.DOTALL
                        )
                        if after_html_match:
                            signature = after_html_match.group(1)
                        else:
                            # Check for signature after (Markdown format)
                            after_md_match = re.search(
                                r"</think>\s*\[//\]: # \(signature: (.*?)\)",
                                content_text,
                                re.DOTALL
                            )
                            if after_md_match:
                                signature = after_md_match.group(1)

                    thinking_content = thinking_content.strip()

                    # Remove the thinking block and signature from the text content
                    # Remove Markdown signature
                    text_content = re.sub(
                        r"<think>.*?</think>(?:\s*\[//\]: # \(signature: .*?\))?\s*",
                        "",
                        content_text,
                        flags=re.DOTALL
                    )
                    # Remove HTML signature (if present instead)
                    text_content = re.sub(
                        r"<think>.*?</think>(?:\s*<!-- signature: .*? -->)?\s*",
                        "",
                        text_content,
                        flags=re.DOTALL
                    ).strip()

                    # Create the thinking block
                    thinking_block = {
                        "type": "thinking",
                        "thinking": thinking_content
                    }

                    if signature:
                        thinking_block["signature"] = signature

                    processed_content.append(thinking_block)

                    # Add remaining text if any
                    if text_content:
                        processed_content.append({
                            "type": "text",
                            "text": text_content
                        })
                else:
                    processed_content = [
                        {"type": "text", "text": content_text}
                    ]

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )

            # Add tools for supported models
        tools = []

        # Add web search tool if supported by model and enabled
        if self.supports_capability(model_name, "web_search") and web_search_enabled:
            tools.append(
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
            )

        # Add code execution tool if supported by model and enabled
        if (
            self.supports_capability(model_name, "code_execution")
            and code_execution_enabled
        ):
            tools.append({"type": "code_execution_20250825", "name": "code_execution"})

        # Add web fetch tool if supported by model and enabled
        if self.supports_capability(model_name, "web_fetch") and web_fetch_enabled:
            tools.append(
                {
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "max_uses": 5,
                    "citations": {"enabled": True},
                }
            )

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
            # Check for Opus 4.6 specifically for adaptive thinking
            if model_name == "claude-opus-4-6":
                params["thinking"] = {
                    "type": "adaptive"
                }
                print(f"Enabling adaptive thinking for {model_name}")

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

        # Add optional parameters
        if system_message:
            params["system"] = str(system_message)

        if body.get("stop"):
            params["stop_sequences"] = body.get("stop")

        if tools:
            params["tools"] = tools

        try:
            if body.get("stream", False):
                return self.stream_response_sdk(params)
            else:
                return self.non_stream_response_sdk(params)
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

    async def stream_response_sdk(self, params):
        try:
            with self.client.messages.stream(**params) as stream:
                # For extended thinking: handle thinking blocks in the stream
                self.is_thinking = False
                self.thinking_start_time = None
                self.thinking_signature = None
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
                web_search_count = 0

                # Track when we're in tool usage to add proper spacing after tools
                in_tool_usage = False

                current_text_block_index = None
                current_citations = []

                for event in stream:
                    # Handle different event types
                    if hasattr(event, "type"):
                        # Usage tracking
                        if event.type == "message_start" and hasattr(event, "message"):
                            if hasattr(event.message, "usage"):
                                input_tokens = event.message.usage.input_tokens

                        elif event.type == "message_delta" and hasattr(event, "usage"):
                            output_tokens = event.usage.output_tokens

                        # Handle thinking content block start
                        if (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "thinking"
                        ):
                            self.is_thinking = True
                            self.thinking_start_time = time.time()
                            self.thinking_signature = None
                            # Yield opening tag for thinking block
                            yield "<think>\n"

                        # Handle redacted thinking content block start
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "redacted_thinking"
                        ):
                            # Show user-friendly message for redacted thinking
                            yield "\n*[Some reasoning has been encrypted for safety]*\n\n"

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
                                yield event.delta.thinking
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
                            thinking_time = None
                            if self.thinking_start_time:
                                thinking_time = time.time() - self.thinking_start_time
                                self.thinking_start_time = None

                            # Inject signature if present
                            # Close the thinking block
                            yield "\n</think>"

                            # Inject signature if present
                            if self.thinking_signature:
                                yield f"\n[//]: # (signature: {self.thinking_signature})"
                                self.thinking_signature = None

                            yield "\n\n"

                        # Handle any tool use start (web search, code execution, etc.)
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and event.content_block.type == "server_tool_use"
                        ):
                            in_tool_usage = True
                        # Handle code execution specifically
                            if (
                                hasattr(event.content_block, "name")
                                and event.content_block.name == "bash_code_execution"
                            ):
                                self.is_code_execution = True
                                self.code_execution_block_index = getattr(
                                    event, "index", None
                                )
                                yield "\n**Bash Command:**\n```bash\n"

                            elif (
                                hasattr(event.content_block, "name")
                                and event.content_block.name == "text_editor_code_execution"
                            ):
                                self.is_code_execution = True
                                self.code_execution_block_index = getattr(
                                    event, "index", None
                                )
                                yield "\n**File Operation:**\n```\n"

                            elif (
                                hasattr(event.content_block, "name")
                                and event.content_block.name == "web_fetch"
                            ):
                                yield "\n**Using web fetch tool...**\n"

                            elif (
                                hasattr(event.content_block, "name")
                                and event.content_block.name == "web_search"
                            ):
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
                            yield "\n```\n"

                        # Handle code execution results
                        elif (
                            event.type == "content_block_start"
                            and hasattr(event, "content_block")
                            and (
                                event.content_block.type == "bash_code_execution_tool_result"
                                or event.content_block.type == "text_editor_code_execution_tool_result"
                            )
                        ):
                            yield "\n**Output:**\n```\n"
                            if hasattr(event.content_block, "content"):
                                content = event.content_block.content

                                # Handle Bash results
                                if event.content_block.type == "bash_code_execution_tool_result":
                                    if hasattr(content, "stdout") and content.stdout:
                                        yield content.stdout
                                    if hasattr(content, "stderr") and content.stderr:
                                        yield f"\n**Error:**\n{content.stderr}"

                                # Handle Text Editor results
                                elif event.content_block.type == "text_editor_code_execution_tool_result":
                                    if hasattr(content, "content") and content.content:
                                        yield content.content
                                    if hasattr(content, "lines") and content.lines:
                                        # Diff format
                                        yield "\n".join(content.lines)
                                    if hasattr(content, "is_file_update"):
                                        yield "File created." if not content.is_file_update else "File updated."

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
                        elif (
                            event.type == "content_block_delta"
                            and hasattr(event, "delta")
                        ):
                            if event.delta.type == "text_delta" and hasattr(event.delta, "text"):
                                # Only yield text if we're not in thinking or code execution mode
                                if not self.is_thinking and not self.is_code_execution:
                                    yield event.delta.text
                                    processed_text_via_events = True

                            elif event.delta.type == "citations_delta" and hasattr(event.delta, "citation"):
                                # Handle citation delta
                                citation = event.delta.citation
                                current_citations.append(citation)

                        # Handle content block stop for text to emit citations
                        elif (
                            event.type == "content_block_stop"
                        ):
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
                                            source["source"]["content"] = citation.cited_text

                                        # Add metadata
                                        source["source"]["metadata"] = {
                                            "document_index": getattr(citation, "document_index", None),
                                            "start_char_index": getattr(citation, "start_char_index", None),
                                            "end_char_index": getattr(citation, "end_char_index", None)
                                        }

                                    elif citation.type == "page_location":
                                        source["source"]["type"] = "document"
                                        if hasattr(citation, "cited_text"):
                                            source["source"]["content"] = citation.cited_text

                                        source["source"]["metadata"] = {
                                            "document_index": getattr(citation, "document_index", None),
                                            "start_page_number": getattr(citation, "start_page_number", None),
                                            "end_page_number": getattr(citation, "end_page_number", None)
                                        }

                                    elif citation.type == "content_block_location":
                                        source["source"]["type"] = "document"
                                        if hasattr(citation, "cited_text"):
                                            source["source"]["content"] = citation.cited_text

                                        source["source"]["metadata"] = {
                                            "document_index": getattr(citation, "document_index", None),
                                            "start_block_index": getattr(citation, "start_block_index", None),
                                            "end_block_index": getattr(citation, "end_block_index", None)
                                        }

                                    elif hasattr(citation, "url"): # Web search/fetch citation (inferred structure)
                                         source["source"]["type"] = "web_search_result"
                                         source["source"]["url"] = getattr(citation, "url", "")
                                         source["source"]["title"] = getattr(citation, "title", "Web Source")
                                         if hasattr(citation, "cited_text"):
                                             source["source"]["content"] = citation.cited_text

                                         if hasattr(citation, "encrypted_index"):
                                             source["source"]["metadata"] = {
                                                 "encrypted_index": citation.encrypted_index
                                             }

                                    # Fallback/General handling
                                    if "name" not in source["source"] and hasattr(citation, "title"):
                                        source["source"]["name"] = citation.title
                                    elif "url" in source["source"]:
                                        source["source"]["name"] = source["source"]["url"]
                                    else:
                                        source["source"]["name"] = "Citation" # Default name

                                    sources.append(source["source"])

                                if sources:
                                    await self.event_emitter.emit_completion(sources=sources)

                                # Reset for next block
                                current_citations = []


                # Fallback to text_stream if no text was processed via events
                # This ensures compatibility if the SDK behavior changes
                if not processed_text_via_events and hasattr(stream, "text_stream"):
                    for text in stream.text_stream:
                        if not self.is_thinking and not self.is_code_execution:
                            yield text

            # Calculate and emit usage
            if self.event_emitter and input_tokens > 0:
                total_cost = self._calculate_cost(input_tokens, output_tokens, params["model"], web_search_count)
                completion_time = time.time() - stream_start_time

                usage_data = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "total_cost": total_cost,
                    "completion_time": round(completion_time, 2)
                }

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

    async def non_stream_response_sdk(self, params):
        try:
            start_time = time.time()
            response = self.client.messages.create(**params)

            # Handle different content types in the response
            if hasattr(response, "content") and response.content:
                result_parts = []
                all_citations = []

                for content_block in response.content:
                    # Handle thinking content blocks
                    if content_block.type == "thinking":
                        if (
                            hasattr(content_block, "thinking")
                            and content_block.thinking
                        ):
                            signature_part = ""
                            if hasattr(content_block, "signature") and content_block.signature:
                                signature_part = f"\n[//]: # (signature: {content_block.signature})"

                            result_parts.append(
                                f"<think>\n{content_block.thinking}\n</think>{signature_part}\n\n"
                            )

                    # Handle redacted thinking content blocks
                    elif content_block.type == "redacted_thinking":
                        result_parts.append(
                            "\n*[Some reasoning has been encrypted for safety]*\n\n"
                        )

                    # Handle text content
                    elif content_block.type == "text":
                        result_parts.append(content_block.text)

                        # Collect and process citations if present
                        if hasattr(content_block, "citations") and content_block.citations:
                            for citation in content_block.citations:
                                source = {"source": {}}

                                # Handle different citation types
                                if citation.type == "char_location":
                                    source["source"]["type"] = "document"
                                    if hasattr(citation, "cited_text"):
                                        source["source"]["content"] = citation.cited_text

                                    source["source"]["metadata"] = {
                                        "document_index": getattr(citation, "document_index", None),
                                        "start_char_index": getattr(citation, "start_char_index", None),
                                        "end_char_index": getattr(citation, "end_char_index", None)
                                    }

                                elif citation.type == "page_location":
                                    source["source"]["type"] = "document"
                                    if hasattr(citation, "cited_text"):
                                        source["source"]["content"] = citation.cited_text

                                    source["source"]["metadata"] = {
                                        "document_index": getattr(citation, "document_index", None),
                                        "start_page_number": getattr(citation, "start_page_number", None),
                                        "end_page_number": getattr(citation, "end_page_number", None)
                                    }

                                elif citation.type == "content_block_location":
                                    source["source"]["type"] = "document"
                                    if hasattr(citation, "cited_text"):
                                        source["source"]["content"] = citation.cited_text

                                    source["source"]["metadata"] = {
                                        "document_index": getattr(citation, "document_index", None),
                                        "start_block_index": getattr(citation, "start_block_index", None),
                                        "end_block_index": getattr(citation, "end_block_index", None)
                                    }

                                elif hasattr(citation, "url"): # Web search/fetch citation (inferred structure)
                                     source["source"]["type"] = "web_search_result"
                                     source["source"]["url"] = getattr(citation, "url", "")
                                     source["source"]["title"] = getattr(citation, "title", "Web Source")
                                     if hasattr(citation, "cited_text"):
                                         source["source"]["content"] = citation.cited_text

                                     if hasattr(citation, "encrypted_index"):
                                         source["source"]["metadata"] = {
                                             "encrypted_index": citation.encrypted_index
                                         }

                                # Fallback/General handling
                                if "name" not in source["source"] and hasattr(citation, "title"):
                                    source["source"]["name"] = citation.title
                                elif "url" in source["source"]:
                                    source["source"]["name"] = source["source"]["url"]
                                else:
                                    source["source"]["name"] = "Citation" # Default name

                                all_citations.append(source["source"])

                    # Handle code execution tool use
                    elif (
                        content_block.type == "server_tool_use"
                    ):
                        if content_block.name == "bash_code_execution":
                            if (
                                hasattr(content_block, "input")
                                and "command" in content_block.input
                            ):
                                command = content_block.input["command"]
                                result_parts.append(f"\n**Bash Command:**\n```bash\n{command}\n```\n")

                        elif content_block.name == "text_editor_code_execution":
                             if hasattr(content_block, "input"):
                                cmd = content_block.input.get("command", "")
                                path = content_block.input.get("path", "")
                                result_parts.append(f"\n**File Operation ({cmd}):** {path}\n")
                                if "file_text" in content_block.input:
                                    result_parts.append(f"```\n{content_block.input['file_text']}\n```\n")

                        elif content_block.name == "web_fetch":
                            if hasattr(content_block, "input") and "url" in content_block.input:
                                url = content_block.input["url"]
                                result_parts.append(f"\n**Web Fetch:** {url}\n")

                        elif content_block.name == "web_search":
                            result_parts.append(f"\n**Web Search:**\n")

                    # Handle code execution results
                    elif content_block.type == "bash_code_execution_tool_result":
                        if hasattr(content_block, "content"):
                            content = content_block.content
                            result_parts.append("\n**Output:**\n```\n")
                            if hasattr(content, "stdout") and content.stdout:
                                result_parts.append(content.stdout)
                            if hasattr(content, "stderr") and content.stderr:
                                result_parts.append(f"\n**Error:**\n{content.stderr}")
                            result_parts.append("\n```\n")

                    elif content_block.type == "text_editor_code_execution_tool_result":
                        if hasattr(content_block, "content"):
                            content = content_block.content
                            result_parts.append("\n**Output:**\n```\n")
                            if hasattr(content, "content") and content.content:
                                result_parts.append(content.content)
                            if hasattr(content, "lines") and content.lines:
                                result_parts.append("\n".join(content.lines))
                            if hasattr(content, "is_file_update"):
                                result_parts.append("File created." if not content.is_file_update else "File updated.")
                            result_parts.append("\n```\n")

                    elif content_block.type == "web_fetch_tool_result":
                        result_parts.append("Fetched.\n\n")

                if all_citations and self.event_emitter:
                     await self.event_emitter.emit_completion(sources=all_citations)

                # Emit usage
                if self.event_emitter and hasattr(response, "usage"):
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens

                    # Try to get web search count from usage if available, otherwise count properties could be used
                    # but simpler to rely on parsing content blocks if we tracked them.
                    # As a fallback, let's recount from content blocks for consistency or check usage
                    web_search_count = 0
                    if hasattr(response, "usage") and isinstance(response.usage, dict):
                         # check dict
                         pass

                    # Manual count from content blocks is reliable for intent
                    for block in response.content:
                        if block.type == "server_tool_use" and block.name == "web_search":
                             web_search_count += 1

                    total_cost = self._calculate_cost(input_tokens, output_tokens, params["model"], web_search_count)
                    completion_time = time.time() - start_time

                    usage_data = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "total_cost": total_cost,
                        "completion_time": round(completion_time, 2)
                    }

                    await self.event_emitter.emit_usage(usage_data)

                return "".join(result_parts)

            return ""
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
