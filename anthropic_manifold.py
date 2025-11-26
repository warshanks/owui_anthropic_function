"""
title: Anthropic Manifold Pipe
authors: warshanks
author_url: https://github.com/warshanks
funding_url: https://github.com/warshanks
version: 0.7.0
license: MIT

This pipe provides access to Anthropic's Claude models with support for:
- Web search capabilities
- Code execution in a secure sandbox environment
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


class Pipe:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")
        REQUIRE_USER_API_KEY: bool = Field(
            default=False,
            description="Whether to require user's own API key (applies to admins too).",
        )
        THINKING_BUDGET: int = Field(
            default=16000,
            description="Token budget for Claude's extended thinking capability (max tokens to use for thinking).",
        )
        MAX_TOKENS: int = Field(
            default=4096,
            description="Default maximum number of tokens to generate in the response.",
        )
        ENABLE_CODE_EXECUTION: bool = Field(
            default=True,
            description="Enable Claude's code execution capability for supported models.",
        )
        ENABLE_THINKING: bool = Field(
            default=True,
            description="Enable Claude's extended thinking capability for supported models.",
        )

    class UserValves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(default="")
        THINKING_BUDGET: int = Field(default=16000)
        MAX_TOKENS: int = Field(default=4096)
        ENABLE_CODE_EXECUTION: bool = Field(default=True)
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

        # Centralized model capability configuration
        self.MODEL_CAPABILITIES = {
            # Web Search: According to https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
            "web_search": {
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
            # Code Execution: According to https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/code-execution-tool
            "code_execution": {
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
        pass

    def get_anthropic_models(self):
        return [
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
        self, user_valves=None, model_name=None, code_execution_enabled=True
    ):
        """Initialize Anthropic client with appropriate API key and beta headers."""
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

            if model_name and code_execution_enabled:
                # Check if the model supports code execution
                if self.supports_capability(model_name, "code_execution"):
                    beta_headers.append("code-execution-2025-05-22")

            # Create default headers
            default_headers = {}
            if beta_headers:
                default_headers["anthropic-beta"] = ",".join(beta_headers)

            self.client = anthropic.Anthropic(
                api_key=api_key,
                default_headers=default_headers if default_headers else None,
            )

    def pipe(
        self, body: dict, __user__=None, **kwargs
    ) -> Union[str, Generator, Iterator]:
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

        # Check if code execution is enabled in valves
        code_execution_enabled = (
            user_valves.ENABLE_CODE_EXECUTION
            if user_valves and hasattr(user_valves, "ENABLE_CODE_EXECUTION")
            else self.valves.ENABLE_CODE_EXECUTION
        )

        # Check if web search is enabled in the UI
        features = body.get("features")
        web_search_enabled = (
            features.get("web_search", False)
            if isinstance(features, dict)
            else False
        )
        # Disable OWUI web search
        features["web_search"] = False

        # Check if thinking is enabled in valves
        thinking_enabled = (
            user_valves.ENABLE_THINKING
            if user_valves and hasattr(user_valves, "ENABLE_THINKING")
            else self.valves.ENABLE_THINKING
        )

        self.init_client(user_valves, model_name, code_execution_enabled)

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
                thinking_match = re.search(
                    r"<think>(.*?)(?:<!-- signature: (.*?) -->)?\s*</think>",
                    content_text,
                    re.DOTALL
                )

                if thinking_match and message["role"] == "assistant":
                    thinking_content = thinking_match.group(1).strip()
                    signature = thinking_match.group(2)

                    # Remove the thinking block from the text content
                    text_content = re.sub(
                        r"<think>.*?</think>\s*",
                        "",
                        content_text,
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
            tools.append({"type": "code_execution_20250522", "name": "code_execution"})

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

    def stream_response_sdk(self, params):
        try:
            with self.client.messages.stream(**params) as stream:
                # For extended thinking: handle thinking blocks in the stream
                self.is_thinking = False
                self.thinking_start_time = None
                self.thinking_signature = None
                # Track code execution state
                self.is_code_execution = False
                self.code_execution_block_index = None

                # Handle streaming events - use a hybrid approach for best compatibility
                processed_text_via_events = False
                # Track when we're in tool usage to add proper spacing after tools
                in_tool_usage = False

                for event in stream:
                    # Handle different event types
                    if hasattr(event, "type"):
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
                            if self.thinking_signature:
                                yield f"\n<!-- signature: {self.thinking_signature} -->"
                                self.thinking_signature = None

                            # Close the thinking block
                            yield "\n</think>\n\n"

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
                                and event.content_block.name == "code_execution"
                            ):
                                self.is_code_execution = True
                                self.code_execution_block_index = getattr(
                                    event, "index", None
                                )
                                yield "\n**Code Execution:**\n```python\n"

                        # Handle code execution input deltas
                        elif (
                            self.is_code_execution
                            and event.type == "content_block_delta"
                            and hasattr(event, "delta")
                            and event.delta.type == "input_json_delta"
                            and hasattr(event.delta, "partial_json")
                        ):
                            # Extract and yield code from the partial JSON
                            import json
                            import re

                            try:
                                # Try to extract the code value using regex as a fallback
                                code_match = re.search(
                                    r'"code":"([^"]*(?:\\.[^"]*)*)"',
                                    event.delta.partial_json,
                                )
                                if code_match:
                                    # Decode escaped characters in the code
                                    code_content = (
                                        code_match.group(1)
                                        .encode()
                                        .decode("unicode_escape")
                                    )
                                    yield code_content
                                else:
                                    # Try JSON parsing as a backup
                                    # Handle incomplete JSON by trying to parse what we have
                                    partial_json = event.delta.partial_json
                                    if not partial_json.endswith("}"):
                                        # Try to close the JSON
                                        if (
                                            '"code":"' in partial_json
                                            and not partial_json.endswith('"')
                                        ):
                                            partial_json += '"}'
                                        elif not partial_json.endswith("}"):
                                            partial_json += "}"

                                    partial_data = json.loads(partial_json)
                                    if "code" in partial_data:
                                        yield partial_data["code"]
                            except (json.JSONDecodeError, AttributeError):
                                # If parsing fails, continue without yielding
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
                            and event.content_block.type == "code_execution_tool_result"
                        ):
                            yield "\n**Output:**\n```\n"
                            if hasattr(event.content_block, "content"):
                                content = event.content_block.content
                                if hasattr(content, "stdout") and content.stdout:
                                    yield content.stdout
                                if hasattr(content, "stderr") and content.stderr:
                                    yield f"\n**Error:**\n{content.stderr}"
                            yield "\n```\n\n"

                        # Handle tool completion - add spacing when tool finishes
                        elif event.type == "content_block_stop" and in_tool_usage:
                            # Add a newline when tool block ends to ensure proper spacing
                            # This fixes the issue where tool results run into subsequent text
                            in_tool_usage = False
                            yield "\n"

                        # Handle regular text content deltas
                        elif (
                            event.type == "content_block_delta"
                            and hasattr(event, "delta")
                            and event.delta.type == "text_delta"
                            and hasattr(event.delta, "text")
                        ):
                            # Only yield text if we're not in thinking or code execution mode
                            if not self.is_thinking and not self.is_code_execution:
                                yield event.delta.text
                                processed_text_via_events = True

                # Fallback to text_stream if no text was processed via events
                # This ensures compatibility if the SDK behavior changes
                if not processed_text_via_events and hasattr(stream, "text_stream"):
                    for text in stream.text_stream:
                        if not self.is_thinking and not self.is_code_execution:
                            yield text

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

    def non_stream_response_sdk(self, params):
        try:
            response = self.client.messages.create(**params)

            # Handle different content types in the response
            if hasattr(response, "content") and response.content:
                result_parts = []

                for content_block in response.content:
                    # Handle thinking content blocks
                    if content_block.type == "thinking":
                        if (
                            hasattr(content_block, "thinking")
                            and content_block.thinking
                        ):
                            signature_part = ""
                            if hasattr(content_block, "signature") and content_block.signature:
                                signature_part = f"\n<!-- signature: {content_block.signature} -->"

                            result_parts.append(
                                f"<think>\n{content_block.thinking}{signature_part}\n</think>\n\n"
                            )

                    # Handle redacted thinking content blocks
                    elif content_block.type == "redacted_thinking":
                        result_parts.append(
                            "\n*[Some reasoning has been encrypted for safety]*\n\n"
                        )

                    # Handle text content
                    elif content_block.type == "text":
                        result_parts.append(content_block.text)

                    # Handle code execution tool use
                    elif (
                        content_block.type == "server_tool_use"
                        and content_block.name == "code_execution"
                    ):
                        if (
                            hasattr(content_block, "input")
                            and "code" in content_block.input
                        ):
                            code = content_block.input["code"]
                            result_parts.append(f"\n```python\n{code}\n```\n")

                    # Handle code execution results
                    elif content_block.type == "code_execution_tool_result":
                        if hasattr(content_block, "content"):
                            content = content_block.content
                            result_parts.append("\n**Output:**\n```\n")
                            if hasattr(content, "stdout") and content.stdout:
                                result_parts.append(content.stdout)
                            if hasattr(content, "stderr") and content.stderr:
                                result_parts.append(f"\n**Error:**\n{content.stderr}")
                            result_parts.append("\n```\n")

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
