# Anthropic Manifold Pipe for Open WebUI

![Version](https://img.shields.io/badge/version-0.11.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This pipe provides seamless integration with Anthropic's Claude models for Open WebUI, enabling advanced capabilities like web search, secure code execution, and extended thinking.

## Features

- **Web Search**: Enable Claude to search the web for real-time information.
- **Web Fetch**: Fetch and process content from specific URLs for deeper analysis.
- **Code Execution**: Run Python code in Anthropic's secure sandbox environment for calculations, data analysis, and more.
- **Extended Thinking**: Leverage Claude's extended thinking capabilities for complex problem-solving, with configurable token budgets.
- **Image Processing**: Analyze images with support for both URL and base64 inputs (up to 5MB).
- **Streaming Support**: Real-time streaming of responses, including thinking blocks and code execution outputs.
- **Cost Tracking**: Track and display the cost of requests in real-time.
- **Flexible Configuration**: Manage capabilities globally or per-user via Valves.

## Supported Models

The pipe automatically handles capabilities for various Claude models, including:

- **Claude Sonnet** (`claude-sonnet-4-5-20250929`)
- **Claude Haiku** (`claude-haiku-4-5-20251001`)
- **Claude Opus** (`claude-opus-4-5-20251101`, `claude-opus-4-1-20250805`, etc.)

## Configuration

You can configure the pipe using **Valves**. These can be set globally by the admin or individually by users (if allowed).

## Installation

1. Open your Open WebUI instance.
2. Go to **Admin Panel** -> **Functions**.
3. Click the **+ New Function** button.
4. Import the `anthropic_manifold.py` file or paste its content.
5. Save and activate the function.

## Usage

### Extended Thinking
When using a supported model, the model may utilize "thinking" blocks to reason through complex problems before answering. These blocks are displayed as:
```
<think>
... reasoning process ...
</think>
```
*Note: Thinking mode is only available for new conversations and requires a minimum budget of 1,024 tokens.*

### Code Execution
If enabled, Claude can write and execute Python code. The code and its output (stdout/stderr) will be displayed in the chat:
```python
print("Hello World")
```
**Output:**
```
Hello World
```

### Web Search
Claude can perform web searches to fetch up-to-date information when enabled via the UI's web search toggle.

### Web Fetch
When enabled via the URL context toggle in the UI, Claude can fetch and analyze content from specific URLs. This is useful for reading documentation, analyzing web pages, or extracting information from specific websites.

Uses [suurt8ll's gemini_url_context_toggle filter](https://github.com/suurt8ll/open_webui_functions/blob/master/plugins/filters/gemini_url_context_toggle.py) to enable in the UI.

## License

MIT License - see the [LICENSE](LICENSE) file for details.
