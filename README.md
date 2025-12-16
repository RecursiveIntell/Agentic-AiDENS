# Agentic Browser

A Linux-first agentic browser runner that controls Chromium via Playwright and executes user goals step-by-step with LLM-powered decision making and safety guardrails.

## Features

- 🌐 **Browser Automation**: Controls Chromium with persistent profiles (cookies/logins preserved)
- 🤖 **LLM-Powered**: Uses any OpenAI-compatible API (LM Studio, OpenAI, Anthropic, Google AI)
- 🛡️ **Safety Guardrails**: Risk classification with approval prompts for dangerous actions
- 📝 **Comprehensive Logging**: Saves every step, screenshot, and decision to artifacts
- 🔄 **Self-Recovery**: Automatically handles errors and retries with alternative approaches
- 🎯 **Goal-Oriented**: Natural language goals executed through iterative planning
- 🖥️ **Dark Theme GUI**: Modern graphical interface with real-time output
- 🔗 **Smart Navigation**: Tracks visited URLs to avoid revisiting same pages
- ⏱️ **Rate Limiting**: Exponential backoff prevents API throttling

## Installation

### Prerequisites

- Linux (tested on Fedora/Nobara)
- Python 3.10+ (3.11+ recommended)
- An LLM server (LM Studio, OpenAI, Anthropic, or Google AI)

### Install from source

```bash
# Clone the repository
git clone https://github.com/RecursiveIntell/agentic-browser.git
cd agentic-browser

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install the package
pip install -e ".[dev]"

# Install Playwright Chromium
python -m playwright install chromium
```

## Quick Start

### GUI Mode (Recommended)

```bash
agentic-browser gui
```

This opens a dark-themed graphical interface where you can:
- Enter goals in natural language
- Configure your LLM provider and API key
- Watch the agent execute steps in real-time with color-coded output
- See visited URLs and extracted data
- Monitor step progress in the status bar

### CLI Mode

```bash
# Simple example
agentic-browser run "Open example.com and tell me the title"

# Search the web (DuckDuckGo recommended - avoids CAPTCHAs)
agentic-browser run "Search DuckDuckGo for Playwright and summarize the results"

# With specific model
agentic-browser run "Check the weather" --model gpt-4o-mini

# Headless mode
agentic-browser run "Scrape headlines from news.ycombinator.com" --headless
```

## GUI Features

### Dark Theme Interface

The GUI features a modern dark theme with:
- Color-coded log output (info, success, warning, error, actions)
- Real-time step counter
- Debug information (provider, model, endpoint)
- Status bar with current activity

### Settings Dialog

| Setting | Description |
|---------|-------------|
| **Provider** | LM Studio, OpenAI, Anthropic, or Google AI |
| **API Key** | Required for cloud providers (not needed for LM Studio) |
| **Model** | Model name (click 🔄 Refresh to fetch available models) |
| **Custom Endpoint** | Override the default API endpoint |
| **Profile Name** | Browser profile for persistent sessions |
| **Max Steps** | Maximum actions before stopping |
| **Headless** | Run browser without visible window |
| **Auto-Approve** | Skip approval for medium-risk actions |

### Provider Configuration

| Provider | Default Endpoint | API Key Required |
|----------|-----------------|------------------|
| LM Studio (Local) | http://127.0.0.1:1234/v1 | No |
| OpenAI | https://api.openai.com/v1 | Yes |
| Anthropic | https://api.anthropic.com/v1 | Yes |
| Google AI | https://generativelanguage.googleapis.com/v1beta | Yes |

## Smart Agent Features

### Visited URL Tracking

The agent tracks every URL visited during a session and displays them to the LLM to prevent revisiting the same pages. This is especially useful for tasks like "check the top 5 results."

### Search Workflow

The agent understands search patterns:
1. Navigate to search engine
2. Type search query
3. Press Enter to submit
4. Extract results or click links

**Tip**: Use DuckDuckGo instead of Google to avoid CAPTCHAs.

### Failure Recovery

When an action fails, the agent:
- Shows the failed selector in context
- Suggests alternative approaches (text-based selectors)
- Advises using `done` if the goal can be answered from visible text

### Rate Limiting

Built-in exponential backoff (2s → 4s → 8s) prevents API rate limit errors with cloud providers.

## CLI Reference

```
agentic-browser run "GOAL" [OPTIONS]
agentic-browser gui
```

### Run Options

| Option | Default | Description |
|--------|---------|-------------|
| `--profile NAME` | `default` | Browser profile name |
| `--headless` | `false` | Run browser in headless mode |
| `--max-steps N` | `30` | Maximum steps before stopping |
| `--model-endpoint URL` | `http://127.0.0.1:1234/v1` | LLM API endpoint |
| `--model NAME` | `qwen2.5:7b` | LLM model name |
| `--auto-approve` | `false` | Auto-approve medium-risk actions |
| `--fresh-profile` | `false` | Create a fresh browser profile |
| `--no-persist` | `false` | Use a temporary profile |
| `--enable-tracing` | `false` | Enable Playwright tracing |

### Environment Variables

- `AGENTIC_BROWSER_ENDPOINT`: Default LLM endpoint
- `AGENTIC_BROWSER_MODEL`: Default model name  
- `AGENTIC_BROWSER_API_KEY`: API key (if required)

## Safety Model

The agent classifies every action into risk levels:

### High Risk (always requires approval)
- Purchase buttons: "Buy", "Checkout", "Pay", "Order"
- Account actions: "Delete", "Remove", "Cancel Account"
- Messaging: "Send", "Post", "Submit"
- Payment domains: PayPal, Stripe, etc.

### Medium Risk (requires approval unless `--auto-approve`)
- Login forms (password fields)
- File uploads
- Permission grants

### Low Risk (no approval needed)
- Navigation, scrolling, reading, screenshots

## Artifacts and Logging

Every run creates artifacts at:
```
~/.agentic_browser/runs/<timestamp>_<goal_slug>/
├── steps.jsonl          # Every step with state, action, result
├── screenshots/         # Screenshots at each step
├── snapshots/           # HTML/text page snapshots
└── trace.zip           # Playwright trace (if enabled)
```

Settings are stored at:
```
~/.agentic_browser/settings.json
```

Browser profiles are stored at:
```
~/.agentic_browser/profiles/<profile_name>/
```

## Action Schema

The LLM responds with JSON actions:

```json
{
  "action": "goto|click|type|press|scroll|wait_for|extract|extract_visible_text|screenshot|back|forward|done",
  "args": { ... },
  "rationale": "short reason",
  "risk": "low|medium|high",
  "requires_approval": true/false,
  "final_answer": "only when action=done"
}
```

### Available Actions

| Action | Arguments | Description |
|--------|-----------|-------------|
| `goto` | `url` | Navigate to URL |
| `click` | `selector`, `timeout_ms` | Click element (with fallback selectors) |
| `type` | `selector`, `text`, `clear_first` | Type into input (auto-detects search boxes) |
| `press` | `key` | Press keyboard key (waits for navigation on Enter) |
| `scroll` | `amount` | Scroll (positive=down) |
| `wait_for` | `selector`, `timeout_ms` | Wait for element |
| `extract` | `selector`, `attribute` | Extract element data |
| `extract_visible_text` | `max_chars` | Get visible text |
| `screenshot` | `label` | Capture screenshot |
| `back` | (none) | Navigate back |
| `forward` | (none) | Navigate forward |
| `done` | `summary_style` | Complete task with answer |

## Project Structure

```
agentic_browser/
├── cli.py              # Command-line interface
├── config.py           # Configuration management
├── agent.py            # Main agent loop with state tracking
├── llm_client.py       # LLM API client with retry and rate limiting
├── tools.py            # Browser action implementations
├── safety.py           # Risk classification
├── logger.py           # Logging and artifacts
├── utils.py            # Utility functions
├── providers.py        # LLM provider configuration
├── model_fetcher.py    # Fetch models from provider APIs
├── settings_store.py   # Persistent settings storage
└── gui/
    ├── main_window.py      # Dark theme main window
    ├── settings_dialog.py  # Settings configuration
    └── approval_dialog.py  # Action approval prompts
```

## Troubleshooting

### "Connection refused" to LLM endpoint
- Ensure your LLM server is running
- Check the endpoint URL in settings

### Browser doesn't start
- Run `python -m playwright install chromium`
- Check for system dependencies: `playwright install-deps chromium`

### "429 Too Many Requests"
- Built-in rate limiting should handle this automatically
- If persists, wait a few minutes or use a local LLM (LM Studio)

### Agent loops on same action
- The agent tracks visited URLs and action history
- Try a more capable model (GPT-4o, Llama 3.1 8B+)
- Reduce task complexity

### Google CAPTCHA
- Use DuckDuckGo instead
- Or solve the CAPTCHA manually once in the persistent profile

### GUI doesn't launch
- Ensure PySide6 is installed: `pip install pyside6`
- On Wayland, try: `QT_QPA_PLATFORM=xcb agentic-browser gui`

## Recommended Models

For best results, use a capable instruction-following model:

| Provider | Recommended Model | Notes |
|----------|------------------|-------|
| LM Studio | Llama 3.1 8B, Qwen 2.5 7B | Free, local, no rate limits |
| OpenAI | gpt-4o-mini, gpt-4o | Best instruction following |
| Anthropic | claude-3-sonnet | Good reasoning |
| Google | gemini-1.5-flash | Fast and capable |

## License

MIT License - see [LICENSE](LICENSE)
