# Agentic Browser

A Linux-first **dual-domain agent** that controls both your **web browser** (via Playwright) and your **local Linux system** (via shell commands). Now powered by **LangGraph** for robust multi-agent orchestration.

## 🚀 Key Features

*   **Multi-Agent Architecture**: Built on [LangGraph](https://langchain-ai.github.io/langgraph/), featuring a Supervisor agent that intelligently routes tasks to specialized Browser and OS agents.
*   **Dual-Domain Control**: Seamlessly switches between web browsing and local system operations.
*   **Multi-Provider Support**: Works with: 
    *   **OpenAI** (GPT-4o, o1)
    *   **Anthropic** (Claude 3.5 Sonnet)
    *   **Google** (Gemini 1.5 Pro/Flash)
    *   **LM Studio** (Local LLMs like Llama 3, Qwen 2.5)
*   **Browser Automation**: Full control via Playwright (navigation, clicking, typing, extraction).
*   **OS Integration**: Safe execution of shell commands and file operations with risk-based guardrails.
*   **Modern GUI**: Dark-themed graphical interface for monitoring agent progress.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/RecursiveIntell/agentic-browser.git
cd agentic-browser

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install the package
pip install -e ".[dev]"

# Install Playwright browsers
python -m playwright install chromium
```

## ⚡ Quick Start

### GUI Mode (Recommended)
Launch the graphical interface to interact with the agent visually:
```bash
agentic-browser gui
```

### CLI Mode
Run the agent directly from the terminal:
```bash
# Browser task
agentic-browser run "Search DuckDuckGo for LangGraph tutorials"

# OS task
agentic-browser run "List files in my downloads folder"

# Complex dual-domain task
agentic-browser run "Download the latest python logo and save it to my pictures folder"
```

## 🛠️ Configuration

Settings are stored in `~/.agentic_browser/settings.json`.

### CLI Options
| Flag | Description |
|------|-------------|
| `--model-endpoint` | Custom LLM API endpoint URL |
| `--model` | Model name (e.g., `gpt-4o`, `claude-3-5-sonnet`) |
| `--headless` | Run browser in headless mode (default: false) |
| `--auto-approve` | Skip approval prompts for medium-risk actions |
| `--fresh-profile` | Start with a clean browser profile |
| `--enable-tracing` | Save Playwright traces for debugging |
| `--legacy` | Use the old single-agent implementation |

### Environment Variables
```bash
# Provider Configuration
AGENTIC_BROWSER_PROVIDER # one of: openai, anthropic, google, lm_studio (default)

# API Keys (if not using local models)
OPENAI_API_KEY
ANTHROPIC_API_KEY
GOOGLE_API_KEY
```

## 🏗️ Architecture

The system uses a graph-based architecture:

1.  **Supervisor Node**: Analyzes the user's goal and current state to decide which specialist to call next.
2.  **Browser Agent**: Handles internet-facing tasks using Playwright tools.
3.  **OS Agent**: Handles local system tasks using shell tools.
4.  **Safety Layer**: Intercepts actions to enforce permissions and prompt for user approval on risky operations.

## 🛡️ Safety & Permissions

The agent creates a sandbox but operates with your user permissions. To prevent accidents, it classifies actions by risk:

*   **HIGH RISK** (Always asks): `rm`, `sudo`, `dd`, writing to system paths.
*   **MEDIUM RISK** (Asks unless auto-approve): Writing files to home dir, running scripts.
*   **LOW RISK** (Allowed): `ls`, `cat`, reading files, web navigation.

## 🐛 Troubleshooting

*   **Browser doesn't start**: Ensure you ran `playwright install`.
*   **Permission errors**: The agent cannot do what your user account cannot do.
*   **Model errors**: Check your API keys and internet connection. For local models, ensure LM Studio/Ollama is running.

## 📄 License

MIT License - see [LICENSE](LICENSE)
