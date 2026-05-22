# 📚 Study Partner Agent

An AI-powered study companion that helps you learn **anything** — from data science to Japanese. Built with LangGraph, Groq (open-source LLMs), and production-grade reliability patterns.

## ✨ Features

| Feature | Description |
|:---|:---|
| 🔍 **Resource Finder** | Searches the web + Reddit for free courses, books, videos, and tutorials — with verified links |
| 📅 **Study Planner** | Creates structured study plans with timelines, milestones, and self-assessment checkpoints |
| 🎓 **Concept Explainer** | Answers questions with web-sourced citations — every claim backed by a URL |
| 🧠 **Quiz Master** | Generates interactive multiple-choice quizzes to reinforce learning through active recall |
| 💾 **Memory** | Remembers your conversation across sessions (persistent SQLite storage) |
| ✅ **Verified Responses** | Every URL is checked against actual search results — no hallucinated links |

## 🚀 Quick Start (3 Steps)

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
```

Edit `.env` and paste your **free** Groq API key:
- Get one at: https://console.groq.com (no credit card required)

### 3. Run the agent

```bash
chainlit run app.py
```

Open http://localhost:8000 in your browser. That's it! 🎉

## 🖥️ Local Mode (Ollama — No API Key Needed)

Want to run 100% locally? Use [Ollama](https://ollama.ai):

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model
ollama pull qwen3:4b

# Set local mode in your .env
echo "USE_OLLAMA=true" >> .env

# Run the agent
chainlit run app.py
```

> **Note:** Local models are slower and less capable than Groq's cloud inference. Recommended for offline use or when you want zero external dependencies.

## 🔍 Optional: Google Search

By default, the agent uses **DuckDuckGo** (free, no API key). For Google search results:

1. Sign up at https://serper.dev (2,500 free searches/month)
2. Add to your `.env`:
   ```
   SERPER_API_KEY=your_key_here
   ```

## 🏗️ Architecture

```
User Message
    │
    ▼
📝 Summarize ──→ 🧭 Router (intent classification)
                     │
                     ├──→ 🔍 Resource Finder (web + Reddit search)
                     ├──→ 📅 Study Planner (structured plans)
                     ├──→ 🎓 Professor (Q&A with web citations)
                     ├──→ 🧠 Quiz Master (active recall quizzes)
                     └──→ 💬 General Chat
                              │
                              ▼
                     ✅ Verification Node (check citations)
                              │
                              ▼
                         Response to User
```

### Quality Patterns

- **Source Grounding**: URLs can only come from actual search results
- **Citation Verification**: Rule-based URL checking before every response
- **Structured Outputs**: Pydantic schemas for predictable, verifiable data
- **Confidence & Abstention**: Agent says "I don't know" rather than hallucinating
- **Token Optimization**: Conversation summarization + message trimming
- **Execution Guardrails**: Max retries, loop limits, token budgets

## 📁 Project Structure

```
study-agent/
├── app.py                # Chainlit UI entry point
├── chainlit.md           # Welcome screen
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
├── agent/
│   ├── graph.py          # LangGraph state machine
│   ├── state.py          # Agent state schema
│   ├── nodes.py          # Node implementations
│   ├── tools.py          # Search tools (web, Reddit)
│   ├── prompts.py        # System prompts (versioned)
│   ├── schemas.py        # Pydantic output schemas
│   ├── llm.py            # LLM provider (Groq + Ollama)
│   ├── memory.py         # SQLite persistence + trim
│   └── verification.py   # Citation verification
└── data/
    └── study_plans/      # Saved study plans (Markdown)
```

## 🔧 Configuration

All configuration is via environment variables in `.env`:

| Variable | Required | Default | Description |
|:---|:---|:---|:---|
| `GROQ_API_KEY` | Yes* | — | Groq API key ([get free](https://console.groq.com)) |
| `SERPER_API_KEY` | No | — | Serper key for Google search |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model to use |
| `OLLAMA_MODEL` | No | `qwen3:4b` | Ollama model to use |
| `USE_OLLAMA` | No | `false` | Set to `true` for local mode |

*Not required if using Ollama.

## 📝 Tech Stack

- **LLM**: [Groq](https://groq.com) (Llama 3.3 70B) / [Ollama](https://ollama.ai) (Qwen3 4B)
- **Agent Framework**: [LangGraph](https://langchain-ai.github.io/langgraph/)
- **Web Search**: [DuckDuckGo](https://duckduckgo.com) / [Serper](https://serper.dev)
- **Memory**: SQLite via LangGraph checkpointer
- **UI**: [Chainlit](https://chainlit.io)
- **Validation**: [Pydantic](https://pydantic.dev)

## 📄 License

MIT — feel free to use, modify, and share.
