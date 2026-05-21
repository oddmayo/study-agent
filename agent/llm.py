"""LLM provider layer with Groq (cloud) and Ollama (local) support.

This module provides a single `get_llm()` function that returns the
configured chat model. It defaults to Groq (fast cloud inference with
Llama 3.3 70B) and falls back to Ollama (local, free) when configured
or when Groq is unavailable.
"""

import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default models — can be overridden via .env
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"


def _is_ollama_available() -> bool:
    """Check if Ollama is running locally by pinging its API."""
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://localhost:11434/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_llm(streaming: bool = True):
    """Get the configured LLM instance.

    Priority:
    1. If USE_OLLAMA=true → use Ollama
    2. If GROQ_API_KEY is set → use Groq
    3. If Ollama is running locally → use Ollama as fallback
    4. Raise an error with setup instructions

    Args:
        streaming: Enable token-by-token streaming (needed for Chainlit UI).

    Returns:
        A LangChain ChatModel instance.
    """
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    # Path 1: User explicitly wants Ollama
    if use_ollama:
        return _get_ollama(ollama_model, streaming)

    # Path 2: Groq API key is available
    if groq_key and groq_key != "your_groq_api_key_here":
        return _get_groq(groq_key, groq_model, streaming)

    # Path 3: No Groq key — try Ollama as automatic fallback
    if _is_ollama_available():
        logger.warning(
            "No GROQ_API_KEY found. Falling back to Ollama (%s). "
            "For faster responses, add your Groq key to .env",
            ollama_model,
        )
        return _get_ollama(ollama_model, streaming)

    # Path 4: Nothing available
    raise RuntimeError(
        "\n\n❌ No LLM provider available!\n\n"
        "Option 1 (Cloud — recommended):\n"
        "  1. Get a free API key at https://console.groq.com\n"
        "  2. Copy .env.example to .env\n"
        "  3. Paste your key in GROQ_API_KEY=...\n\n"
        "Option 2 (Local):\n"
        "  1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh\n"
        "  2. Pull a model: ollama pull qwen3:4b\n"
        "  3. Set USE_OLLAMA=true in .env\n"
    )


def _get_groq(api_key: str, model: str, streaming: bool):
    """Initialize the Groq chat model."""
    from langchain_groq import ChatGroq

    logger.info("Using Groq (%s)", model)
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0.3,
        streaming=streaming,
        max_retries=2,
    )


def _get_ollama(model: str, streaming: bool):
    """Initialize the Ollama chat model."""
    from langchain_ollama import ChatOllama

    if not _is_ollama_available():
        raise RuntimeError(
            f"\n\n❌ Ollama is not running!\n\n"
            f"Start it with: ollama serve\n"
            f"Then pull the model: ollama pull {model}\n"
        )

    logger.info("Using Ollama (%s)", model)
    return ChatOllama(
        model=model,
        temperature=0.3,
        streaming=streaming,
    )
