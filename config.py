from __future__ import annotations

import os
import logging

# ---------------------------------------------------------------------------
# API & Modèles
# ---------------------------------------------------------------------------
# Fallback hardcodé — préférer la variable d'environnement OPENROUTER_API_KEY
API_KEY = os.environ.get("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b:free"

FREE_MODELS: list[dict[str, str]] = [
    {"id": "minimax/minimax-m2.5:free",                        "label": "MiniMax M2.5"},
    {"id": "openrouter/elephant-alpha",                      "label": "elephant"},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free",           "label": "Nemotron 3"},
    {"id": "arcee-ai/trinity-large-preview:free",              "label": "Trinity"},
    {"id": "openai/gpt-oss-120b:free",              "label": "GPT-oss"},
]

# ---------------------------------------------------------------------------
# Base de données
# ---------------------------------------------------------------------------
DB_PATH = "neuralchat.db"

# ---------------------------------------------------------------------------
# Keep-alive (Render)
# ---------------------------------------------------------------------------
PING_URL = "https://neuralclaw.onrender.com"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
