from __future__ import annotations

from typing import Any

from flask import session as flask_session

from config.py import DEFAULT_MODEL
from database.py import load_session_from_db

# ---------------------------------------------------------------------------
# Store en mémoire pour les utilisateurs anonymes
# ---------------------------------------------------------------------------
_anon_store: dict[str, dict[str, Any]] = {}


def get_session() -> dict:
    """
    Retourne la session de chat pour l'utilisateur courant.
    - Anonymes  : stockée dans le dict en mémoire.
    - Connectés : chargée depuis la base de données.
    """
    username: str = flask_session.get("username", "__anon__")

    if username == "__anon__":
        if "__anon__" not in _anon_store:
            _anon_store["__anon__"] = {
                "messages":      [],
                "system_prompt": "",
                "model":         DEFAULT_MODEL,
                "skills":        [],
                "lang":          "fr",
            }
        return _anon_store["__anon__"]

    return load_session_from_db(username)


def build_system_content(sess: dict) -> str:
    """Assemble le contenu du message système à partir des skills et du prompt."""
    parts: list[str] = []
    if sess.get("skills"):
        parts.append(f"Skills and areas of expertise: {', '.join(sess['skills'])}.")
    if sess.get("system_prompt"):
        parts.append(sess["system_prompt"])
    return "\n\n".join(parts)


def estimate_tokens(messages: list[dict]) -> int:
    """Estimation grossière du nombre de tokens (÷ 4 caractères)."""
    return sum(len(m.get("content", "")) for m in messages) // 4


# ---------------------------------------------------------------------------
# Constantes de langue partagées
# ---------------------------------------------------------------------------

LANG_INSTRUCTIONS: dict[str, str] = {
    "fr": "Tu réponds TOUJOURS en français, quelle que soit la langue du message reçu.",
    "en": "You ALWAYS respond in English, regardless of the language of the received message.",
    "es": "Respondes SIEMPRE en español, independientemente del idioma del mensaje recibido.",
    "de": "Du antwortest IMMER auf Deutsch, unabhängig von der Sprache der empfangenen Nachricht.",
    "pt": "Você responde SEMPRE em português, independentemente do idioma da mensagem recebida.",
    "ar": "أنت ترد دائماً باللغة العربية، بغض النظر عن لغة الرسالة المستلمة.",
    "zh": "你始终用中文回复，不管收到的消息是什么语言。",
    "ja": "受け取ったメッセージの言語に関わらず、常に日本語で返答してください。",
}

# Variante courte (utilisée dans la War Room)
LANG_RULES_SHORT: dict[str, str] = {
    "fr": "Réponds TOUJOURS en français, sans exception.",
    "en": "Always respond in English, without exception.",
    "es": "Responde SIEMPRE en español, sin excepción.",
    "de": "Antworte IMMER auf Deutsch, ohne Ausnahme.",
    "pt": "Responda SEMPRE em português, sem exceção.",
    "ar": "أجب دائماً باللغة العربية، دون استثناء.",
    "zh": "始终用中文回答，没有例外。",
    "ja": "常に日本語で答えてください、例外なく。",
}
