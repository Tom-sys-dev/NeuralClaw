from __future__ import annotations

import json
import sqlite3
from contextlib import closing

from config import DB_PATH, DEFAULT_MODEL, logger


# ---------------------------------------------------------------------------
# Connexion
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Retourne une connexion à la base de données avec Row factory."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Crée les tables si elles n'existent pas encore."""
    with closing(get_db()) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username   TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email      TEXT,
                created_at TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                username      TEXT PRIMARY KEY,
                messages      TEXT NOT NULL,
                system_prompt TEXT,
                model         TEXT NOT NULL,
                skills        TEXT NOT NULL,
                lang          TEXT NOT NULL DEFAULT 'fr',
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)

        db.execute("CREATE INDEX IF NOT EXISTS idx_users_username    ON users(username)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_username ON chat_sessions(username)")

        db.commit()
        logger.info("Base de données initialisée")


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def _default_session() -> dict:
    return {
        "messages":      [],
        "system_prompt": "",
        "model":         DEFAULT_MODEL,
        "skills":        [],
        "lang":          "fr",
    }


def load_session_from_db(username: str) -> dict:
    """
    Charge la session d'un utilisateur depuis la DB.
    Crée une session vide si elle n'existe pas encore.
    """
    with closing(get_db()) as db:
        row = db.execute(
            "SELECT * FROM chat_sessions WHERE username = ?", (username,)
        ).fetchone()

        if row is None:
            sess = _default_session()
            db.execute(
                "INSERT INTO chat_sessions (username, messages, system_prompt, model, skills, lang) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    username,
                    json.dumps(sess["messages"]),
                    sess["system_prompt"],
                    sess["model"],
                    json.dumps(sess["skills"]),
                    sess["lang"],
                ),
            )
            db.commit()
            return sess

        try:
            return {
                "messages":      json.loads(row["messages"]),
                "system_prompt": row["system_prompt"] or "",
                "model":         row["model"],
                "skills":        json.loads(row["skills"]),
                "lang":          row["lang"] or "fr",
            }
        except json.JSONDecodeError as exc:
            logger.error("Erreur décodage JSON pour %s : %s", username, exc)
            return _default_session()


def save_session_to_db(username: str, sess: dict) -> None:
    """Persiste la session d'un utilisateur (ignoré pour les anonymes)."""
    if username == "__anon__":
        return

    with closing(get_db()) as db:
        exists = db.execute(
            "SELECT 1 FROM chat_sessions WHERE username = ?", (username,)
        ).fetchone()

        if exists is None:
            db.execute(
                "INSERT INTO chat_sessions (username, messages, system_prompt, model, skills, lang) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    username,
                    json.dumps(sess["messages"]),
                    sess["system_prompt"],
                    sess["model"],
                    json.dumps(sess["skills"]),
                    sess.get("lang", "fr"),
                ),
            )
        else:
            db.execute(
                "UPDATE chat_sessions "
                "SET messages=?, system_prompt=?, model=?, skills=?, lang=? "
                "WHERE username=?",
                (
                    json.dumps(sess["messages"]),
                    sess["system_prompt"],
                    sess["model"],
                    json.dumps(sess["skills"]),
                    sess.get("lang", "fr"),
                    username,
                ),
            )
        db.commit()
