"""
NeuralChat — Application Flask avec authentification.

Structure du projet :
    app.py              ← point d'entrée (ce fichier)
    config.py           ← constantes, clé API, modèles disponibles
    database.py         ← init DB, load/save sessions
    llm.py              ← appel OpenRouter, recherche DuckDuckGo
    session_helpers.py  ← get_session(), build_system_content(), constantes de langue
    templates_html.py   ← charge login.html et chat.html
    routes_auth.py      ← /login, /api/auth/*
    routes_chat.py      ← /, /api/chat, /api/history, /api/clear, /api/context, /api/skills
    routes_warroom.py   ← /api/warroom, /api/advocate
    login.html          ← template page de connexion
    chat.html           ← template page de chat principale
"""
from __future__ import annotations

import os

from flask import Flask

from config.py import logger
from database.py import init_db
from routes_auth.py import auth_bp
from routes_chat.py import chat_bp
from routes_warroom.py import warroom_bp

# ---------------------------------------------------------------------------
# Création de l'application Flask
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    application = Flask(__name__)
    application.config["MAX_CONTENT_LENGTH"]  = 50 * 1024 * 1024
    application.config["MAX_FORM_MEMORY_SIZE"] = 50 * 1024 * 1024
    # BUG FIX : utiliser une variable d'environnement pour la clé secrète en prod
    application.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production-use-os-urandom")

    application.register_blueprint(auth_bp)
    application.register_blueprint(chat_bp)
    application.register_blueprint(warroom_bp)

    return application


app = create_app()

# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("NeuralChat with auth — starting")
    init_db()

    port = int(os.environ.get("PORT", 5000))
    # BUG FIX : debug=False en production (ne jamais exposer le debugger)
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
