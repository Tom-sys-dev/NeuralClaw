from __future__ import annotations

import json
import re
import logging
from typing import List, Dict
import time
import uuid
from typing import Any
import os
import requests
from flask import Flask, jsonify, render_template_string, request, session, redirect
import urllib.parse
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from contextlib import closing

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "minimax/minimax-m2.5:free"

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 50 * 1024 * 1024
app.secret_key = "change-me-in-production-use-os-urandom"

# ---------------------------------------------------------------------------
# Base de données
# ---------------------------------------------------------------------------
DB_PATH = "neuralchat.db"


def get_db():
    """Retourne une connexion à la base de données"""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    """Initialise la base de données avec les tables nécessaires"""
    with closing(get_db()) as db:
        # Table des utilisateurs
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Table des sessions de chat
        db.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                username TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                system_prompt TEXT,
                model TEXT NOT NULL,
                skills TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            )
        """)

        # Index pour améliorer les performances
        db.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_username ON chat_sessions(username)")

        db.commit()
        logger.info("Base de données initialisée")


def save_session_to_db(username: str, sess: dict):
    """Sauvegarde la session d'un utilisateur dans la base de données"""
    if username == "__anon__":
        return

    with closing(get_db()) as db:
        # Vérifier si la session existe déjà
        cur = db.execute(
            "SELECT 1 FROM chat_sessions WHERE username = ?",
            (username,)
        )
        if cur.fetchone() is None:
            # Insertion initiale
            db.execute(
                "INSERT INTO chat_sessions (username, messages, system_prompt, model, skills) VALUES (?, ?, ?, ?, ?)",
                (
                    username,
                    json.dumps(sess["messages"]),
                    sess["system_prompt"],
                    sess["model"],
                    json.dumps(sess["skills"])
                )
            )
        else:
            # Mise à jour
            db.execute(
                "UPDATE chat_sessions SET messages=?, system_prompt=?, model=?, skills=? WHERE username=?",
                (
                    json.dumps(sess["messages"]),
                    sess["system_prompt"],
                    sess["model"],
                    json.dumps(sess["skills"]),
                    username
                )
            )
        db.commit()


# ---------------------------------------------------------------------------
# Stores en mémoire (pour anonymes uniquement)
# ---------------------------------------------------------------------------
store: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _get_session():
    """Return the chat session for the current user (or anonymous)."""
    username = session.get("username", "__anon__")

    if username == "__anon__":
        # Pour les anonymes, on garde en mémoire comme avant
        if "__anon__" not in store:
            store["__anon__"] = {
                "messages": [],
                "system_prompt": "",
                "model": DEFAULT_MODEL,
                "skills": [],
            }
        return store["__anon__"]
    else:
        # Pour les utilisateurs connectés, on charge depuis la base
        with closing(get_db()) as db:
            cur = db.execute(
                "SELECT * FROM chat_sessions WHERE username = ?",
                (username,)
            )
            row = cur.fetchone()

            if row is None:
                # Créer une nouvelle session si elle n'existe pas
                session_data = {
                    "messages": [],
                    "system_prompt": "",
                    "model": DEFAULT_MODEL,
                    "skills": [],
                }
                db.execute(
                    "INSERT INTO chat_sessions (username, messages, system_prompt, model, skills) VALUES (?, ?, ?, ?, ?)",
                    (
                        username,
                        json.dumps(session_data["messages"]),
                        session_data["system_prompt"],
                        session_data["model"],
                        json.dumps(session_data["skills"])
                    )
                )
                db.commit()
                return session_data
            else:
                # Charger la session depuis la base
                try:
                    return {
                        "messages": json.loads(row["messages"]),
                        "system_prompt": row["system_prompt"],
                        "model": row["model"],
                        "skills": json.loads(row["skills"])
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur décodage JSON pour {username}: {e}")
                    # Retourner une session vide en cas d'erreur
                    return {
                        "messages": [],
                        "system_prompt": "",
                        "model": DEFAULT_MODEL,
                        "skills": [],
                    }


def _build_system_content(sess: dict) -> str:
    parts: list[str] = []
    if sess.get("skills"):
        parts.append(f"Skills and areas of expertise: {', '.join(sess['skills'])}.")
    if sess.get("system_prompt"):
        parts.append(sess["system_prompt"])
    return "\n\n".join(parts)


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "https://localhost",
    "X-Title": "NeuralChat",
    "Content-Type": "application/json",
}


# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------
def perform_search(query: str, max_results: int = 5) -> str:
    try:
        params: Dict[str, str] = {
            "q": urllib.parse.quote(query),
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        response = requests.get("https://api.duckduckgo.com/", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results: List[str] = []
        abstract = data.get("AbstractText", "").strip()
        abstract_src = data.get("AbstractSource", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()

        if abstract:
            results.append(
                f"Résultat instantané (source : {abstract_src or 'DuckDuckGo'})\n"
                f"{abstract}\n{abstract_url}\n"
            )
        else:
            results.append(f"Recherche Internet pour {query} :\n")

        related: List[Dict] = data.get("RelatedTopics", [])
        count = 0
        for item in related:
            if count >= max_results:
                break
            if isinstance(item, dict) and "FirstURL" in item:
                title = item.get("Text", "Titre inconnu").split(" – ")[0]
                snippet = re.sub(r"<[^>]+>", "", item.get("Text", ""))
                url = item.get("FirstURL", "#")
                results.append(f"{count + 1}. {title}\n{snippet[:200]}\n{url}\n")
                count += 1
            elif isinstance(item, dict) and "Topics" in item:
                for sub in item["Topics"]:
                    if count >= max_results:
                        break
                    title = sub.get("Text", "").split(" – ")[0]
                    snippet = re.sub(r"<[^>]+>", "", sub.get("Text", ""))
                    url = sub.get("FirstURL", "#")
                    results.append(f"{count + 1}. {title}\n{snippet[:200]}\n{url}\n")
                    count += 1

        if count == 0 and not abstract:
            results.append("Aucun résultat trouvé.")
        return "\n".join(results)
    except Exception as e:
        return f"Erreur de recherche : {str(e)}"


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/login")
def login_page():
    if session.get("username"):
        return redirect("/")
    return render_template_string(LOGIN_TEMPLATE)


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    username = data.get("username", "").strip().lower()
    password = data.get("password", "").strip()
    email = data.get("email", "").strip()

    if not username or not password:
        return jsonify({"error": "Nom d'utilisateur et mot de passe requis"}), 400
    if len(username) < 3:
        return jsonify({"error": "Le nom d'utilisateur doit faire au moins 3 caractères"}), 400
    if len(password) < 6:
        return jsonify({"error": "Le mot de passe doit faire au moins 6 caractères"}), 400

    with closing(get_db()) as db:
        cur = db.execute("SELECT * FROM users WHERE username = ?", (username,))
        if cur.fetchone() is not None:
            return jsonify({"error": "Ce nom d'utilisateur est déjà pris"}), 409

        db.execute(
            "INSERT INTO users (username, password_hash, email, created_at) VALUES (?, ?, ?, ?)",
            (
                username,
                generate_password_hash(password),
                email,
                time.strftime("%Y-%m-%d %H:%M:%S")
            )
        )
        db.commit()

    session["username"] = username
    logger.info("New user registered: %s", username)
    return jsonify({"ok": True, "username": username})


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username", "").strip().lower()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Identifiants manquants"}), 400

    with closing(get_db()) as db:
        cur = db.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()

        if not row or not check_password_hash(row["password_hash"], password):
            return jsonify({"error": "Identifiants incorrects"}), 401

    session["username"] = username
    logger.info("User logged in: %s", username)
    return jsonify({"ok": True, "username": username})


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/me", methods=["GET"])
def me():
    username = session.get("username")
    if not username:
        return jsonify({"logged_in": False})

    with closing(get_db()) as db:
        cur = db.execute(
            "SELECT username, email, created_at FROM users WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()

        if not row:
            session.clear()
            return jsonify({"logged_in": False})

        return jsonify({
            "logged_in": True,
            "username": row["username"],
            "email": row["email"] or "",
            "created_at": row["created_at"] or "",
        })


# ---------------------------------------------------------------------------
# Chat routes
# ---------------------------------------------------------------------------
@app.route("/")
def index() -> str:
    return render_template_string(HTML_TEMPLATE, models=FREE_MODELS, default_model=DEFAULT_MODEL)


@app.route("/api/context", methods=["POST"])
def set_context():
    data = request.get_json(force=True)
    username = session.get("username", "__anon__")
    sess = _get_session()

    if "system_prompt" in data:
        sess["system_prompt"] = data["system_prompt"].strip()
    if "model" in data:
        sess["model"] = data["model"]
    if "skills" in data:
        sess["skills"] = [s for s in data["skills"] if isinstance(s, str)]

    # Sauvegarder pour les utilisateurs connectés
    if username != "__anon__":
        save_session_to_db(username, sess)

    return jsonify({"ok": True})


@app.route("/api/skills", methods=["POST"])
def set_skills():
    data = request.get_json(force=True)
    username = session.get("username", "__anon__")
    sess = _get_session()
    sess["skills"] = [s for s in data.get("skills", []) if isinstance(s, str)]

    # Sauvegarder pour les utilisateurs connectés
    if username != "__anon__":
        save_session_to_db(username, sess)

    return jsonify({"ok": True, "skills": sess["skills"]})


@app.route("/api/chat", methods=["POST"])
def chat():
    username = session.get("username", "__anon__")
    sess = _get_session()

    if request.is_json:
        data = request.get_json()
        user_msg = data.get("message", "").strip()
    else:
        user_msg = request.form.get("message", "").strip()

    file_contents = []
    for f in request.files.getlist("files"):
        if f.filename:
            try:
                content = f.read().decode("utf-8", errors="replace")
                file_contents.append(f"--- Fichier joint: {f.filename} ---\n{content}")
            except Exception as e:
                logger.warning(f"Erreur lecture fichier: {e}")

    full_msg = user_msg
    if file_contents:
        full_msg += "\n\n" + "\n\n".join(file_contents)

    if not full_msg:
        return jsonify({"error": "Message vide"}), 400

    sys_content = _build_system_content(sess)
    tool_instruction = (
        "\n\nOUTILS DISPONIBLES:\n"
        "Tu as accès à un outil de recherche internet. Pour l'utiliser, "
        "tape exactement: \\recherche <ta question>\n"
        "Utilise cet outil pour les questions factuelles ou qui nécessitent des informations à jour."
    )
    full_sys_content = (sys_content + tool_instruction) if sys_content else tool_instruction.strip()

    sess["messages"].append({"role": "user", "content": full_msg})
    payload_messages = [{"role": "system", "content": full_sys_content}]
    payload_messages.extend(sess["messages"])

    max_search_attempts = 5
    search_count = 0
    final_reply = None

    while search_count < max_search_attempts:
        payload = {"model": sess["model"], "messages": payload_messages, "max_tokens": 65000}
        try:
            response = requests.post(OPENROUTER_URL, headers=_HEADERS, json=payload, timeout=90)
            result = response.json()
        except requests.RequestException as exc:
            logger.error(f"OpenRouter error: {exc}")
            sess["messages"].pop()
            return jsonify({"error": "Problème de connexion au service AI"}), 502

        if "error" in result:
            err = result["error"]
            sess["messages"].pop()
            return jsonify({"error": f"IA indisponible ({err.get('code', '?')}): {err.get('message', '')}"}), 502

        reply: str = result["choices"][0]["message"]["content"]

        if reply.strip().startswith('\\recherche '):
            search_count += 1
            query = reply.strip()[len('\\recherche '):].strip()
            if query:
                search_results = perform_search(query)
                payload_messages.append({"role": "assistant", "content": reply})
                payload_messages.append({
                    "role": "system",
                    "content": (
                        f"RÉSULTATS DE RECHERCHE pour '{query}':\n{search_results}\n\n"
                        "Rédige maintenant ta réponse finale basée sur ces résultats."
                    )
                })
            continue
        else:
            final_reply = reply
            payload_messages.append({"role": "assistant", "content": reply})
            break

    if final_reply is None:
        return jsonify({"reply": "Limite de recherches atteinte. Réessaie."})

    sess["messages"] = [m for m in payload_messages if m["role"] in ["user", "assistant"]]

    # Sauvegarder pour les utilisateurs connectés
    if username != "__anon__":
        save_session_to_db(username, sess)

    return jsonify({
        "reply": final_reply,
        "total_messages": len(sess["messages"]),
        "estimated_tokens": _estimate_tokens(sess["messages"]),
    })


@app.route("/api/clear", methods=["POST"])
def clear():
    username = session.get("username", "__anon__")
    sess = _get_session()
    sess["messages"] = []

    # Sauvegarder pour les utilisateurs connectés
    if username != "__anon__":
        save_session_to_db(username, sess)

    return jsonify({"ok": True})


@app.route("/api/history", methods=["GET"])
def history():
    sess = _get_session()
    return jsonify({
        "messages": sess["messages"],
        "system_prompt": sess["system_prompt"],
        "model": sess["model"],
        "skills": sess.get("skills", []),
        "estimated_tokens": _estimate_tokens(sess["messages"]),
    })


# ---------------------------------------------------------------------------
# Available free models on OpenRouter
# ---------------------------------------------------------------------------
FREE_MODELS: list[dict[str, str]] = [
    {"id": "minimax/minimax-m2.5:free", "label": "MiniMax M2.5"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B"},
    {"id": "stepfun/step-3.5-flash:free", "label": "Step 3.5"},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free", "label": "Nemotron 3"},
]

# ---------------------------------------------------------------------------
# LOGIN PAGE TEMPLATE
# ---------------------------------------------------------------------------
LOGIN_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NeuralChat — Connexion</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:      #0a0e1a;
  --panel:   #111827;
  --border:  #1e2d45;
  --accent:  #3b82f6;
  --accent2: #06b6d4;
  --accent3: #8b5cf6;
  --text:    #e2e8f0;
  --muted:   #64748b;
  --danger:  #ef4444;
  --success: #22c55e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'JetBrains Mono', monospace;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  display: flex; align-items: center; justify-content: center;
  padding: 20px; position: relative; overflow: hidden;
}
.bg-blob {
  position: absolute; border-radius: 50%;
  filter: blur(80px); opacity: 0.12;
  animation: drift 12s ease-in-out infinite alternate;
  pointer-events: none;
}
.bg-blob-1 { width: 500px; height: 500px; background: var(--accent);  top: -150px; left: -150px; animation-delay: 0s; }
.bg-blob-2 { width: 400px; height: 400px; background: var(--accent3); bottom: -100px; right: -100px; animation-delay: 3s; }
.bg-blob-3 { width: 300px; height: 300px; background: var(--accent2); top: 40%; left: 50%; animation-delay: 6s; }
@keyframes drift {
  from { transform: translate(0, 0) scale(1); }
  to   { transform: translate(30px, -30px) scale(1.08); }
}
.card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 20px; padding: 44px 40px;
  width: 100%; max-width: 420px;
  position: relative; z-index: 1;
  box-shadow: 0 24px 80px rgba(0,0,0,0.5);
}
.logo {
  font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; text-align: center; margin-bottom: 6px;
}
.logo-sub { text-align: center; font-size: 12px; color: var(--muted); margin-bottom: 36px; }
.tabs {
  display: flex; background: var(--bg);
  border-radius: 10px; padding: 4px; margin-bottom: 28px;
  border: 1px solid var(--border);
}
.tab-btn {
  flex: 1; padding: 9px; border: none; border-radius: 8px;
  background: transparent; color: var(--muted);
  font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 600;
  cursor: pointer; transition: all .2s;
}
.tab-btn.active { background: var(--accent); color: #fff; }
.form-group { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; }
.form-group label { font-size: 11px; color: var(--muted); letter-spacing: 0.5px; }
.form-group input {
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 10px; color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  padding: 13px 16px; outline: none; transition: border-color .2s, box-shadow .2s; width: 100%;
}
.form-group input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }
.form-group input::placeholder { color: var(--muted); }
.submit-btn {
  width: 100%; padding: 14px; border: none; border-radius: 10px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #fff; font-family: 'Syne', sans-serif;
  font-size: 14px; font-weight: 700; cursor: pointer;
  transition: opacity .2s, transform .15s; margin-top: 8px;
}
.submit-btn:hover:not(:disabled) { opacity: 0.9; transform: translateY(-1px); }
.submit-btn:disabled { opacity: .5; cursor: not-allowed; }
.error-msg {
  background: rgba(239,68,68,.1); border: 1px solid rgba(239,68,68,.3);
  color: var(--danger); border-radius: 8px; padding: 10px 14px;
  font-size: 12px; margin-top: 12px; display: none;
  animation: shake .3s ease;
}
@keyframes shake { 0%,100%{ transform: translateX(0); } 25%{ transform: translateX(-6px); } 75%{ transform: translateX(6px); } }
.error-msg.visible { display: block; }
.success-msg {
  background: rgba(34,197,94,.1); border: 1px solid rgba(34,197,94,.3);
  color: var(--success); border-radius: 8px; padding: 10px 14px;
  font-size: 12px; margin-top: 12px; display: none;
}
.success-msg.visible { display: block; }
.divider {
  text-align: center; color: var(--muted); font-size: 11px;
  margin: 20px 0 16px; position: relative;
}
.divider::before, .divider::after {
  content: ''; position: absolute; top: 50%; width: 40%;
  height: 1px; background: var(--border);
}
.divider::before { left: 0; }
.divider::after  { right: 0; }
.guest-btn {
  width: 100%; padding: 12px; border: 1px solid var(--border);
  border-radius: 10px; background: transparent; color: var(--muted);
  font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 600;
  cursor: pointer; transition: all .2s;
}
.guest-btn:hover { border-color: var(--accent3); color: var(--accent3); }
</style>
</head>
<body>
<div class="bg-blob bg-blob-1"></div>
<div class="bg-blob bg-blob-2"></div>
<div class="bg-blob bg-blob-3"></div>

<div class="card">
  <div class="logo">NeuralChat</div>
  <div class="logo-sub">Powered by OpenRouter</div>

  <div class="tabs">
    <button class="tab-btn active" id="tab-login"    onclick="switchTab('login')">Connexion</button>
    <button class="tab-btn"        id="tab-register" onclick="switchTab('register')">Créer un compte</button>
  </div>

  <!-- LOGIN -->
  <div id="form-login">
    <div class="form-group">
      <label>Nom d'utilisateur</label>
      <input type="text" id="login-username" placeholder="votre_pseudo" autocomplete="username"/>
    </div>
    <div class="form-group">
      <label>Mot de passe</label>
      <input type="password" id="login-password" placeholder="••••••••" autocomplete="current-password"/>
    </div>
    <button class="submit-btn" id="login-btn" onclick="doLogin()">Se connecter →</button>
    <div class="error-msg"   id="login-error"></div>
    <div class="success-msg" id="login-success"></div>
  </div>

  <!-- REGISTER -->
  <div id="form-register" style="display:none">
    <div class="form-group">
      <label>Nom d'utilisateur <span style="color:var(--danger)">*</span></label>
      <input type="text" id="reg-username" placeholder="votre_pseudo" autocomplete="username"/>
    </div>
    <div class="form-group">
      <label>Email <span style="color:var(--muted);font-size:10px;">(optionnel)</span></label>
      <input type="email" id="reg-email" placeholder="you@example.com" autocomplete="email"/>
    </div>
    <div class="form-group">
      <label>Mot de passe <span style="color:var(--danger)">*</span> <span style="color:var(--muted);font-size:10px;">(min. 6 caractères)</span></label>
      <input type="password" id="reg-password" placeholder="••••••••" autocomplete="new-password"/>
    </div>
    <div class="form-group">
      <label>Confirmer le mot de passe <span style="color:var(--danger)">*</span></label>
      <input type="password" id="reg-confirm" placeholder="••••••••" autocomplete="new-password"/>
    </div>
    <button class="submit-btn" id="register-btn" onclick="doRegister()">Créer mon compte →</button>
    <div class="error-msg"   id="register-error"></div>
    <div class="success-msg" id="register-success"></div>
  </div>

  <div class="divider">ou</div>
  <button class="guest-btn" onclick="window.location='/'">Continuer sans compte</button>
</div>

<script>
function switchTab(tab) {
  document.getElementById('form-login').style.display    = tab === 'login'    ? 'block' : 'none';
  document.getElementById('form-register').style.display = tab === 'register' ? 'block' : 'none';
  document.getElementById('tab-login').classList.toggle('active', tab === 'login');
  document.getElementById('tab-register').classList.toggle('active', tab === 'register');
  clearMessages();
}
function clearMessages() {
  ['login-error','login-success','register-error','register-success'].forEach(id => {
    const el = document.getElementById(id);
    el.className = el.className.replace(' visible', '');
  });
}
function showMsg(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.classList.add('visible');
}

async function doLogin() {
  const btn      = document.getElementById('login-btn');
  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value.trim();
  clearMessages();
  if (!username || !password) return showMsg('login-error', 'Veuillez remplir tous les champs');
  btn.disabled = true; btn.textContent = 'Connexion…';
  try {
    const res  = await fetch('/api/auth/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    if (data.ok) {
      showMsg('login-success', 'Bienvenue, ' + data.username + ' !');
      setTimeout(() => window.location = '/', 800);
    } else {
      showMsg('login-error', data.error || 'Erreur inconnue');
      btn.disabled = false; btn.textContent = 'Se connecter →';
    }
  } catch {
    showMsg('login-error', 'Erreur réseau');
    btn.disabled = false; btn.textContent = 'Se connecter →';
  }
}

async function doRegister() {
  const btn      = document.getElementById('register-btn');
  const username = document.getElementById('reg-username').value.trim();
  const email    = document.getElementById('reg-email').value.trim();
  const password = document.getElementById('reg-password').value.trim();
  const confirm  = document.getElementById('reg-confirm').value.trim();
  clearMessages();
  if (!username || !password || !confirm) return showMsg('register-error', 'Veuillez remplir les champs obligatoires');
  if (password !== confirm) return showMsg('register-error', 'Les mots de passe ne correspondent pas');
  if (password.length < 6) return showMsg('register-error', 'Le mot de passe doit faire au moins 6 caractères');
  btn.disabled = true; btn.textContent = 'Création…';
  try {
    const res  = await fetch('/api/auth/register', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, email })
    });
    const data = await res.json();
    if (data.ok) {
      showMsg('register-success', 'Compte créé ! Redirection…');
      setTimeout(() => window.location = '/', 900);
    } else {
      showMsg('register-error', data.error || 'Erreur inconnue');
      btn.disabled = false; btn.textContent = 'Créer mon compte →';
    }
  } catch {
    showMsg('register-error', 'Erreur réseau');
    btn.disabled = false; btn.textContent = 'Créer mon compte →';
  }
}

document.addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  if (document.getElementById('form-login').style.display !== 'none') doLogin();
  else doRegister();
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# MAIN CHAT TEMPLATE
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NeuralChat</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/tokyo-night-dark.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<style>
:root {
  --bg:      #0a0e1a;
  --panel:   #111827;
  --border:  #1e2d45;
  --accent:  #3b82f6;
  --accent2: #06b6d4;
  --accent3: #8b5cf6;
  --user-bg: #1d3461;
  --bot-bg:  #131f35;
  --text:    #e2e8f0;
  --muted:   #64748b;
  --danger:  #ef4444;
  --success: #22c55e;
  --warn:    #f59e0b;
  --radius:  12px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'JetBrains Mono', monospace; background: var(--bg); color: var(--text); height: 100vh; display: flex; overflow: hidden; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Sidebar ── */
#sidebar {
  width: 300px; min-width: 300px; background: var(--panel);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column; padding: 24px 20px; gap: 20px;
  overflow-y: auto; transition: transform 0.3s ease;
}
#sidebar h1 {
  font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; letter-spacing: -0.5px;
}
#sidebar h1 span { display: block; font-size: 11px; font-weight: 400; color: var(--muted); margin-top: 2px; -webkit-text-fill-color: var(--muted); }
.section-label { font-size: 10px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.field-group { display: flex; flex-direction: column; gap: 8px; }
label.field-label { font-size: 12px; color: var(--muted); }
select, textarea, input[type="text"] {
  background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
  color: var(--text); font-family: 'JetBrains Mono', monospace;
  font-size: 12px; padding: 10px 12px; width: 100%;
  transition: border-color 0.2s; outline: none; resize: none;
}
select:focus, textarea:focus, input:focus { border-color: var(--accent); }
#system-prompt { min-height: 110px; line-height: 1.6; }
.btn {
  display: flex; align-items: center; justify-content: center;
  gap: 8px; padding: 10px 16px; border-radius: 8px; border: none;
  cursor: pointer; font-family: 'Syne', sans-serif; font-weight: 600;
  font-size: 13px; transition: all 0.2s;
}
.btn-secondary { background: var(--accent3); color: #fff; }
.btn-secondary:hover { background: #7c3aed; }
.btn-danger { background: transparent; color: var(--danger); border: 1px solid var(--danger); }
.btn-danger:hover { background: rgba(239,68,68,.1); }
.btn-ghost { background: transparent; color: var(--muted); border: 1px solid var(--border); }
.btn-ghost:hover { border-color: var(--accent); color: var(--accent); }
.btn-full { width: 100%; }
#stats { font-size: 11px; color: var(--muted); padding: 12px; background: var(--bg); border-radius: 8px; border: 1px solid var(--border); line-height: 2; }
#stats span { color: var(--accent2); }
.skills-list { max-height: 150px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
.skill-tag { display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; font-size: 12px; animation: fadeIn .3s ease; }
@keyframes fadeIn { from { opacity:0; transform: translateX(-10px); } to { opacity:1; } }
.skill-tag .skill-name { color: var(--accent2); font-weight: 500; }
.skill-tag .skill-remove { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 14px; padding: 2px 6px; border-radius: 4px; transition: all .2s; }
.skill-tag .skill-remove:hover { color: var(--danger); background: rgba(239,68,68,.1); }

/* ── Skill modal ── */
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.75); display: flex; align-items: center; justify-content: center; z-index: 1000; opacity: 0; visibility: hidden; transition: all .3s; }
.modal-overlay.active { opacity: 1; visibility: visible; }
.modal { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 24px; width: 90%; max-width: 420px; transform: scale(.9); transition: transform .3s; }
.modal-overlay.active .modal { transform: scale(1); }
.modal h2 { font-family: 'Syne', sans-serif; font-size: 18px; margin-bottom: 16px; }
.modal-input { width: 100%; padding: 12px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 13px; margin-bottom: 16px; outline: none; }
.modal-input:focus { border-color: var(--accent3); }
.modal-buttons { display: flex; gap: 10px; justify-content: flex-end; }

/* ── User menu (account button in topbar) ── */
.user-menu-wrapper { position: relative; }
.user-avatar-btn {
  display: flex; align-items: center; gap: 8px;
  background: rgba(59,130,246,.1); border: 1px solid rgba(59,130,246,.3);
  border-radius: 10px; padding: 6px 12px; cursor: pointer; transition: all .2s;
  font-size: 12px; color: var(--accent); font-family: 'JetBrains Mono', monospace;
  outline: none;
}
.user-avatar-btn:hover { background: rgba(59,130,246,.2); }
.avatar-circle {
  width: 26px; height: 26px; border-radius: 8px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; color: #fff; font-family: 'Syne', sans-serif;
}
.user-dropdown {
  position: absolute; top: calc(100% + 8px); right: 0;
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 12px; padding: 8px; min-width: 210px;
  box-shadow: 0 12px 40px rgba(0,0,0,.5); z-index: 200;
  opacity: 0; visibility: hidden; transform: translateY(-6px); transition: all .2s;
}
.user-dropdown.open { opacity: 1; visibility: visible; transform: none; }
.user-dropdown-header { padding: 10px 12px 14px; border-bottom: 1px solid var(--border); margin-bottom: 8px; }
.user-dropdown-header .uname { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 14px; color: var(--text); }
.user-dropdown-header .uemail { font-size: 11px; color: var(--muted); margin-top: 2px; }
.dropdown-item {
  display: flex; align-items: center; gap: 10px; padding: 9px 12px;
  border-radius: 8px; cursor: pointer; font-size: 12px; color: var(--muted);
  transition: all .2s; border: none; background: none; width: 100%; text-align: left;
}
.dropdown-item:hover { background: rgba(255,255,255,.05); color: var(--text); }
.dropdown-item.danger { color: var(--danger); }
.dropdown-item.danger:hover { background: rgba(239,68,68,.1); }

/* Login button for guests */
.login-btn-topbar {
  display: flex; align-items: center; gap: 6px; padding: 7px 14px;
  border-radius: 10px; border: 1px solid var(--border); background: transparent;
  color: var(--muted); font-family: 'Syne', sans-serif;
  font-size: 12px; font-weight: 600; cursor: pointer; transition: all .2s;
}
.login-btn-topbar:hover { border-color: var(--accent); color: var(--accent); }

/* ── Layout ── */
#app-wrapper { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#topbar { padding: 12px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
#topbar-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
#model-badge { font-size: 11px; padding: 4px 10px; border-radius: 20px; background: rgba(59,130,246,.15); color: var(--accent); border: 1px solid rgba(59,130,246,.3); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }
#context-indicator { font-size: 11px; color: var(--success); display: none; align-items: center; gap: 5px; }
#context-indicator::before { content: '●'; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
#skills-indicator { font-size: 11px; color: var(--accent3); display: none; align-items: center; gap: 5px; }
#sidebar-toggle { display: none; background: none; border: none; cursor: pointer; color: var(--text); padding: 4px; font-size: 18px; }

/* ── Messages ── */
#messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; scroll-behavior: smooth; }
.msg-wrapper { display: flex; gap: 12px; max-width: 85%; animation: slideIn .25s ease; }
@keyframes slideIn { from { opacity:0; transform: translateY(8px); } to { opacity:1; } }
.msg-wrapper.user { flex-direction: row-reverse; align-self: flex-end; }
.msg-wrapper.bot  { align-self: flex-start; }
.avatar { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; margin-top: 2px; }
.avatar.user { background: var(--user-bg); }
.avatar.bot  { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
.bubble { padding: 14px 18px; border-radius: var(--radius); font-size: 13.5px; line-height: 1.7; max-width: 100%; }
.bubble.user { background: var(--user-bg); border-top-right-radius: 4px; color: #c7d9f8; }
.bubble.bot  { background: var(--bot-bg); border-top-left-radius: 4px; border: 1px solid var(--border); }
.bubble.bot p { margin-bottom: 10px; }
.bubble.bot p:last-child { margin-bottom: 0; }
.bubble.bot h1,.bubble.bot h2,.bubble.bot h3 { font-family: 'Syne', sans-serif; margin: 14px 0 8px; color: #fff; }
.bubble.bot ul,.bubble.bot ol { padding-left: 20px; margin: 8px 0; }
.bubble.bot li { margin-bottom: 4px; }
.bubble.bot code:not(pre code) { background: rgba(59,130,246,.15); color: var(--accent2); padding: 2px 6px; border-radius: 4px; font-size: 12px; }
.bubble.bot pre { background: #0d1117; border-radius: 8px; padding: 14px; overflow-x: auto; margin: 10px 0; border: 1px solid var(--border); }
.bubble.bot pre code { font-size: 12px; }
.bubble.bot blockquote { border-left: 3px solid var(--accent); padding-left: 12px; color: var(--muted); margin: 8px 0; }
.bubble.bot table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }
.bubble.bot th { background: rgba(59,130,246,.15); padding: 8px 12px; text-align: left; }
.bubble.bot td { padding: 7px 12px; border-bottom: 1px solid var(--border); }
.msg-time { font-size: 10px; color: var(--muted); margin-top: 5px; padding: 0 4px; }

/* ── Typing ── */
#typing { display: none; align-self: flex-start; gap: 12px; align-items: center; }
#typing.visible { display: flex; }
.typing-dots { background: var(--bot-bg); border: 1px solid var(--border); border-radius: var(--radius); border-top-left-radius: 4px; padding: 14px 18px; display: flex; gap: 5px; align-items: center; }
.typing-dots span { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); animation: bounce 1.2s infinite; }
.typing-dots span:nth-child(2) { animation-delay: .2s; }
.typing-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-7px)} }

/* ── Welcome ── */
#welcome { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: var(--muted); text-align: center; padding: 40px; }
#welcome h2 { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: var(--text); }
#welcome p  { font-size: 13px; line-height: 1.8; max-width: 400px; }
#welcome .hint { font-size: 11px; color: #334155; margin-top: 12px; }

/* ── Input area ── */
#input-area { padding: 16px 24px 20px; border-top: 1px solid var(--border); display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; flex-shrink: 0; }
#file-attach-area { width: 100%; display: none; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
#file-attach-area.visible { display: flex; }
.file-chip { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; font-size: 11px; display: flex; align-items: center; gap: 6px; color: var(--accent2); }
.file-chip .remove { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 14px; padding: 0 2px; transition: color .2s; }
.file-chip .remove:hover { color: var(--danger); }
#message-input { flex: 1; min-height: 50px; max-height: 160px; padding: 14px 16px; background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 13.5px; resize: none; outline: none; transition: border-color .2s; line-height: 1.5; }
#message-input:focus { border-color: var(--accent); }
#message-input::placeholder { color: var(--muted); }
.action-btn { width: 50px; height: 50px; border-radius: 12px; border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: transform .15s, opacity .15s; flex-shrink: 0; }
#attach-btn { background: var(--bg); color: var(--muted); }
#attach-btn:hover { color: var(--accent); transform: scale(1.05); }
#send-btn { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
#send-btn:hover:not(:disabled) { transform: scale(1.05); }
#send-btn:disabled { opacity: .4; cursor: not-allowed; }
.action-btn svg { width: 20px; height: 20px; fill: currentColor; }

/* ── Toast ── */
#toast { position: fixed; bottom: 30px; right: 30px; background: var(--panel); border: 1px solid var(--border); color: var(--text); padding: 12px 20px; border-radius: 10px; font-size: 13px; transform: translateY(80px); opacity: 0; transition: all .3s; z-index: 9999; pointer-events: none; }
#toast.show    { transform: none; opacity: 1; }
#toast.success { border-color: var(--success); color: var(--success); }
#toast.error   { border-color: var(--danger);  color: var(--danger); }
#toast.warn    { border-color: var(--warn);    color: var(--warn); }

@media (max-width: 768px) {
  #sidebar { position: absolute; z-index: 50; height: 100%; transform: translateX(-100%); }
  #sidebar.open { transform: none; }
  #sidebar-toggle { display: block; }
}
</style>
</head>
<body>

<!-- Skill modal -->
<div class="modal-overlay" id="skill-modal">
  <div class="modal">
    <h2>✨ Ajouter un Skill</h2>
    <input type="text" class="modal-input" id="skill-input" placeholder="Ex: Python, Rédaction, Data Science…"/>
    <div class="modal-buttons">
      <button class="btn btn-ghost" onclick="closeSkillModal()">Annuler</button>
      <button class="btn btn-secondary" onclick="addSkill()">Ajouter</button>
    </div>
  </div>
</div>

<aside id="sidebar">
  <div><h1>NeuralChat<span>Powered by OpenRouter</span></h1></div>

  <div>
    <div class="section-label">Modèle IA</div>
    <select id="model-select">
      {% for m in models %}
      <option value="{{ m.id }}" {% if m.id == default_model %}selected{% endif %}>{{ m.label }}</option>
      {% endfor %}
    </select>
  </div>

  <div>
    <div class="section-label">Contexte / Persona <span style="color:var(--success);font-size:10px;">(auto)</span></div>
    <div class="field-group">
      <label class="field-label">Comportement de l'IA</label>
      <textarea id="system-prompt" placeholder="Ex: Tu es un expert Python. Réponds avec des exemples concrets…" oninput="debouncedSaveContext()"></textarea>
      <small style="color:var(--muted);font-size:11px;margin-top:4px;">Sauvegardé automatiquement</small>
    </div>
  </div>

  <div>
    <div class="section-label">Skills & Expertise</div>
    <div class="field-group">
      <div class="skills-list" id="skills-list"></div>
      <button class="btn btn-secondary btn-full" onclick="openSkillModal()">➕ Ajouter un skill</button>
    </div>
  </div>

  <div>
    <div class="section-label">Session</div>
    <div id="stats">
      Messages : <span id="stat-msgs">0</span><br/>
      Tokens estimés : <span id="stat-tokens">0</span><br/>
      Contexte actif : <span id="stat-ctx">Non</span><br/>
      Skills actifs : <span id="stat-skills">0</span>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:8px;margin-top:auto;">
    <button class="btn btn-ghost btn-full" onclick="exportChat()">⬇ Exporter la conv.</button>
    <button class="btn btn-danger btn-full" onclick="clearChat()">✕ Effacer la mémoire</button>
  </div>
</aside>

<div id="app-wrapper">
  <div id="topbar">
    <button id="sidebar-toggle" onclick="document.getElementById('sidebar').classList.toggle('open')">☰</button>
    <span style="font-family:'Syne',sans-serif;font-weight:700;font-size:15px;color:var(--text);">💬 Chat</span>
    <div id="topbar-right">
      <span id="skills-indicator">✨ Skills actifs</span>
      <span id="context-indicator">Contexte actif</span>
      <span id="model-badge">{{ default_model }}</span>
      <!-- Auth zone injected by JS -->
      <div id="auth-zone"></div>
    </div>
  </div>

  <div id="messages">
    <div id="welcome">
      <h2>Bonjour 👋</h2>
      <p>Assistant IA avec mémoire de conversation. Configure un contexte dans le panneau gauche pour personnaliser mon comportement.</p>
      <div class="hint">Entrée pour envoyer · Shift+Entrée pour un saut de ligne</div>
    </div>
    <div id="typing">
      <div class="avatar bot">🤖</div>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>
  </div>

  <div id="input-area">
    <div id="file-attach-area"></div>
    <button id="attach-btn" class="action-btn" title="Attacher un fichier"
            onclick="document.getElementById('file-input').click()">
      <svg viewBox="0 0 24 24"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
    </button>
    <input type="file" id="file-input" multiple style="display:none" onchange="handleFileSelect(event)"/>
    <textarea id="message-input" placeholder="Envoie un message…" rows="1"></textarea>
    <button id="send-btn" class="action-btn" onclick="sendMessage()" title="Envoyer">
      <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
    </button>
  </div>
</div>

<div id="toast"></div>

<script>
/* ===================================================
   GLOBALS
   =================================================== */
let hasContext    = false;
let skills        = [];
let attachedFiles = [];
let contextSaveTimeout;
let currentUser   = null; // { username, email, created_at } | null

/* ===================================================
   AUTH ZONE
   =================================================== */
function renderAuthZone() {
  const zone = document.getElementById('auth-zone');
  if (currentUser) {
    const initial = currentUser.username.charAt(0).toUpperCase();
    zone.innerHTML = `
      <div class="user-menu-wrapper" id="user-menu-wrapper">
        <button class="user-avatar-btn" onclick="toggleUserMenu(event)">
          <div class="avatar-circle">${escHtml(initial)}</div>
          <span>${escHtml(currentUser.username)}</span>
          <span style="font-size:10px;color:var(--muted);margin-left:2px">▾</span>
        </button>
        <div class="user-dropdown" id="user-dropdown">
          <div class="user-dropdown-header">
            <div class="uname">${escHtml(currentUser.username)}</div>
            <div class="uemail">${escHtml(currentUser.email || 'Pas d\'email renseigné')}</div>
          </div>
          <button class="dropdown-item" onclick="exportChat()">⬇ Exporter la conversation</button>
          <button class="dropdown-item danger" onclick="doLogout()">↩ Se déconnecter</button>
        </div>
      </div>`;
  } else {
    zone.innerHTML = `
      <button class="login-btn-topbar" onclick="window.location='/login'">
        👤 Se connecter
      </button>`;
  }
}

function toggleUserMenu(e) {
  e.stopPropagation();
  document.getElementById('user-dropdown').classList.toggle('open');
}

document.addEventListener('click', () => {
  const dd = document.getElementById('user-dropdown');
  if (dd) dd.classList.remove('open');
});

async function doLogout() {
  await fetch('/api/auth/logout', { method: 'POST' });
  currentUser = null;
  renderAuthZone();
  showToast('Déconnecté', 'success');
}

async function loadUser() {
  try {
    const res  = await fetch('/api/auth/me');
    const data = await res.json();
    currentUser = data.logged_in
      ? { username: data.username, email: data.email, created_at: data.created_at }
      : null;
  } catch {
    currentUser = null;
  }
  renderAuthZone();
}

/* ===================================================
   TEXTAREA + ENTER
   =================================================== */
const msgInput = document.getElementById('message-input');
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 160) + 'px';
});
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

/* ===================================================
   CONTEXT AUTO-SAVE
   =================================================== */
function saveContext() {
  const prompt = document.getElementById('system-prompt').value.trim();
  const model  = document.getElementById('model-select').value;
  fetch('/api/context', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system_prompt: prompt, model })
  }).then(() => {
    hasContext = !!prompt;
    document.getElementById('context-indicator').style.display = hasContext ? 'flex' : 'none';
    document.getElementById('stat-ctx').textContent = hasContext ? 'Oui' : 'Non';
    document.getElementById('model-badge').textContent = model;
  }).catch(console.error);
}
function debouncedSaveContext() {
  clearTimeout(contextSaveTimeout);
  contextSaveTimeout = setTimeout(saveContext, 1000);
}
document.getElementById('model-select').addEventListener('change', saveContext);

/* ===================================================
   SKILLS
   =================================================== */
function openSkillModal() {
  document.getElementById('skill-modal').classList.add('active');
  document.getElementById('skill-input').value = '';
  setTimeout(() => document.getElementById('skill-input').focus(), 100);
}
function closeSkillModal() { document.getElementById('skill-modal').classList.remove('active'); }
async function addSkill() {
  const name = document.getElementById('skill-input').value.trim();
  if (!name) return showToast('Veuillez entrer un nom de skill', 'error');
  if (skills.includes(name)) return showToast('Ce skill existe déjà', 'error');
  skills.push(name); renderSkills(); closeSkillModal();
  await syncSkills(); showToast('Skill "' + name + '" ajouté', 'success');
}
async function removeSkill(name) {
  skills = skills.filter(s => s !== name); renderSkills();
  await syncSkills(); showToast('Skill supprimé', 'success');
}
async function syncSkills() {
  await fetch('/api/skills', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ skills }) });
}
function renderSkills() {
  const container = document.getElementById('skills-list');
  container.innerHTML = skills.map(s => `
    <div class="skill-tag">
      <span class="skill-name">✨ ${escHtml(s)}</span>
      <button class="skill-remove" onclick="removeSkill('${escHtml(s)}')" title="Supprimer">✕</button>
    </div>`).join('');
  const has = skills.length > 0;
  document.getElementById('skills-indicator').style.display = has ? 'flex' : 'none';
  document.getElementById('stat-skills').textContent = skills.length;
}

/* ===================================================
   FILE HANDLING
   =================================================== */
function handleFileSelect(event) {
  Array.from(event.target.files).forEach(f => {
    if (f.size > 5 * 1024 * 1024) return showToast(f.name + ' dépasse 5MB', 'error');
    if (attachedFiles.find(x => x.name === f.name)) return showToast(f.name + ' déjà ajouté', 'error');
    attachedFiles.push(f);
  });
  renderAttachedFiles(); event.target.value = '';
}
function renderAttachedFiles() {
  const c = document.getElementById('file-attach-area');
  c.innerHTML = attachedFiles.map(f => `
    <div class="file-chip">
      📄 ${escHtml(f.name)} <small>(${(f.size/1024).toFixed(1)}KB)</small>
      <button class="remove" onclick="removeAttached('${escHtml(f.name)}')">✕</button>
    </div>`).join('');
  c.className = attachedFiles.length ? 'visible' : '';
}
function removeAttached(name) {
  attachedFiles = attachedFiles.filter(f => f.name !== name);
  renderAttachedFiles();
}

/* ===================================================
   SEND MESSAGE
   =================================================== */
async function sendMessage() {
    const text = msgInput.value.trim();
    if (!text && attachedFiles.length === 0) return;
    msgInput.value = ''; msgInput.style.height = 'auto';
    document.getElementById('welcome')?.remove();

    let display = text || '[Fichier joint]';
    if (attachedFiles.length) display += '\n📎 ' + attachedFiles.map(f => f.name).join(', ');
    appendMessage('user', display);
    setTyping(true);

    try {
        const fd = new FormData();
        fd.append('message', text);
        attachedFiles.forEach(f => fd.append('files', f));
        attachedFiles = []; renderAttachedFiles();

        const res = await fetch('/api/chat', { method: 'POST', body: fd });
        const data = await res.json();
        setTyping(false);

        if (data.error) return showToast(data.error, 'error');

        // Gérer les fichiers ignorés
        if (data.ignored_files && data.ignored_files.length) {
            data.ignored_files.forEach(fileReason => {
                showToast(`Fichier ignoré: ${fileReason}`, 'warn');
            });
        }

        appendMessage('bot', data.reply);
        updateStats(data.total_messages, data.estimated_tokens);
    } catch {
        setTyping(false);
        showToast('Erreur réseau', 'error');
    }
}

/* ===================================================
   RENDER MESSAGE
   =================================================== */
function appendMessage(role, text) {
  const container = document.getElementById('messages');
  const typing    = document.getElementById('typing');
  const wrapper   = document.createElement('div');
  wrapper.className = `msg-wrapper ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === 'user'
    ? (currentUser ? currentUser.username.charAt(0).toUpperCase() : '🧑')
    : '🤖';

  const inner  = document.createElement('div');
  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  bubble.innerHTML = role === 'bot' ? marked.parse(text) : escHtml(text);
  if (role === 'bot') bubble.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));

  const time = document.createElement('div');
  time.className   = 'msg-time';
  time.textContent = new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  if (role === 'user') time.style.textAlign = 'right';

  inner.append(bubble, time);
  if (role === 'user') wrapper.append(inner, avatar);
  else                 wrapper.append(avatar, inner);

  container.insertBefore(wrapper, typing);
  container.scrollTop = container.scrollHeight;
}

function setTyping(on) {
  document.getElementById('typing').className = on ? 'visible' : '';
  document.getElementById('send-btn').disabled = on;
  if (on) document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
}
function updateStats(msgs, tokens) {
  document.getElementById('stat-msgs').textContent   = msgs   ?? '–';
  document.getElementById('stat-tokens').textContent = tokens ?? '–';
}

/* ===================================================
   CLEAR & EXPORT
   =================================================== */
async function clearChat() {
  if (!confirm('Effacer toute la mémoire de conversation ?')) return;
  await fetch('/api/clear', { method: 'POST' });
  document.getElementById('messages').innerHTML = `
    <div id="welcome">
      <h2>Conversation effacée ✓</h2><p>La mémoire a été réinitialisée.</p>
    </div>
    <div id="typing">
      <div class="avatar bot">🤖</div>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;
  updateStats(0, 0);
  showToast('Mémoire effacée', 'success');
}

async function exportChat() {
  const res  = await fetch('/api/history');
  const data = await res.json();
  if (!data.messages?.length) return showToast('Aucun message à exporter', 'error');

  let md = '# Conversation — ' + new Date().toLocaleString('fr-FR') + '\n\n';
  if (currentUser) md += '> **Utilisateur :** ' + currentUser.username + '\n\n';
  if (data.system_prompt) md += '> **Contexte :** ' + data.system_prompt + '\n\n';
  if (data.skills?.length) md += '> **Skills :** ' + data.skills.join(', ') + '\n\n---\n\n';
  data.messages.forEach(m => {
    md += (m.role === 'user' ? '**Vous**' : '**IA**') + '\n\n' + m.content + '\n\n---\n\n';
  });
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([md], { type: 'text/markdown' })),
    download: 'conversation_' + Date.now() + '.md'
  });
  a.click();
  showToast('Exporté', 'success');
}

/* ===================================================
   UTILS
   =================================================== */
function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/'/g,"&#39;").replace(/"/g,'&quot;');
}
function showToast(msg, type = '') {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'show ' + type;
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.className = '', 3000);
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeSkillModal();
});

/* ===================================================
   INIT
   =================================================== */
(async () => {
  await loadUser();
  try {
    const res  = await fetch('/api/history');
    const data = await res.json();
    if (data.messages?.length) {
      document.getElementById('welcome')?.remove();
      data.messages.forEach(m => appendMessage(m.role, m.content));
      updateStats(data.messages.length, data.estimated_tokens);
    }
    if (data.system_prompt) {
      document.getElementById('system-prompt').value = data.system_prompt;
      hasContext = true;
      document.getElementById('context-indicator').style.display = 'flex';
      document.getElementById('stat-ctx').textContent = 'Oui';
    }
    if (data.skills?.length) { skills = data.skills; renderSkills(); }
    if (data.model) {
      document.getElementById('model-select').value      = data.model;
      document.getElementById('model-badge').textContent = data.model;
    }
  } catch (e) { console.error('Init error:', e); }
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    logger.info("NeuralChat with auth — starting")

    # Initialiser la base de données
    init_db()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
