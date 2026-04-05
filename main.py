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
from flask import Flask, jsonify, render_template_string, request, session
import threading
import urllib.parse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "minimax/minimax-m2.5:free"

KEEP_ALIVE_URL = "https://neuralclaw.onrender.com"
KEEP_ALIVE_INTERVAL = 30  # secondes

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keep-Alive Background Thread
# ---------------------------------------------------------------------------
def keep_alive_pinger():
    """Ping le site toutes les 30 secondes pour éviter l'idle timeout."""
    logger.info(f"🚀 Thread Keep-Alive démarré (ping toutes les {KEEP_ALIVE_INTERVAL}s)")
    while True:
        try:
            response = requests.get(KEEP_ALIVE_URL, timeout=10)
            logger.info(f"✅ Keep-Alive ping: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Keep-Alive échoué: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur Keep-Alive: {e}")
        
        time.sleep(KEEP_ALIVE_INTERVAL)


def start_keep_alive_thread():
    """Démarre le thread de ping en arrière-plan."""
    ping_thread = threading.Thread(target=keep_alive_pinger, daemon=True)
    ping_thread.start()
    logger.info("✅ Thread Keep-Alive lancé")

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 50 * 1024 * 1024
app.secret_key = "change-me-in-production-use-os-urandom"

store: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Available free models on OpenRouter
# ---------------------------------------------------------------------------
FREE_MODELS: list[dict[str, str]] = [
    {"id": "minimax/minimax-m2.5:free",                    "label": "MiniMax M2.5"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free",       "label": "Llama 3.3 70B"},
    {"id": "stepfun/step-3.5-flash:free",                  "label": "Step 3.5"},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free",       "label": "Nemotron 3"},
]

# ---------------------------------------------------------------------------
# Helpers — session
# ---------------------------------------------------------------------------

def _get_session() -> dict[str, Any]:
    sid = session.get("id")
    if not sid or sid not in store:
        sid = str(uuid.uuid4())
        session["id"] = sid
        store[sid] = {
            "messages": [],
            "system_prompt": "",
            "model": DEFAULT_MODEL,
            "skills": [],
        }
    return store[sid]


def _build_system_content(sess: dict) -> str:
    parts: list[str] = []
    if sess.get("skills"):
        parts.append(f"Skillsand areas of expertise: {', '.join(sess['skills'])}.")
    if sess.get("system_prompt"):
        parts.append(sess["system_prompt"])
    return "\n\n".join(parts)


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


# ---------------------------------------------------------------------------
# Helpers — LLM calls
# ---------------------------------------------------------------------------

_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "https://localhost",
    "X-Title": "AI Chatbot NeuralChat",
    "Content-Type": "application/json",
}


def _call_llm(messages: list[dict], model: str) -> str:
    payload = {"model": model, "messages": messages, "max_tokens": 65000}
    for attempt in range(3):
        try:
            resp = requests.post(OPENROUTER_URL, headers=_HEADERS, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if content:
                    return content.strip()
        except requests.exceptions.RequestException as exc:
            logger.error("LLM request error (attempt %d): %s", attempt + 1, exc)
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))
    return "ℹ️ Le modèle n'a généré aucun texte. Réessaie."


# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------

def perform_search(query: str, max_results: int = 5) -> str:
    try:
        encoded_query = urllib.parse.quote(query)
        ddg_url = "https://api.duckduckgo.com/"
        params: Dict[str, str] = {
            "q": encoded_query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        response = requests.get(ddg_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results: List[str] = []
        abstract = data.get("AbstractText", "").strip()
        abstract_src = data.get("AbstractSource", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()

        if abstract:
            results.append(
                f"🔎 **Résultat instantané** (source : {abstract_src or 'DuckDuckGo'})\n"
                f"{abstract}\n"
                f"🔗 {abstract_url}\n"
            )
        else:
            results.append(f"🔍 Recherche Internet pour « {query} » :\n")

        related: List[Dict] = data.get("RelatedTopics", [])
        count = 0

        for item in related:
            if count >= max_results:
                break
            if isinstance(item, dict) and "FirstURL" in item:
                title = item.get("Text", "Titre inconnu").split(" – ")[0]
                snippet = re.sub(r"<[^>]+>", "", item.get("Text", ""))
                url = item.get("FirstURL", "#")
                results.append(
                    f"{count + 1}. **{title}**\n"
                    f"{snippet[:200]}…\n"
                    f"🔗 {url}\n"
                )
                count += 1
            elif isinstance(item, dict) and "Topics" in item:
                for sub in item["Topics"]:
                    if count >= max_results:
                        break
                    title = sub.get("Text", "Titre inconnu").split(" – ")[0]
                    snippet = re.sub(r"<[^>]+>", "", sub.get("Text", ""))
                    url = sub.get("FirstURL", "#")
                    results.append(
                        f"{count + 1}. **{title}**\n"
                        f"{snippet[:200]}…\n"
                        f"🔗 {url}\n"
                    )
                    count += 1

        if count == 0 and not abstract:
            results.append("❌ Aucun résultat trouvé pour cette requête.")

        return "\n".join(results)

    except requests.RequestException as e:
        return f"❌ Erreur de connexion : {str(e)}"
    except Exception as e:
        return f"❌ Erreur inattendue : {str(e)}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    return render_template_string(HTML_TEMPLATE, models=FREE_MODELS, default_model=DEFAULT_MODEL)


@app.route("/api/context", methods=["POST"])
def set_context():
    data = request.get_json(force=True)
    sess = _get_session()
    if "system_prompt" in data:
        sess["system_prompt"] = data["system_prompt"].strip()
    if "model" in data:
        sess["model"] = data["model"]
    if "skills" in data:
        sess["skills"] = [s for s in data["skills"] if isinstance(s, str)]
    return jsonify({"ok": True})


@app.route("/api/skills", methods=["POST"])
def set_skills():
    data = request.get_json(force=True)
    sess = _get_session()
    sess["skills"] = [s for s in data.get("skills", []) if isinstance(s, str)]
    return jsonify({"ok": True, "skills": sess["skills"]})


@app.route("/api/chat", methods=["POST"])
def chat():
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
        "Exemple: \\recherche comment créer une classe en Python\n"
        "Le système remplacera cette commande par les résultats de recherche "
        "pour que tu puisses rédiger une réponse complète et précise.\n"
        "IMPORTANT: Utilise cet outil pour les questions factuelles, techniques, "
        "ou qui nécessitent des informations à jour."
    )
    full_sys_content = (sys_content + tool_instruction) if sys_content else tool_instruction.strip()

    sess["messages"].append({"role": "user", "content": full_msg})

    payload_messages = [{"role": "system", "content": full_sys_content}]
    payload_messages.extend(sess["messages"])

    # Search loop
    max_search_attempts = 5
    search_count = 0
    final_reply = None

    while search_count < max_search_attempts:
        payload = {"model": sess["model"], "messages": payload_messages, "max_tokens": 65000}
        try:
            response = requests.post(OPENROUTER_URL, headers=_HEADERS, json=payload, timeout=90)
            result = response.json()
        except requests.RequestException as exc:
            logger.error(f"❌ Erreur de connexion OpenRouter: {exc}")
            sess["messages"].pop()
            return jsonify({"error": "Problème de connexion au service AI"}), 502

        if "error" in result:
            err_data = result["error"]
            detailed_msg = err_data.get("message", "Erreur inconnue")
            err_code = err_data.get("code", "N/A")
            logger.error(f"OpenRouter Error [{err_code}]: {detailed_msg}")
            sess["messages"].pop()
            return jsonify({"error": f"IA indisponible ({err_code}): {detailed_msg}"}), 502

        reply: str = result["choices"][0]["message"]["content"]

        if reply.strip().startswith('\\recherche '):
            search_count += 1
            query = reply.strip()[len('\\recherche '):].strip()
            if not query:
                payload_messages.append({"role": "assistant", "content": reply})
                continue

            logger.info(f"🔍 Recherche demandée: {query}")
            search_results = perform_search(query)
            payload_messages.append({"role": "assistant", "content": reply})
            payload_messages.append({
                "role": "system",
                "content": (
                    f"RÉSULTATS DE RECHERCHE pour '{query}':\n{search_results}\n\n"
                    "Tu peux maintenant rédiger ta réponse finale basée sur ces résultats."
                )
            })
            continue
        else:
            final_reply = reply
            payload_messages.append({"role": "assistant", "content": reply})
            break

    if final_reply is None:
        error_msg = (
            "⚠️ Limite de recherches atteinte. L'IA n'a pas fourni de réponse finale."
            if search_count >= max_search_attempts
            else "❌ Erreur lors de la génération de la réponse."
        )
        return jsonify({"reply": error_msg})

    sess["messages"] = [
        msg for msg in payload_messages
        if msg["role"] in ["user", "assistant"]
    ]

    return jsonify({
        "reply": final_reply,
        "total_messages": len(sess["messages"]),
        "estimated_tokens": _estimate_tokens(sess["messages"]),
    })


@app.route("/api/clear", methods=["POST"])
def clear():
    sess = _get_session()
    sess["messages"] = []
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
# HTML Template — Chat only
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta name="google-site-verification" content="vzCrSu00ltws5LqYzv30DlzzN4GfanP6A_gGuf6TrS0" />
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

body {
  font-family: 'JetBrains Mono', monospace;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  overflow: hidden;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Sidebar ── */
#sidebar {
  width: 300px;
  min-width: 300px;
  background: var(--panel);
  border-right: 1px solidvar(--border);
  display: flex;
  flex-direction: column;
  padding: 24px 20px;
  gap: 20px;
  overflow-y: auto;
  transition: transform 0.3s ease;
}

#sidebar h1 {
  font-family: 'Syne', sans-serif;
  font-size: 22px;
  font-weight: 800;
  background: linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.5px;
}
#sidebar h1 span {
  display: block;
  font-size: 11px;
  font-weight: 400;
  color: var(--muted);
  margin-top: 2px;
  -webkit-text-fill-color: var(--muted);
}

.section-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}

.field-group { display: flex; flex-direction: column; gap: 8px; }
label.field-label { font-size: 12px; color: var(--muted); }

select, textarea, input[type="text"] {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  padding: 10px 12px;
  width: 100%;
  transition: border-color 0.2s;
  outline: none;
  resize: none;
}
select:focus, textarea:focus, input:focus { border-color: var(--accent); }
#system-prompt { min-height: 110px; line-height: 1.6; }

.btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 16px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  font-family: 'Syne', sans-serif;
  font-weight: 600;
  font-size: 13px;
  transition: all 0.2s;
}
.btn-secondary { background: var(--accent3); color: #fff; }
.btn-secondary:hover { background: #7c3aed; }
.btn-danger    { background: transparent; color: var(--danger); border: 1px solid var(--danger); }
.btn-danger:hover    { background: rgba(239,68,68,.1); }
.btn-ghost     { background: transparent; color: var(--muted); border: 1px solid var(--border); }
.btn-ghost:hover     { border-color: var(--accent); color: var(--accent); }
.btn-full  { width: 100%; }

#stats {
  font-size: 11px;
  color: var(--muted);
  padding: 12px;
  background: var(--bg);
  border-radius: 8px;
  border: 1px solid var(--border);
  line-height: 2;
}
#stats span { color: var(--accent2); }

.skills-list {
  max-height: 150px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.skill-tag {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 12px;
  animation: fadeIn .3s ease;
}
@keyframes fadeIn { from { opacity:0; transform: translateX(-10px); } to { opacity:1; } }
.skill-tag .skill-name { color: var(--accent2); font-weight: 500; }
.skill-tag .skill-remove {
  background: none; border: none; color: var(--muted);
  cursor: pointer; font-size: 14px; padding: 2px 6px;
  border-radius: 4px; transition: all .2s;
}
.skill-tag .skill-remove:hover { color: var(--danger); background: rgba(239,68,68,.1); }

/* ── Skill modal ── */
.modal-overlay {
  position: fixed; inset: 0;
  background: rgba(0,0,0,.75);
  display: flex; align-items: center; justify-content: center;
  z-index: 1000;
  opacity: 0; visibility: hidden;
  transition: all .3s;
}
.modal-overlay.active { opacity: 1; visibility: visible; }
.modal {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  width: 90%;
  max-width: 420px;
  transform: scale(.9);
  transition: transform .3s;
}
.modal-overlay.active .modal { transform: scale(1); }
.modal h2 {
  font-family: 'Syne', sans-serif;
  font-size: 18px; margin-bottom: 16px;
}
.modal-input {
  width: 100%; padding: 12px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px; margin-bottom: 16px; outline: none;
}
.modal-input:focus { border-color: var(--accent3); }
.modal-buttons { display: flex; gap: 10px; justify-content: flex-end; }

/* ── Main layout ── */
#app-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

#topbar {
  padding: 14px 24px;
  border-bottom: 1px solidvar(--border);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

#topbar-right {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 10px;
}
#model-badge {
  font-size: 11px; padding: 4px 10px;
  border-radius: 20px;
  background: rgba(59,130,246,.15);
  color: var(--accent);
  border: 1px solid rgba(59,130,246,.3);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 200px;
}
#context-indicator {
  font-size: 11px; color: var(--success);
  display: none; align-items: center; gap: 5px;
}
#context-indicator::before { content: '●'; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

#skills-indicator {
  font-size: 11px; color: var(--accent3);
  display: none; align-items: center; gap: 5px;
}

#sidebar-toggle {
  display: none; background: none; border: none;
  cursor: pointer; color: var(--text); padding: 4px;
  font-size: 18px;
}

/* ── Messages ── */
#messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  scroll-behavior: smooth;
}

.msg-wrapper { display: flex; gap: 12px; max-width: 85%; animation: slideIn .25s ease; }
@keyframes slideIn { from { opacity:0; transform: translateY(8px); } to { opacity:1; } }
.msg-wrapper.user { flex-direction: row-reverse; align-self: flex-end; }
.msg-wrapper.bot  { align-self: flex-start; }

.avatar {
  width: 34px; height: 34px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0; margin-top: 2px;
}
.avatar.user { background: var(--user-bg); }
.avatar.bot  { background: linear-gradient(135deg,var(--accent),var(--accent2)); }

.bubble {
  padding: 14px 18px; border-radius: var(--radius);
  font-size: 13.5px; line-height: 1.7; max-width: 100%;
}
.bubble.user {
  background: var(--user-bg);
  border-top-right-radius: 4px;
  color: #c7d9f8;
}
.bubble.bot {
  background: var(--bot-bg);
  border-top-left-radius: 4px;
  border: 1px solidvar(--border);
}
.bubble.bot p { margin-bottom: 10px; }
.bubble.bot p:last-child { margin-bottom: 0; }
.bubble.bot h1,.bubble.bot h2,.bubble.bot h3 {
  font-family: 'Syne', sans-serif; margin: 14px 0 8px; color: #fff;
}
.bubble.bot ul,.bubble.bot ol { padding-left: 20px; margin: 8px 0; }
.bubble.bot li { margin-bottom: 4px; }
.bubble.bot code:not(pre code) {
  background: rgba(59,130,246,.15); color: var(--accent2);
  padding: 2px 6px; border-radius: 4px; font-size: 12px;
}
.bubble.bot pre {
  background: #0d1117; border-radius: 8px; padding: 14px;
  overflow-x: auto; margin: 10px 0; border: 1px solidvar(--border);
}
.bubble.bot pre code { font-size: 12px; }
.bubble.bot blockquote {
  border-left: 3px solid var(--accent);
  padding-left: 12px; color: var(--muted); margin: 8px 0;
}
.bubble.bot table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }
.bubble.bot th { background: rgba(59,130,246,.15); padding: 8px 12px; text-align: left; }
.bubble.bot td { padding: 7px 12px; border-bottom: 1px solidvar(--border); }

.msg-time { font-size: 10px; color: var(--muted); margin-top: 5px; padding: 0 4px; }

/* ── Typing indicator ── */
#typing { display: none; align-self: flex-start; gap: 12px; align-items: center; }
#typing.visible { display: flex; }
.typing-dots {
  background: var(--bot-bg); border: 1px solidvar(--border);
  border-radius: var(--radius); border-top-left-radius: 4px;
  padding: 14px 18px; display: flex; gap: 5px; align-items: center;
}
.typing-dots span {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent); animation: bounce 1.2s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: .2s; }
.typing-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-7px)} }

/* ── Welcome screen ── */
#welcome {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 12px; color: var(--muted); text-align: center; padding: 40px;
}
#welcome h2 { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: var(--text); }
#welcome p  { font-size: 13px; line-height: 1.8; max-width: 400px; }
#welcome .hint { font-size: 11px; color: #334155; margin-top: 12px; }

/* ── Input area ── */
#input-area {
  padding: 16px 24px 20px;
  border-top: 1px solidvar(--border);
  display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap;
  flex-shrink: 0;
}
#file-attach-area {
  width: 100%; display: none; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;
}
#file-attach-area.visible { display: flex; }
.file-chip {
  background: var(--bg); border: 1px solidvar(--border);
  border-radius: 8px; padding: 6px 10px; font-size: 11px;
  display: flex; align-items: center; gap: 6px; color: var(--accent2);
}
.file-chip .remove {
  background: none; border: none; color: var(--muted);
  cursor: pointer; font-size: 14px; padding: 0 2px; transition: color .2s;
}
.file-chip .remove:hover { color: var(--danger); }

#message-input {
  flex: 1; min-height: 50px; max-height: 160px;
  padding: 14px 16px;
  background: var(--panel); border: 1px solidvar(--border);
  border-radius: var(--radius); color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: 13.5px;
  resize: none; outline: none; transition: border-color .2s; line-height: 1.5;
}
#message-input:focus { border-color: var(--accent); }
#message-input::placeholder { color: var(--muted); }

.action-btn {
  width: 50px; height: 50px; border-radius: 12px; border: none;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  transition: transform .15s, opacity .15s; flex-shrink: 0;
}
#attach-btn { background: var(--bg); color: var(--muted); }
#attach-btn:hover { color: var(--accent); transform: scale(1.05); }
#send-btn { background: linear-gradient(135deg,var(--accent),var(--accent2)); }
#send-btn:hover:not(:disabled) { transform: scale(1.05); }
#send-btn:disabled { opacity: .4; cursor: not-allowed; }
.action-btn svg { width: 20px; height: 20px; fill: currentColor; }

/* ── Toast ── */
#toast {
  position: fixed; bottom: 30px; right: 30px;
  background: var(--panel); border: 1px solidvar(--border);
  color: var(--text); padding: 12px 20px; border-radius: 10px;
  font-size: 13px; transform: translateY(80px); opacity: 0;
  transition: all .3s; z-index: 9999; pointer-events: none;
}
#toast.show    { transform: none; opacity: 1; }
#toast.success { border-color: var(--success); color: var(--success); }
#toast.error   { border-color: var(--danger);  color: var(--danger); }
#toast.warn    { border-color: var(--warn);    color: var(--warn); }

/* ── Responsive ── */
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
    <input type="text" class="modal-input" id="skill-input"
           placeholder="Ex: Python, Rédaction, Analyse de données…"/>
    <div class="modal-buttons">
      <button class="btn btn-ghost" onclick="closeSkillModal()">Annuler</button>
      <button class="btn btn-secondary" onclick="addSkill()">Ajouter</button>
    </div>
  </div>
</div>

<aside id="sidebar">
  <div>
    <h1>NeuralChat<span>Powered by OpenRouter</span></h1>
  </div>

  <div>
    <div class="section-label">Modèle IA</div>
    <select id="model-select">
      {%for m in models%}
      <option value="{{ m.id }}" {%if m.id == default_model%}selected{%endif%}>{{ m.label }}</option>
      {%endfor%}
    </select>
  </div>

  <div>
    <div class="section-label">Contexte / Persona <span style="color:var(--success); font-size:10px;">(auto)</span></div>
    <div class="field-group">
      <label class="field-label">Comportement de l'IA</label>
      <textarea id="system-prompt"
        placeholder="Ex: Tu es un expert Python. Réponds avec des exemples concrets…"
        oninput="debouncedSaveContext()"></textarea>
      <small style="color:var(--muted); font-size:11px; margin-top:4px;">Sauvegardé automatiquement</small>
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

  <div style="display:flex; flex-direction:column; gap:8px; margin-top:auto;">
    <button class="btn btn-ghost btn-full" onclick="exportChat()">⬇ Exporter la conv.</button>
    <button class="btn btn-danger btn-full" onclick="clearChat()">✕ Effacer la mémoire</button>
  </div>
</aside>

<div id="app-wrapper">

  <div id="topbar">
    <button id="sidebar-toggle" onclick="document.getElementById('sidebar').classList.toggle('open')">☰</button>
    <span style="font-family:'Syne',sans-serif; font-weight:700; font-size:15px; color:var(--text);">💬 Chat</span>
    <div id="topbar-right">
      <span id="skills-indicator">✨ Skills actifs</span>
      <span id="context-indicator">Contexte actif</span>
      <span id="model-badge">{{ default_model }}</span>
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
/* =========================================================
   GLOBALS
   ========================================================= */
let hasContext    = false;
let skills        = [];
let attachedFiles = [];
let contextSaveTimeout;

/* =========================================================
   TEXTAREA AUTO-RESIZE + ENTER KEY
   ========================================================= */
const msgInput = document.getElementById('message-input');
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 160) + 'px';
});
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

/* =========================================================
   CONTEXT AUTO-SAVE
   ========================================================= */
function saveContext() {
  const prompt = document.getElementById('system-prompt').value.trim();
  const model  = document.getElementById('model-select').value;
  fetch('/api/context', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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

/* =========================================================
   SKILLS
   ========================================================= */
function openSkillModal() {
  document.getElementById('skill-modal').classList.add('active');
  document.getElementById('skill-input').value = '';
  setTimeout(() => document.getElementById('skill-input').focus(), 100);
}
function closeSkillModal() {
  document.getElementById('skill-modal').classList.remove('active');
}

async function addSkill() {
  const name = document.getElementById('skill-input').value.trim();
  if (!name) return showToast('❌ Veuillez entrer un nom de skill', 'error');
  if (skills.includes(name)) return showToast('❌ Ce skill existe déjà', 'error');
  skills.push(name);
  renderSkills();
  closeSkillModal();
  await syncSkills();
  showToast(`✓ Skill "${name}" ajouté`, 'success');
}

async function removeSkill(name) {
  skills = skills.filter(s => s !== name);
  renderSkills();
  await syncSkills();
  showToast(`✕ Skill "${name}" supprimé`, 'success');
}

async function syncSkills() {
  await fetch('/api/skills', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ skills })
  });
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

/* =========================================================
   FILE HANDLING
   ========================================================= */
function handleFileSelectevent() {
  Array.from(event.target.files.forEach(f => {
    if (f.size > 5 * 1024 * 1024) return showToast(`❌ ${f.name} dépasse 5MB`, 'error');
    if (attachedFiles.find(x => x.name === f.name)) return showToast(`❌ ${f.name} déjà ajouté`, 'error');
    attachedFiles.push(f);
  });
  renderAttachedFiles();
  event.target.value = '';
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

/* =========================================================
   SEND MESSAGE
   ========================================================= */
async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text && attachedFiles.length === 0) return;

  msgInput.value = '';
  msgInput.style.height = 'auto';
  document.getElementById('welcome')?.remove();

  let display = text || '[Fichier joint sans texte]';
  if (attachedFiles.length) display += `\n📎 ${attachedFiles.map(f => f.name).join(', ')}`;
  appendMessage('user', display);
  setTyping(true);

  try {
    const fd = new FormData();
    fd.append('message', text);
    attachedFiles.forEach(f => fd.append('files', f));
    attachedFiles = [];
    renderAttachedFiles();

    const res  = await fetch('/api/chat', { method: 'POST', body: fd });
    const data = await res.json();
    setTyping(false);
    if (data.error) return showToast('❌ ' + data.error, 'error');
    appendMessage('bot', data.reply);
    updateStats(data.total_messages, data.estimated_tokens);
  } catch {
    setTyping(false);
    showToast('❌ Erreur réseau', 'error');
  }
}

/* =========================================================
   RENDER MESSAGE
   ========================================================= */
function appendMessage(role, text) {
  const container = document.getElementById('messages');
  const typing    = document.getElementById('typing');

  const wrapper = document.createElement('div');
  wrapper.className = `msg-wrapper ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === 'user' ? '🧑' : '🤖';

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
  if (on) {
    const c = document.getElementById('messages');
    c.scrollTop = c.scrollHeight;
  }
}

function updateStats(msgs, tokens) {
  document.getElementById('stat-msgs').textContent   = msgs   ?? '–';
  document.getElementById('stat-tokens').textContent = tokens ?? '–';
}

/* =========================================================
   CLEAR & EXPORT
   ========================================================= */
async function clearChat() {
  if (!confirm('Effacer toute la mémoire de conversation ?')) return;
  await fetch('/api/clear', { method: 'POST' });
  const m = document.getElementById('messages');
  m.innerHTML = `
    <div id="welcome">
      <h2>Conversation effacée ✓</h2>
      <p>La mémoire a été réinitialisée.</p>
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
  if (data.system_prompt) md += '> **Contexte :** ' + data.system_prompt + '\n\n';
  if (data.skills?.length) md += '> **Skills :** ' + data.skills.join(', ') + '\n\n---\n\n';
  data.messages.forEach(m => {
    const role = m.role === 'user' ? '**Vous**' : '**IA**';
    md += `${role}\n\n${m.content}\n\n---\n\n`;
  });

  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([md], { type: 'text/markdown' })),
    download: 'conversation_' + Date.now() + '.md'
  });
  a.click();
  showToast('✓ Exporté', 'success');
}

/* =========================================================
   UTILS
   ========================================================= */
function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/'/g,"&#39;").replace(/"/g,'&quot;');
}

function showToast(msg, type = '') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className   = 'show ' + type;
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.className = '', 3000);
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeSkillModal();
});

/* =========================================================
   INIT
   ========================================================= */
(async () => {
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
    if (data.skills?.length) {
      skills = data.skills;
      renderSkills();
    }
    if (data.model) {
      document.getElementById('model-select').value      = data.model;
      document.getElementById('model-badge').textContent = data.model;
    }
  } catch (e) {
    console.error('Init error:', e);
  }
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    # Démarrer le thread keep-alive au lancement
    start_keep_alive_thread()
    
    logger.info("🚀 NeuralChat — Chat only avec Keep-Alive")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
