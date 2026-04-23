"""
NeuralChat — Application Flask complète en un seul fichier.
Fusion de : config.py, database.py, llm.py, session_helpers.py,
            routes_auth.py, routes_chat.py, routes_warroom.py, app.py

Usage :
    python main.py
    # ou avec variables d'environnement :
    OPENROUTER_API_KEY=sk-... SECRET_KEY=xxx python main.py

Le template HTML du chat est chargé depuis main.html (même dossier).
"""
from __future__ import annotations

# ===========================================================================
# IMPORTS
# ===========================================================================
import ast
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import closing
from typing import Any, Dict, Generator, List

import requests as http_requests
from flask import (
    Blueprint, Flask, Response, jsonify,
    redirect, render_template_string, request,
    session as flask_session, stream_with_context,
)
from werkzeug.security import check_password_hash, generate_password_hash


# ===========================================================================
# CONFIG
# ===========================================================================

API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL   = "openai/gpt-oss-120b:free"

FREE_MODELS: list[dict[str, str]] = [
    {"id": "minimax/minimax-m2.5:free",               "label": "MiniMax M2.5"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free",  "label": "Llama 3.3 70B"},
    {"id": "openrouter/elephant-alpha",               "label": "elephant"},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free",  "label": "Nemotron 3"},
    {"id": "arcee-ai/trinity-large-preview:free",     "label": "Trinity"},
    {"id": "openai/gpt-oss-120b:free",                "label": "GPT-oss"},
]

DB_PATH  = "neuralchat.db"
PING_URL = "https://neuralclaw.onrender.com"

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# DATABASE
# ===========================================================================

def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    with closing(get_db()) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username      TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                email         TEXT,
                created_at    TEXT NOT NULL
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


def _default_session() -> dict:
    return {
        "messages":      [],
        "system_prompt": "",
        "model":         DEFAULT_MODEL,
        "skills":        [],
        "lang":          "fr",
    }


def load_session_from_db(username: str) -> dict:
    with closing(get_db()) as db:
        row = db.execute(
            "SELECT * FROM chat_sessions WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            sess = _default_session()
            db.execute(
                "INSERT INTO chat_sessions "
                "(username, messages, system_prompt, model, skills, lang) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (username, json.dumps(sess["messages"]), sess["system_prompt"],
                 sess["model"], json.dumps(sess["skills"]), sess["lang"]),
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
    if username == "__anon__":
        return
    with closing(get_db()) as db:
        exists = db.execute(
            "SELECT 1 FROM chat_sessions WHERE username = ?", (username,)
        ).fetchone()
        if exists is None:
            db.execute(
                "INSERT INTO chat_sessions "
                "(username, messages, system_prompt, model, skills, lang) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (username, json.dumps(sess["messages"]), sess["system_prompt"],
                 sess["model"], json.dumps(sess["skills"]), sess.get("lang", "fr")),
            )
        else:
            db.execute(
                "UPDATE chat_sessions "
                "SET messages=?, system_prompt=?, model=?, skills=?, lang=? "
                "WHERE username=?",
                (json.dumps(sess["messages"]), sess["system_prompt"], sess["model"],
                 json.dumps(sess["skills"]), sess.get("lang", "fr"), username),
            )
        db.commit()


# ===========================================================================
# LLM
# ===========================================================================

def _get_headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }


def call_llm(
    messages:   list[dict],
    model:      str = DEFAULT_MODEL,
    max_tokens: int = 500000,
) -> str:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    resp = http_requests.post(
        OPENROUTER_URL, headers=_get_headers(), json=payload, timeout=180
    )
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        err = result["error"]
        raise RuntimeError(
            f"IA indisponible ({err.get('code', '?')}): {err.get('message', '')}"
        )
    return result["choices"][0]["message"]["content"]


def perform_search(query: str, max_results: int = 5) -> str:
    try:
        params: Dict[str, str] = {
            "q": query, "format": "json", "no_html": "1", "skip_disambig": "1",
        }
        response = http_requests.get(
            "https://api.duckduckgo.com/", params=params, timeout=10
        )
        response.raise_for_status()
        data = response.json()
        results: List[str] = []
        abstract     = data.get("AbstractText",   "").strip()
        abstract_src = data.get("AbstractSource", "").strip()
        abstract_url = data.get("AbstractURL",    "").strip()
        if abstract:
            results.append(
                f"Résultat instantané (source : {abstract_src or 'DuckDuckGo'})\n"
                f"{abstract}\n{abstract_url}\n"
            )
        else:
            results.append(f"Recherche Internet pour « {query} » :\n")
        related: List[Dict] = data.get("RelatedTopics", [])
        count = 0
        for item in related:
            if count >= max_results:
                break
            if isinstance(item, dict) and "FirstURL" in item:
                title   = item.get("Text", "Titre inconnu").split(" – ")[0]
                snippet = re.sub(r"<[^>]+>", "", item.get("Text", ""))
                url     = item.get("FirstURL", "#")
                results.append(f"{count + 1}. {title}\n{snippet[:200]}\n{url}\n")
                count += 1
            elif isinstance(item, dict) and "Topics" in item:
                for sub in item["Topics"]:
                    if count >= max_results:
                        break
                    title   = sub.get("Text", "").split(" – ")[0]
                    snippet = re.sub(r"<[^>]+>", "", sub.get("Text", ""))
                    url     = sub.get("FirstURL", "#")
                    results.append(f"{count + 1}. {title}\n{snippet[:200]}\n{url}\n")
                    count += 1
        if count == 0 and not abstract:
            results.append("Aucun résultat trouvé.")
        return "\n".join(results)
    except Exception as exc:
        return f"Erreur de recherche : {exc}"


# ===========================================================================
# SESSION HELPERS
# ===========================================================================

_anon_store: dict[str, dict[str, Any]] = {}

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


def get_session() -> dict:
    username: str = flask_session.get("username", "__anon__")
    if username == "__anon__":
        if "__anon__" not in _anon_store:
            _anon_store["__anon__"] = {
                "messages": [], "system_prompt": "", "model": DEFAULT_MODEL,
                "skills": [], "lang": "fr",
            }
        return _anon_store["__anon__"]
    return load_session_from_db(username)


def build_system_content(sess: dict) -> str:
    parts: list[str] = []
    if sess.get("skills"):
        parts.append(f"Skills and areas of expertise: {', '.join(sess['skills'])}.")
    if sess.get("system_prompt"):
        parts.append(sess["system_prompt"])
    return "\n\n".join(parts)


def estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


# ===========================================================================
# TEMPLATES HTML
# ===========================================================================

_BASE = os.path.dirname(os.path.abspath(__file__))

LOGIN_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NeuralChat — Connexion</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#0a0e1a;--panel:#111827;--border:#1e2d45;--accent:#3b82f6;--accent2:#06b6d4;--accent3:#8b5cf6;--text:#e2e8f0;--muted:#64748b;--danger:#ef4444;--success:#22c55e}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'JetBrains Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;position:relative;overflow:hidden}
.bg-blob{position:absolute;border-radius:50%;filter:blur(80px);opacity:.12;animation:drift 12s ease-in-out infinite alternate;pointer-events:none}
.bg-blob-1{width:500px;height:500px;background:var(--accent);top:-150px;left:-150px;animation-delay:0s}
.bg-blob-2{width:400px;height:400px;background:var(--accent3);bottom:-100px;right:-100px;animation-delay:3s}
.bg-blob-3{width:300px;height:300px;background:var(--accent2);top:40%;left:50%;animation-delay:6s}
@keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(30px,-30px) scale(1.08)}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:20px;padding:44px 40px;width:100%;max-width:420px;position:relative;z-index:1;box-shadow:0 24px 80px rgba(0,0,0,.5)}
.logo{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-align:center;margin-bottom:6px}
.logo-sub{text-align:center;font-size:12px;color:var(--muted);margin-bottom:36px}
.tabs{display:flex;background:var(--bg);border-radius:10px;padding:4px;margin-bottom:28px;border:1px solid var(--border)}
.tab-btn{flex:1;padding:9px;border:none;border-radius:8px;background:transparent;color:var(--muted);font-family:'Syne',sans-serif;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s}
.tab-btn.active{background:var(--accent);color:#fff}
.form-group{display:flex;flex-direction:column;gap:6px;margin-bottom:16px}
.form-group label{font-size:11px;color:var(--muted);letter-spacing:.5px}
.form-group input{background:var(--bg);border:1px solid var(--border);border-radius:10px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:13px;padding:13px 16px;outline:none;transition:border-color .2s,box-shadow .2s;width:100%}
.form-group input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(59,130,246,.15)}
.form-group input::placeholder{color:var(--muted)}
.submit-btn{width:100%;padding:14px;border:none;border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;font-family:'Syne',sans-serif;font-size:14px;font-weight:700;cursor:pointer;transition:opacity .2s,transform .15s;margin-top:8px}
.submit-btn:hover:not(:disabled){opacity:.9;transform:translateY(-1px)}
.submit-btn:disabled{opacity:.5;cursor:not-allowed}
.error-msg{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:var(--danger);border-radius:8px;padding:10px 14px;font-size:12px;margin-top:12px;display:none;animation:shake .3s ease}
@keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-6px)}75%{transform:translateX(6px)}}
.error-msg.visible{display:block}
.success-msg{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);color:var(--success);border-radius:8px;padding:10px 14px;font-size:12px;margin-top:12px;display:none}
.success-msg.visible{display:block}
.divider{text-align:center;color:var(--muted);font-size:11px;margin:20px 0 16px;position:relative}
.divider::before,.divider::after{content:'';position:absolute;top:50%;width:40%;height:1px;background:var(--border)}
.divider::before{left:0}
.divider::after{right:0}
.guest-btn{width:100%;padding:12px;border:1px solid var(--border);border-radius:10px;background:transparent;color:var(--muted);font-family:'Syne',sans-serif;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s}
.guest-btn:hover{border-color:var(--accent3);color:var(--accent3)}
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
  <div id="form-login">
    <div class="form-group"><label>Nom d'utilisateur</label><input type="text" id="login-username" placeholder="votre_pseudo" autocomplete="username"/></div>
    <div class="form-group"><label>Mot de passe</label><input type="password" id="login-password" placeholder="••••••••" autocomplete="current-password"/></div>
    <button class="submit-btn" id="login-btn" onclick="doLogin()">Se connecter →</button>
    <div class="error-msg"   id="login-error"></div>
    <div class="success-msg" id="login-success"></div>
  </div>
  <div id="form-register" style="display:none">
    <div class="form-group"><label>Nom d'utilisateur <span style="color:var(--danger)">*</span></label><input type="text" id="reg-username" placeholder="votre_pseudo" autocomplete="username"/></div>
    <div class="form-group"><label>Email <span style="color:var(--muted);font-size:10px;">(optionnel)</span></label><input type="email" id="reg-email" placeholder="you@example.com" autocomplete="email"/></div>
    <div class="form-group"><label>Mot de passe <span style="color:var(--danger)">*</span> <span style="color:var(--muted);font-size:10px;">(min. 6 caractères)</span></label><input type="password" id="reg-password" placeholder="••••••••" autocomplete="new-password"/></div>
    <div class="form-group"><label>Confirmer le mot de passe <span style="color:var(--danger)">*</span></label><input type="password" id="reg-confirm" placeholder="••••••••" autocomplete="new-password"/></div>
    <button class="submit-btn" id="register-btn" onclick="doRegister()">Créer mon compte →</button>
    <div class="error-msg"   id="register-error"></div>
    <div class="success-msg" id="register-success"></div>
  </div>
  <div class="divider">ou</div>
  <button class="guest-btn" onclick="window.location='/'">Continuer sans compte</button>
</div>
<script>
function switchTab(tab){
  document.getElementById('form-login').style.display=tab==='login'?'block':'none';
  document.getElementById('form-register').style.display=tab==='register'?'block':'none';
  document.getElementById('tab-login').classList.toggle('active',tab==='login');
  document.getElementById('tab-register').classList.toggle('active',tab==='register');
  clearMessages();
}
function clearMessages(){
  ['login-error','login-success','register-error','register-success'].forEach(id=>{
    const el=document.getElementById(id);el.className=el.className.replace(' visible','');
  });
}
function showMsg(id,msg){const el=document.getElementById(id);el.textContent=msg;el.classList.add('visible');}
async function doLogin(){
  const btn=document.getElementById('login-btn');
  const username=document.getElementById('login-username').value.trim();
  const password=document.getElementById('login-password').value.trim();
  clearMessages();
  if(!username||!password)return showMsg('login-error','Veuillez remplir tous les champs');
  btn.disabled=true;btn.textContent='Connexion…';
  try{
    const res=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username,password})});
    const data=await res.json();
    if(data.ok){showMsg('login-success','Bienvenue, '+data.username+' !');setTimeout(()=>window.location='/',800);}
    else{showMsg('login-error',data.error||'Erreur inconnue');btn.disabled=false;btn.textContent='Se connecter →';}
  }catch{showMsg('login-error','Erreur réseau');btn.disabled=false;btn.textContent='Se connecter →';}
}
async function doRegister(){
  const btn=document.getElementById('register-btn');
  const username=document.getElementById('reg-username').value.trim();
  const email=document.getElementById('reg-email').value.trim();
  const password=document.getElementById('reg-password').value.trim();
  const confirm=document.getElementById('reg-confirm').value.trim();
  clearMessages();
  if(!username||!password||!confirm)return showMsg('register-error','Veuillez remplir les champs obligatoires');
  if(password!==confirm)return showMsg('register-error','Les mots de passe ne correspondent pas');
  if(password.length<6)return showMsg('register-error','Le mot de passe doit faire au moins 6 caractères');
  btn.disabled=true;btn.textContent='Création…';
  try{
    const res=await fetch('/api/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username,password,email})});
    const data=await res.json();
    if(data.ok){showMsg('register-success','Compte créé ! Redirection…');setTimeout(()=>window.location='/',900);}
    else{showMsg('register-error',data.error||'Erreur inconnue');btn.disabled=false;btn.textContent='Créer mon compte →';}
  }catch{showMsg('register-error','Erreur réseau');btn.disabled=false;btn.textContent='Créer mon compte →';}
}
document.addEventListener('keydown',e=>{
  if(e.key!=='Enter')return;
  if(document.getElementById('form-login').style.display!=='none')doLogin();
  else doRegister();
});
</script>
</body>
</html>"""


def _load_html_template() -> str:
    path = os.path.join(_BASE, "main.html")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ===========================================================================
# WAR ROOM — Constantes & helpers
# ===========================================================================

MAX_RETRIES        = 3
RETRY_DELAY        = 1.5
RETRY_BACKOFF      = 2.0
TOKENS_PM          = 2_500
TOKENS_ORCHESTRATOR= 2_000
TOKENS_AGENT       = 2_000
TOKENS_DEBATE      = 4_000
TOKENS_ADVOCATE    = 3_500
TOKENS_SYNTHESIS   = 10_000
TOKENS_PROGRAMMER  = 64_000
TOKENS_REVIEWER    = 8_000
TOKENS_TESTS       = 12_000
TOKENS_CONTINUATION= 32_000
TOKENS_SUMMARY     = 1_500
EXPERTS_TIMEOUT_SECONDS = 300
FINAL_CODE_END_MARKER   = "__WARROOM_COMPLETE__"
MAX_QUERY_LENGTH        = 8_000

ALLOWED_MODELS: set[str] = {
    "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
    "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-6",
    DEFAULT_MODEL,
}

MODE_PRESETS: dict[str, dict] = {
    "fast": {
        "num_experts": 1, "debate_rounds": 0, "advocate": False,
        "reviewer": False, "generate_tests": False, "executive_summary": False,
        "tokens_agent": 3_000, "tokens_programmer": 16_000,
    },
    "balanced": {
        "num_experts": 3, "debate_rounds": 0, "advocate": True,
        "reviewer": True, "generate_tests": False, "executive_summary": True,
        "tokens_agent": 8_000, "tokens_programmer": 64_000,
    },
    "full": {
        "num_experts": 5, "debate_rounds": 1, "advocate": True,
        "reviewer": True, "generate_tests": True, "executive_summary": True,
        "tokens_agent": 8_000, "tokens_programmer": 64_000,
    },
}

_cache_store:   dict[str, dict]  = {}
_session_store: dict[str, list]  = {}
_job_store:     dict[str, dict]  = {}
_rate_store:    dict[str, list]  = defaultdict(list)
_feedback_store: list[dict]      = []

CACHE_TTL_SECONDS  = 3600
RATE_LIMIT_MAX     = 20
RATE_LIMIT_WINDOW  = 60


def _cache_get(key: str) -> dict | None:
    entry = _cache_store.get(key)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        del _cache_store[key]
        return None
    return entry["value"]


def _cache_set(key: str, value: dict, ttl: int = CACHE_TTL_SECONDS) -> None:
    _cache_store[key] = {"value": value, "expires_at": time.time() + ttl}


def _cache_key(query: str, model: str, experts: list[str]) -> str:
    raw = f"{query}|{model}|{'_'.join(sorted(experts))}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _rate_check(ip: str) -> bool:
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_store[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_store[ip].append(now)
    return True


def call_llm_with_retry(
        messages: list[dict], model: str = DEFAULT_MODEL,
        max_tokens: int = 4_000, step_name: str = "LLM",
) -> str:
    delay = RETRY_DELAY
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = call_llm(messages, model=model, max_tokens=max_tokens)
            if not isinstance(result, str) or not result.strip():
                raise ValueError("Réponse LLM vide")
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                logger.warning("[%s] Tentative %d/%d échouée : %s — retry dans %.1fs",
                               step_name, attempt, MAX_RETRIES, exc, delay)
                time.sleep(delay)
                delay *= RETRY_BACKOFF
            else:
                logger.error("[%s] Échec définitif (%d/%d) : %s", step_name, attempt, MAX_RETRIES, exc)
    raise last_exc  # type: ignore[misc]


def _clean_markdown_fences(raw: str) -> str:
    return re.sub(r"```(?:json|python|javascript|ts|tsx|js|html|bash)?|```", "", raw).strip()


def _parse_json_llm(raw: str) -> dict:
    clean = _clean_markdown_fences(raw)
    try:
        return json.loads(clean)
    except Exception:
        match = re.search(r"\{.*\}", clean, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _is_likely_truncated(text: str) -> bool:
    if not text or len(text.strip()) < 80:
        return True
    stripped = text.rstrip()
    suspicious = ("```", "def ", "class ", "return", "except", "finally",
                  "else:", "elif ", "for ", "while ", "if ", "{", "[", "(", ",", ":")
    if any(stripped.endswith(x) for x in suspicious):
        return True
    if stripped.count("```") % 2 != 0:
        return True
    return FINAL_CODE_END_MARKER not in stripped


def _safe_excerpt(text: str, max_chars: int = 8_000) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[tronqué]"


def _normalize_expert_payload(expert: dict[str, Any]) -> dict[str, str]:
    return {
        "role":      str(expert.get("role", "Expert")),
        "emoji":     str(expert.get("emoji", "🔹")),
        "specialty": str(expert.get("specialty", "Analyse spécialisée")),
    }


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_STACK_KEYWORDS: dict[str, list[str]] = {
    "python/fastapi":    ["fastapi", "uvicorn", "pydantic", "async def"],
    "python/flask":      ["flask", "blueprint", "app.route", "jsonify"],
    "python/django":     ["django", "models.py", "views.py", "urls.py"],
    "python":            ["python", ".py", "pip", "pandas", "numpy", "pytest"],
    "typescript/react":  ["react", "tsx", "jsx", "next.js", "nextjs", "vite"],
    "typescript/node":   ["express", "nestjs", "ts-node", "typescript node"],
    "typescript":        ["typescript", ".ts", "tsc", "interface ", "type "],
    "javascript":        ["javascript", "node.js", "nodejs", ".js", "npm", "yarn"],
    "go":                ["golang", " go ", ".go", "goroutine", "gin"],
    "rust":              ["rust", "cargo", "tokio", ".rs"],
    "java":              ["java", "spring", "maven", "gradle", ".java"],
    "kotlin":            ["kotlin", ".kt", "coroutine"],
    "swift":             ["swift", "swiftui", "uikit", ".swift"],
    "sql":               ["sql", "postgres", "mysql", "sqlite", "query"],
}


def _detect_stack(query: str, explicit_stack: str | None = None) -> str | None:
    if explicit_stack:
        return explicit_stack
    q = query.lower()
    for stack, keywords in _STACK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return stack
    return None


def _validate_syntax(code: str, lang: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if lang == "python":
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as exc:
            errors.append(f"SyntaxError ligne {exc.lineno}: {exc.msg}")
            return False, errors
    opens  = code.count("{") + code.count("(") + code.count("[")
    closes = code.count("}") + code.count(")") + code.count("]")
    if abs(opens - closes) > 3:
        errors.append(f"Déséquilibre parenthèses : {opens} ouvrantes / {closes} fermantes")
    return len(errors) == 0, errors


def _extract_files(programmer_output: str) -> list[dict]:
    files: list[dict] = []
    pattern = re.compile(r"###\s*FILE:\s*(\S+)\s*\n```(\w*)\n(.*?)```", re.S)
    for match in pattern.finditer(programmer_output):
        filename, lang_hint, content = match.group(1), match.group(2).lower(), match.group(3)
        if not lang_hint:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            lang_hint = {
                "py": "python", "ts": "typescript", "tsx": "typescript",
                "js": "javascript", "jsx": "javascript", "go": "go",
                "rs": "rust", "java": "java", "kt": "kotlin",
                "swift": "swift", "sql": "sql", "sh": "bash", "html": "html", "css": "css",
            }.get(ext, ext)
        valid, errs = _validate_syntax(content, lang_hint)
        files.append({
            "filename": filename, "language": lang_hint, "content": content.strip(),
            "line_count": content.count("\n") + 1, "syntax_valid": valid, "syntax_errors": errs,
        })
    if not files:
        code_match = re.search(r"```(\w*)\n(.*?)```", programmer_output, re.S)
        if code_match:
            lang_hint = code_match.group(1).lower() or "text"
            content   = code_match.group(2)
            valid, errs = _validate_syntax(content, lang_hint)
            files.append({
                "filename": f"output.{lang_hint or 'txt'}", "language": lang_hint,
                "content": content.strip(), "line_count": content.count("\n") + 1,
                "syntax_valid": valid, "syntax_errors": errs,
            })
    return files


def _clean_programmer_output(raw: str) -> str:
    return raw.replace(FINAL_CODE_END_MARKER, "").rstrip()


# ===========================================================================
# WAR ROOM — Expert profiles
# ===========================================================================

EXPERT_FAMILIES: list[dict] = [
    {"id": "informatique", "label": "💻 Informatique & Tech",       "color": "#3b82f6"},
    {"id": "art",          "label": "🎨 Art & Design",              "color": "#ec4899"},
    {"id": "business",     "label": "💼 Business & Management",     "color": "#f97316"},
    {"id": "sante",        "label": "🏥 Santé & Bien-être",         "color": "#22c55e"},
    {"id": "sciences",     "label": "🔬 Sciences",                   "color": "#06b6d4"},
    {"id": "education",    "label": "📚 Éducation & Coaching",      "color": "#a855f7"},
    {"id": "societe",      "label": "🌍 Société & Environnement",   "color": "#f59e0b"},
]

EXPERT_PROFILES: dict[str, dict] = {
    "dev_senior":    {"role":"Développeur Senior","emoji":"💻","family":"informatique","description":"Architecte logiciel, code propre et performance.","specialty":"Architecture logicielle, design patterns, refactoring, bonnes pratiques, performance applicative et revue de code."},
    "frontend_dev":  {"role":"Développeur Frontend","emoji":"🖥️","family":"informatique","description":"Interfaces web modernes et réactives.","specialty":"React, Vue, Angular, CSS avancé, accessibilité WCAG et optimisation front."},
    "backend_dev":   {"role":"Développeur Backend","emoji":"🔧","family":"informatique","description":"APIs robustes, bases de données et microservices.","specialty":"APIs REST/GraphQL, microservices, SQL/NoSQL, caching, scalabilité et sécurité backend."},
    "mobile_dev":    {"role":"Développeur Mobile","emoji":"📱","family":"informatique","description":"Applications iOS, Android et cross-platform.","specialty":"React Native, Flutter, Swift, Kotlin, UX mobile et optimisation des performances."},
    "data_scientist":{"role":"Data Scientist","emoji":"📊","family":"informatique","description":"Analyse de données, ML et statistiques.","specialty":"Machine learning, statistiques, visualisation et évaluation de modèles."},
    "ml_engineer":   {"role":"ML Engineer","emoji":"🧠","family":"informatique","description":"Pipelines IA, déploiement et fine-tuning de modèles.","specialty":"MLOps, déploiement, pipelines de données et monitoring de modèles."},
    "data_engineer": {"role":"Data Engineer","emoji":"🗄️","family":"informatique","description":"ETL, data warehouses et pipelines de données.","specialty":"Spark, Kafka, Airflow, dbt, data lakes et gouvernance des données."},
    "security_expert":{"role":"Expert Cybersécurité","emoji":"🔒","family":"informatique","description":"Protection des systèmes et gestion des vulnérabilités.","specialty":"Pentest, threat modeling, IAM, cryptographie, cloud security et réponse aux incidents."},
    "devops":        {"role":"DevOps / Cloud","emoji":"⚙️","family":"informatique","description":"Infrastructure, CI/CD et scalabilité cloud.","specialty":"Docker, Kubernetes, Terraform, CI/CD, observabilité et SRE."},
    "ux_designer":   {"role":"UX/UI Designer","emoji":"🎨","family":"art","description":"Expérience utilisateur et interfaces intuitives.","specialty":"Research UX, wireframes, prototypage, design systems et accessibilité."},
    "creative_dir":  {"role":"Directeur Créatif","emoji":"✨","family":"art","description":"Innovation visuelle, branding et storytelling.","specialty":"Direction artistique, identité de marque et stratégie créative."},
    "graphic_designer":{"role":"Graphiste","emoji":"🖌️","family":"art","description":"Identité visuelle, print et illustration.","specialty":"Branding, illustration, mise en page éditoriale et packaging."},
    "motion_designer":{"role":"Motion Designer","emoji":"🎬","family":"art","description":"Animations, vidéo et effets visuels.","specialty":"Animation 2D/3D, motion graphics, storyboard et post-production."},
    "photographer":  {"role":"Directeur Photo","emoji":"📷","family":"art","description":"Composition, lumière et direction artistique.","specialty":"Direction photo, composition, éclairage et post-traitement."},
    "copywriter":    {"role":"Copywriter","emoji":"✍️","family":"art","description":"Contenu persuasif, storytelling et conversion.","specialty":"Copywriting, content marketing, SEO rédactionnel et UX writing."},
    "architect":     {"role":"Architecte / Designer Intérieur","emoji":"🏛️","family":"art","description":"Conception spatiale, ergonomie et esthétique.","specialty":"Conception architecturale, design d'intérieur et durabilité."},
    "product_manager":{"role":"Product Manager","emoji":"🗺️","family":"business","description":"Roadmap produit, priorisation et go-to-market.","specialty":"Vision produit, user stories, OKRs, priorisation et métriques produit."},
    "business_strat":{"role":"Stratège Business","emoji":"🎯","family":"business","description":"Vision stratégique, business model et croissance.","specialty":"SWOT, business model, analyse concurrentielle et stratégie de croissance."},
    "marketing":     {"role":"Expert Marketing","emoji":"📣","family":"business","description":"Acquisition, branding et growth hacking.","specialty":"Stratégie digitale, SEO/SEA, growth et analyse d'audience."},
    "finance":       {"role":"Analyste Finance","emoji":"💰","family":"business","description":"ROI, budgets, rentabilité et levée de fonds.","specialty":"Analyse financière, modélisation, trésorerie et gestion des risques."},
    "legal":         {"role":"Conseiller Légal","emoji":"⚖️","family":"business","description":"Droit des affaires, contrats et propriété intellectuelle.","specialty":"Contrats, conformité, propriété intellectuelle et protection des données."},
    "hr_expert":     {"role":"Expert RH & Management","emoji":"👥","family":"business","description":"Recrutement, culture d'entreprise et leadership.","specialty":"Stratégie RH, talents, leadership et conduite du changement."},
    "entrepreneur":  {"role":"Entrepreneur / Startuper","emoji":"🚀","family":"business","description":"Lean startup, MVP, pivots et traction.","specialty":"Lean startup, validation d'hypothèses, MVP et product-market fit."},
    "project_manager":{"role":"Chef de Projet","emoji":"📋","family":"business","description":"Planning, ressources et livraison des projets.","specialty":"Agile, gestion des risques, coordination et pilotage des KPIs."},
    "medecin":       {"role":"Médecin Généraliste","emoji":"🩺","family":"sante","description":"Diagnostics, prévention et santé globale.","specialty":"Prévention, orientation clinique et médecine basée sur les preuves."},
    "psychologue":   {"role":"Psychologue","emoji":"🧠","family":"sante","description":"Santé mentale, thérapies et bien-être émotionnel.","specialty":"TCC, stress, burn-out et relation d'aide."},
    "nutritionniste":{"role":"Nutritionniste","emoji":"🥗","family":"sante","description":"Alimentation équilibrée et santé métabolique.","specialty":"Nutrition, micronutrition, troubles alimentaires et performance."},
    "pharmacien":    {"role":"Pharmacien","emoji":"💊","family":"sante","description":"Médicaments, interactions et protocoles de traitement.","specialty":"Pharmacologie, interactions médicamenteuses et suivi des traitements."},
    "coach_sportif": {"role":"Coach Sportif / Kinésithérapeute","emoji":"🏋️","family":"sante","description":"Performance physique, entraînement et récupération.","specialty":"Planification d'entraînement, biomécanique et prévention des blessures."},
    "physicien":     {"role":"Physicien","emoji":"⚛️","family":"sciences","description":"Mécanique, énergie, physique quantique et optique.","specialty":"Physique théorique et appliquée, matériaux et modélisation."},
    "biologiste":    {"role":"Biologiste","emoji":"🧬","family":"sciences","description":"Génétique, biologie cellulaire et écosystèmes.","specialty":"Biologie moléculaire, microbiologie, biotechnologies et écologie."},
    "chimiste":      {"role":"Chimiste","emoji":"🧪","family":"sciences","description":"Réactions, matériaux et formulations.","specialty":"Chimie analytique, organique, matériaux et sécurité chimique."},
    "mathematicien": {"role":"Mathématicien","emoji":"📐","family":"sciences","description":"Modélisation, algorithmes et statistiques avancées.","specialty":"Probabilités, optimisation, algorithmes et cryptographie."},
    "ingenieur":     {"role":"Ingénieur Généraliste","emoji":"🔩","family":"sciences","description":"Conception mécanique, fabrication et optimisation.","specialty":"Conception, simulation, production et contrôle qualité."},
    "formateur":     {"role":"Formateur / Pédagogue","emoji":"🎓","family":"education","description":"Conception pédagogique et transmission des savoirs.","specialty":"Ingénierie pédagogique, e-learning et évaluation des apprentissages."},
    "coach_life":    {"role":"Coach de Vie / Executive Coach","emoji":"🌱","family":"education","description":"Développement personnel, objectifs et mindset.","specialty":"Coaching orienté solution, objectifs SMART et leadership personnel."},
    "philosophe":    {"role":"Philosophe / Éthicien","emoji":"🦉","family":"education","description":"Éthique, logique et analyse critique.","specialty":"Éthique appliquée, logique argumentative et pensée critique."},
    "sociologue":    {"role":"Sociologue","emoji":"🌐","family":"societe","description":"Comportements sociaux, tendances et cultures.","specialty":"Analyse sociologique, dynamiques de groupe et tendances culturelles."},
    "economiste":    {"role":"Économiste","emoji":"📈","family":"societe","description":"Marchés, politiques économiques et analyses macro.","specialty":"Micro, macro, politiques publiques et économétrie."},
    "expert_rse":    {"role":"Expert RSE / Durabilité","emoji":"♻️","family":"societe","description":"Responsabilité sociale et impact environnemental.","specialty":"Stratégie RSE, bilan carbone, CSRD et transition énergétique."},
    "journaliste":   {"role":"Journaliste / Analyste","emoji":"📰","family":"societe","description":"Investigation, fact-checking et communication.","specialty":"Fact-checking, investigation, communication de crise et analyse médiatique."},
}


# ===========================================================================
# WAR ROOM — Sélection experts & pipeline
# ===========================================================================

def _select_experts(
        query: str, model: str, lang_rule: str,
        selected_experts: list[str], num_experts: int = 3,
) -> tuple[list[dict], str]:
    num_experts = max(1, min(num_experts, 6))
    forced = [_normalize_expert_payload(EXPERT_PROFILES[eid])
              for eid in selected_experts if eid in EXPERT_PROFILES]
    if len(forced) >= num_experts:
        experts = forced[:num_experts]
        return experts, f"Analyse selon les profils sélectionnés : {', '.join(e['role'] for e in experts)}"
    if 0 < len(forced) < num_experts:
        local_fillers = [
            {"role":"Développeur Senior","emoji":"💻","specialty":"Architecture, qualité et complétude du code"},
            {"role":"Chef de Projet","emoji":"📋","specialty":"Priorisation, risques et exécution rapide"},
            {"role":"Expert QA","emoji":"✅","specialty":"Tests, robustesse, cas limites et fiabilité"},
            {"role":"Expert Cybersécurité","emoji":"🔒","specialty":"Sécurité, vulnérabilités et conformité"},
            {"role":"DevOps / Cloud","emoji":"⚙️","specialty":"Déploiement, CI/CD et observabilité"},
            {"role":"Stratège Business","emoji":"🎯","specialty":"Vision stratégique et business model"},
        ]
        picked_roles = {e["role"] for e in forced}
        fillers  = [f for f in local_fillers if f["role"] not in picked_roles]
        experts  = (forced + fillers)[:num_experts]
        return experts, "Analyse hybride rapide avec experts sélectionnés et compléments locaux"
    prompt = (
        f"{lang_rule}\n"
        f"Tu es un orchestrateur minimaliste. Choisis EXACTEMENT {num_experts} experts utiles.\n"
        '{"experts": [{"role": "...", "emoji": "...", "specialty": "..."}], "strategy": "..."}\n\n'
        f"Demande : {query}"
    )
    try:
        ai_data = _parse_json_llm(call_llm_with_retry(
            [{"role":"user","content":prompt}], model=model,
            max_tokens=TOKENS_ORCHESTRATOR, step_name="Orchestrateur",
        ))
        experts = [_normalize_expert_payload(e) for e in ai_data.get("experts", [])][:num_experts]
        if len(experts) < num_experts:
            raise ValueError(f"Pas assez d'experts ({len(experts)}/{num_experts})")
        return experts, ai_data.get("strategy", "Analyse multi-angle")
    except Exception as exc:
        logger.warning("Orchestrateur fallback : %s", exc)
        fallback = [
            {"role":"Expert Technique","emoji":"💻","specialty":"Analyse technique et implémentation"},
            {"role":"Stratège","emoji":"🎯","specialty":"Vision stratégique et risques"},
            {"role":"Expert QA","emoji":"✅","specialty":"Tests, robustesse et validation"},
            {"role":"Expert Cybersécurité","emoji":"🔒","specialty":"Sécurité et conformité"},
            {"role":"DevOps","emoji":"⚙️","specialty":"Déploiement et infrastructure"},
            {"role":"Product Manager","emoji":"🗺️","specialty":"Vision produit et priorisation"},
        ]
        return fallback[:num_experts], "Analyse multi-angle en fallback"


def _build_pm_data(query: str, model: str, lang_rule: str) -> tuple[dict, float]:
    t0 = time.time()
    pm_prompt = (
        f"{lang_rule}\nTu es un Chef de Projet Senior. Fournis un cadrage TRÈS CONCIS.\n"
        "Retourne UNIQUEMENT un JSON valide :\n"
        '{"project_title":"...","objective":"...","action_plan":[{"step":1,"action":"...","priority":"haute|moyenne|basse"}],"key_risks":["..."],"success_criteria":"..."}\n\n'
        f"Demande : {query}"
    )
    try:
        result = _parse_json_llm(call_llm_with_retry(
            [{"role":"user","content":pm_prompt}], model=model,
            max_tokens=TOKENS_PM, step_name="Chef-de-Projet",
        ))
        return result, (time.time() - t0) * 1000
    except Exception as exc:
        logger.warning("Chef de projet fallback : %s", exc)
        return {
            "project_title": "Analyse en cours", "objective": "Résoudre la demande",
            "action_plan": [{"step":1,"action":"Cadrer le besoin","priority":"haute"},
                            {"step":2,"action":"Produire une solution","priority":"haute"}],
            "key_risks": ["Ambiguïté du besoin"], "success_criteria": "Réponse complète et exploitable",
        }, (time.time() - t0) * 1000


def _call_agent(query: str, model: str, lang_rule: str, expert: dict, idx: int, tokens_agent: int = TOKENS_AGENT) -> dict:
    query_excerpt   = _safe_excerpt(query, 2_000)
    effective_tokens = min(tokens_agent, 1_500)
    agent_prompt = (
        f"{lang_rule}\nTu es {expert['role']} ({expert['specialty']}).\n"
        "Analyse COURTE et très actionnable en 4 points (2-3 phrases max chacun) :\n"
        "1) Diagnostic\n2) Recommandations\n3) Risques\n4) Actions immédiates\n\n"
        f"Demande : {query_excerpt}"
    )
    t0 = time.time()
    try:
        response = call_llm_with_retry(
            [{"role":"user","content":agent_prompt}], model=model,
            max_tokens=effective_tokens, step_name=f"Expert-{idx}-{expert['role']}",
        )
        return {"expert":expert,"proposal":response,"idx":idx,"error":None,
                "duration_ms":int((time.time()-t0)*1000)}
    except Exception as exc:
        logger.warning("Expert %s erreur : %s", expert.get("role"), exc)
        return {"expert":expert,"proposal":"Analyse indisponible.",
                "idx":idx,"error":str(exc),"duration_ms":int((time.time()-t0)*1000)}


def _run_parallel_experts(query: str, model: str, lang_rule: str, experts: list[dict], tokens_agent: int = TOKENS_AGENT) -> list[dict]:
    proposals: list[dict | None] = [None] * len(experts)
    if not experts:
        return []
    with ThreadPoolExecutor(max_workers=len(experts)) as executor:
        futures = {executor.submit(_call_agent, query, model, lang_rule, exp, i, tokens_agent): i
                   for i, exp in enumerate(experts)}
        for future, idx in futures.items():
            expert = experts[idx]
            try:
                result = future.result(timeout=EXPERTS_TIMEOUT_SECONDS)
                proposals[idx] = result
            except TimeoutError:
                proposals[idx] = {"expert":expert,"proposal":"Analyse non revenue à temps.",
                                  "idx":idx,"error":"timeout","duration_ms":EXPERTS_TIMEOUT_SECONDS*1000}
            except Exception as exc:
                proposals[idx] = {"expert":expert,"proposal":"Analyse indisponible.",
                                  "idx":idx,"error":str(exc),"duration_ms":0}
    return [p for p in proposals if p]


def _run_expert_debate(query: str, model: str, lang_rule: str, proposals: list[dict], rounds: int = 1) -> list[dict]:
    if rounds <= 0 or len(proposals) < 2:
        return proposals
    updated = list(proposals)
    for round_idx in range(1, rounds + 1):
        others_text_by_idx: dict[int, str] = {}
        for p in updated:
            others = [f"--- {o['expert']['emoji']} {o['expert']['role']} ---\n{_safe_excerpt(o['proposal'],2500)}"
                      for o in updated if o["idx"] != p["idx"]]
            others_text_by_idx[p["idx"]] = "\n\n".join(others)

        def _debate_one(p: dict) -> dict:
            debate_prompt = (
                f"{lang_rule}\nTu es {p['expert']['role']} ({p['expert']['specialty']}).\n"
                "Tu viens de lire les analyses de tes collègues. Enrichis ou corrige ta position.\n"
                "Structure :\n1) Points d'accord\n2) Points de désaccord\n3) Position enrichie\n\n"
                f"DEMANDE : {query}\n\nTA PREMIÈRE ANALYSE :\n{_safe_excerpt(p['proposal'],2000)}\n\n"
                f"ANALYSES DES AUTRES :\n{others_text_by_idx[p['idx']]}"
            )
            try:
                updated_proposal = call_llm_with_retry(
                    [{"role":"user","content":debate_prompt}], model=model,
                    max_tokens=TOKENS_DEBATE, step_name=f"Débat-R{round_idx}-{p['expert']['role']}",
                )
                return {**p, "proposal": updated_proposal, "debated": True}
            except Exception as exc:
                logger.warning("Débat %s R%d erreur : %s", p["expert"]["role"], round_idx, exc)
                return {**p, "debated": False}

        with ThreadPoolExecutor(max_workers=len(updated)) as ex:
            futures_debate = {ex.submit(_debate_one, p): p["idx"] for p in updated}
            new_updated = list(updated)
            for fut, idx in futures_debate.items():
                try:
                    res = fut.result(timeout=EXPERTS_TIMEOUT_SECONDS)
                    pos = next(i for i, p in enumerate(new_updated) if p["idx"] == idx)
                    new_updated[pos] = res
                except Exception as exc:
                    logger.warning("Résultat débat idx %d perdu : %s", idx, exc)
            updated = new_updated
    return updated


def _run_advocate(query: str, synthesis: str, model: str, lang_rule: str) -> str:
    prompt = (
        f"{lang_rule}\nTu es l'Avocat du Diable. Analyse la synthèse et identifie les failles.\n"
        "Structure :\n## Failles critiques\n## Risques majeurs\n## Hypothèses dangereuses\n## Points de vigilance pour le code\n\n"
        f"DEMANDE ORIGINALE : {query}\n\nSYNTHÈSE :\n{_safe_excerpt(synthesis, 5000)}"
    )
    try:
        return call_llm_with_retry(
            [{"role":"user","content":prompt}], model=model,
            max_tokens=TOKENS_ADVOCATE, step_name="Avocat-du-Diable",
        )
    except Exception as exc:
        logger.warning("Avocat du diable erreur : %s", exc)
        return "Critique indisponible."


def _build_synthesis(query: str, pm_data: dict, proposals: list[dict], model: str, lang_rule: str) -> tuple[str, float]:
    t0 = time.time()
    proposals_text = "\n\n".join(
        f"--- {p['expert']['emoji']} {p['expert']['role']} ---\n{_safe_excerpt(p['proposal'],7000)}"
        for p in proposals
    )
    synthesis_prompt = (
        f"{lang_rule}\nTu es un synthétiseur critique.\n"
        f"DEMANDE : {query}\n\nCADRAGE PM :\n"
        f"- Objectif : {pm_data.get('objective','')}\n"
        f"- Étapes : {', '.join(s.get('action','') for s in pm_data.get('action_plan',[]))}\n"
        f"- Risques : {', '.join(pm_data.get('key_risks',[]))}\n\n"
        f"AVIS EXPERTS :\n{proposals_text}\n\n"
        "Réponds avec :\n1. Diagnostic consolidé\n2. Décisions / arbitrages\n"
        "3. Plan d'action final priorisé\n4. Risques et mitigations\n5. Inputs minimum pour coder"
    )
    try:
        result = call_llm_with_retry(
            [{"role":"user","content":synthesis_prompt}], model=model,
            max_tokens=TOKENS_SYNTHESIS, step_name="Synthèse",
        )
        return result, (time.time() - t0) * 1000
    except Exception as exc:
        logger.warning("Synthèse fallback : %s", exc)
        return ("1. Diagnostic consolidé\nBesoin traité.\n\n2. Décisions\nSimplifier.\n\n"
                "3. Plan d'action\n- Générer le code\n- Valider complétude\n\n"
                "4. Risques\n- Troncature => continuation\n\n5. Inputs\nStack cible, exemples."), (time.time()-t0)*1000


def _build_programmer_prompt(query: str, synthesis: str, lang_rule: str,
                              advocate_critique: str | None = None, stack: str | None = None) -> str:
    stack_hint = f"\nSTACK CIBLE : {stack}" if stack else ""
    advocate_section = (
        f"\nPOINTS DE VIGILANCE (Avocat du Diable) :\n{_safe_excerpt(advocate_critique, 2000)}\n"
        if advocate_critique else ""
    )
    return (
        f"{lang_rule}\nTu es un Développeur Expert Senior Full-Stack.\n"
        f"{stack_hint}\n\nDEMANDE ORIGINALE : {query}\n\nPLAN FINAL :\n{synthesis}\n"
        f"{advocate_section}\n"
        "RÈGLES DE SORTIE — ABSOLUMENT OBLIGATOIRES :\n"
        "1. INTERDIT : toute phrase narrative.\n"
        "2. Commence DIRECTEMENT par le code ou par `### FILE:`\n"
        "3. Format multi-fichiers : ### FILE: nom.ext\n   ```lang\n   ...\n   ```\n"
        "4. Ne coupe jamais une fonction ou un fichier.\n"
        f"5. Termine impérativement par : {FINAL_CODE_END_MARKER}\n"
        "6. AUCUN texte après le marqueur final.\n"
    )


def _complete_generation_if_needed(initial_output: str, original_prompt: str, model: str, max_rounds: int = 5) -> str:
    output = initial_output
    if not _is_likely_truncated(output):
        return output
    logger.info("Génération tronquée — lancement de %d round(s) de continuation.", max_rounds)
    for round_idx in range(1, max_rounds + 1):
        continuation_prompt = (
            "Tu dois TERMINER une génération de code potentiellement tronquée.\n"
            "Règles :\n1) Ne répète JAMAIS le début déjà généré.\n"
            "2) Reprends EXACTEMENT à partir de la dernière ligne utile.\n"
            "3) Ferme tous les blocs ouverts.\n"
            f"4) Termine par : {FINAL_CODE_END_MARKER}\n"
            f"DEMANDE : {original_prompt[:3_000]}\n\nSORTIE DÉJÀ GÉNÉRÉE (fin) :\n{output[-30_000:]}\n\n"
            "Donne UNIQUEMENT la suite manquante."
        )
        try:
            suffix = call_llm_with_retry(
                [{"role":"user","content":continuation_prompt}], model=model,
                max_tokens=TOKENS_CONTINUATION, step_name=f"Continuation-{round_idx}",
            )
            output = output.rstrip() + "\n" + suffix.lstrip()
            if not _is_likely_truncated(output):
                logger.info("Continuation terminée au round %d.", round_idx)
                return output
        except Exception as exc:
            logger.warning("Continuation round %d erreur : %s", round_idx, exc)
            continue
    return output


def _run_code_review(query: str, programmer_output: str, model: str, lang_rule: str) -> tuple[str, list[dict]]:
    review_prompt = (
        f"{lang_rule}\nTu es un Expert Reviewer de code Senior.\n"
        "Si tout est correct : LGTM\n"
        'Sinon : {"corrections": [{"file": "...", "line_hint": "...", "issue": "...", "fix": "..."}], "summary": "..."}\n\n'
        f"DEMANDE : {query}\n\nCODE :\n{_safe_excerpt(programmer_output, 30000)}"
    )
    try:
        raw = call_llm_with_retry(
            [{"role":"user","content":review_prompt}], model=model,
            max_tokens=TOKENS_REVIEWER, step_name="Code-Reviewer",
        )
        if "LGTM" in raw.upper() and "{" not in raw:
            return raw, []
        data = _parse_json_llm(raw)
        return data.get("summary", raw), data.get("corrections", [])
    except Exception as exc:
        logger.warning("Code reviewer erreur : %s", exc)
        return "Review indisponible.", []


def _apply_corrections_if_needed(programmer_output: str, corrections: list[dict], query: str,
                                  synthesis: str, model: str, lang_rule: str,
                                  advocate_critique: str | None, stack: str | None) -> str:
    if not corrections:
        return programmer_output
    corrections_text = "\n".join(
        f"- [{c.get('file','?')}] {c.get('issue','')} => {c.get('fix','')}" for c in corrections
    )
    corrected_prompt = (
        _build_programmer_prompt(query, synthesis, lang_rule, advocate_critique, stack)
        + f"\n\nCORRECTIONS DU REVIEWER :\n{corrections_text}\nApplique TOUTES ces corrections."
    )
    try:
        corrected = call_llm_with_retry(
            [{"role":"user","content":corrected_prompt}], model=model,
            max_tokens=TOKENS_PROGRAMMER, step_name="Programmeur-post-review",
        )
        return _complete_generation_if_needed(corrected, corrected_prompt, model=model)
    except Exception as exc:
        logger.warning("Programmeur post-review erreur : %s", exc)
        return programmer_output


def _generate_tests(query: str, files: list[dict], model: str, lang_rule: str) -> str:
    if not files:
        return ""
    code_context = "\n\n".join(
        f"### FILE: {f['filename']}\n```{f['language']}\n{_safe_excerpt(f['content'],6000)}\n```"
        for f in files
    )
    test_prompt = (
        f"{lang_rule}\nTu es un Expert QA Senior. Génère des tests unitaires complets.\n"
        "- Utilise le framework de test standard du langage.\n"
        "- Couvre cas nominaux, limites, erreurs.\n"
        f"- Commence par `### FILE: test_xxx.ext` — zéro introduction.\n"
        f"- Termine par : {FINAL_CODE_END_MARKER}\n\n"
        f"DEMANDE : {query}\n\nCODE :\n{code_context}"
    )
    try:
        return call_llm_with_retry(
            [{"role":"user","content":test_prompt}], model=model,
            max_tokens=TOKENS_TESTS, step_name="Génération-Tests",
        )
    except Exception as exc:
        logger.warning("Génération tests erreur : %s", exc)
        return ""


def _build_executive_summary(query: str, synthesis: str, programmer_output: str, model: str, lang_rule: str) -> str:
    prompt = (
        f"{lang_rule}\nTu es un assistant de direction. Rédige un résumé exécutif en 3 points max.\n"
        "Niveau manager : sans jargon, orienté résultat. Format : 3 bullet points courts.\n\n"
        f"DEMANDE : {query}\n\nSYNTHÈSE : {_safe_excerpt(synthesis, 2000)}"
    )
    try:
        return call_llm_with_retry(
            [{"role":"user","content":prompt}], model=model,
            max_tokens=TOKENS_SUMMARY, step_name="Résumé-Exécutif",
        )
    except Exception as exc:
        logger.warning("Résumé exécutif erreur : %s", exc)
        return ""


def _run_pipeline(params: dict) -> dict:
    timings: dict[str, float] = {}
    started_at = time.time()
    query          = params["query"]
    model          = params["model"]
    lang_rule      = params["lang_rule"]
    num_experts    = params["num_experts"]
    selected_experts = params["selected_experts"]
    debate_rounds  = params["debate_rounds"]
    advocate_enabled = params["advocate"]
    reviewer_enabled = params["reviewer"]
    generate_tests = params["generate_tests"]
    exec_summary   = params["executive_summary"]
    tokens_agent   = params["tokens_agent"]
    tokens_programmer = params["tokens_programmer"]
    stack          = params["stack"]
    session_id     = params.get("session_id")

    session_context = ""
    if session_id and session_id in _session_store and _session_store[session_id]:
        prev = _session_store[session_id][-1]
        session_context = (
            f"\nCONTEXTE SESSION PRÉCÉDENTE :\n"
            f"- Demande précédente : {prev.get('query','')[:300]}\n"
            f"- Synthèse résumée : {_safe_excerpt(prev.get('synthesis',''), 800)}\n"
        )
    query_with_context = query + session_context if session_context else query

    pm_data, timings["pm_ms"] = _build_pm_data(query_with_context, model, lang_rule)
    t0 = time.time()
    experts, strategy = _select_experts(query_with_context, model, lang_rule, selected_experts, num_experts)
    timings["orchestrator_ms"] = (time.time() - t0) * 1000
    t0 = time.time()
    proposals = _run_parallel_experts(query_with_context, model, lang_rule, experts, tokens_agent)
    timings["experts_ms"] = (time.time() - t0) * 1000
    if debate_rounds > 0:
        t0 = time.time()
        proposals = _run_expert_debate(query_with_context, model, lang_rule, proposals, debate_rounds)
        timings["debate_ms"] = (time.time() - t0) * 1000
    synthesis, timings["synthesis_ms"] = _build_synthesis(query_with_context, pm_data, proposals, model, lang_rule)
    advocate_critique: str | None = None
    if advocate_enabled:
        t0 = time.time()
        advocate_critique = _run_advocate(query, synthesis, model, lang_rule)
        timings["advocate_ms"] = (time.time() - t0) * 1000
    programmer_prompt = _build_programmer_prompt(query, synthesis, lang_rule, advocate_critique, stack)
    t0 = time.time()
    try:
        programmer_output = call_llm_with_retry(
            [{"role":"user","content":programmer_prompt}], model=model,
            max_tokens=tokens_programmer, step_name="Programmeur",
        )
        programmer_output = _complete_generation_if_needed(programmer_output, programmer_prompt, model=model)
    except Exception as exc:
        programmer_output = f"Erreur lors de la génération du code : {exc}"
    timings["programmer_ms"] = (time.time() - t0) * 1000
    review_text = ""
    corrections  = []
    if reviewer_enabled:
        t0 = time.time()
        review_text, corrections = _run_code_review(query, programmer_output, model, lang_rule)
        timings["reviewer_ms"] = (time.time() - t0) * 1000
        if corrections:
            t0 = time.time()
            programmer_output = _apply_corrections_if_needed(
                programmer_output, corrections, query, synthesis, model, lang_rule, advocate_critique, stack)
            timings["programmer_post_review_ms"] = (time.time() - t0) * 1000
    files       = _extract_files(programmer_output)
    clean_output = _clean_programmer_output(programmer_output)
    tests_output  = ""
    test_files: list[dict] = []
    if generate_tests and files:
        t0 = time.time()
        tests_output = _generate_tests(query, files, model, lang_rule)
        test_files   = _extract_files(tests_output)
        timings["tests_ms"] = (time.time() - t0) * 1000
    executive_summary_text = ""
    if exec_summary:
        t0 = time.time()
        executive_summary_text = _build_executive_summary(query, synthesis, programmer_output, model, lang_rule)
        timings["executive_summary_ms"] = (time.time() - t0) * 1000
    complete       = FINAL_CODE_END_MARKER in programmer_output
    total_chars    = sum(len(f["content"]) for f in files)
    experts_ok     = sum(1 for p in proposals if not p.get("error"))
    quality_score  = round(
        (0.4*(1 if complete else 0)) + (0.3*experts_ok/max(len(proposals),1)) + (0.3*(1 if files else 0)), 2)
    timings["total_ms"] = int((time.time() - started_at) * 1000)
    result = {
        "pm": pm_data, "strategy": strategy, "experts": experts,
        "proposals": [{"expert":p["expert"],"text":p["proposal"],"error":p.get("error"),
                       "debated":p.get("debated",False),"duration_ms":p.get("duration_ms",0)} for p in proposals],
        "synthesis": synthesis, "advocate_critique": advocate_critique,
        "programmer_output": clean_output, "files": files,
        "review_comments": review_text, "review_corrections": corrections,
        "tests_output": _clean_programmer_output(tests_output) if tests_output else "",
        "test_files": test_files, "executive_summary": executive_summary_text,
        "meta": {
            "query": query, "model": model, "stack_detected": stack,
            "num_experts": len(experts), "debate_rounds": debate_rounds,
            "advocate_enabled": advocate_enabled, "reviewer_enabled": reviewer_enabled,
            "tests_generated": bool(test_files), "programmer_complete": complete,
            "total_files": len(files), "total_code_chars": total_chars,
            "quality_score": quality_score, "timings_ms": {k: int(v) for k, v in timings.items()},
        },
    }
    if session_id:
        if session_id not in _session_store:
            _session_store[session_id] = []
        _session_store[session_id].append({"query":query,"synthesis":synthesis,"timestamp":time.time()})
        _session_store[session_id] = _session_store[session_id][-10:]
    return result


def _parse_warroom_payload(data: dict) -> tuple[dict | None, str | None]:
    query = data.get("query", "").strip()
    if not query:
        return None, "Requête vide"
    if len(query) > MAX_QUERY_LENGTH:
        return None, f"Requête trop longue (max {MAX_QUERY_LENGTH} caractères)"
    model = data.get("model", DEFAULT_MODEL)
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL
    lang      = data.get("lang", "fr")
    lang_rule = LANG_RULES_SHORT.get(lang, LANG_RULES_SHORT["fr"])
    selected_experts_raw = data.get("selected_experts", [])
    if not isinstance(selected_experts_raw, list):
        selected_experts_raw = []
    selected_experts = [e for e in selected_experts_raw if e in EXPERT_PROFILES]
    mode   = data.get("mode", "balanced")
    preset = MODE_PRESETS.get(mode, MODE_PRESETS["balanced"])
    params: dict = {
        "query": query, "model": model, "lang_rule": lang_rule,
        "selected_experts": selected_experts,
        "num_experts":      int(data.get("num_experts",      preset["num_experts"])),
        "debate_rounds":    int(data.get("debate_rounds",    preset["debate_rounds"])),
        "advocate":        bool(data.get("advocate",         preset["advocate"])),
        "reviewer":        bool(data.get("reviewer",         preset["reviewer"])),
        "generate_tests":  bool(data.get("generate_tests",   preset["generate_tests"])),
        "executive_summary":bool(data.get("executive_summary",preset["executive_summary"])),
        "tokens_agent":     int(data.get("tokens_agent",     preset["tokens_agent"])),
        "tokens_programmer":int(data.get("tokens_programmer",preset["tokens_programmer"])),
        "stack":   _detect_stack(query, data.get("stack")),
        "session_id":  data.get("session_id"),
        "webhook_url": data.get("webhook_url"),
    }
    params["num_experts"] = max(1, min(params["num_experts"], 6))
    return params, None


# ===========================================================================
# BLUEPRINTS
# ===========================================================================

auth_bp    = Blueprint("auth",    __name__)
chat_bp    = Blueprint("chat",    __name__)
warroom_bp = Blueprint("warroom", __name__)

# ── AUTH ────────────────────────────────────────────────────────────────────

@auth_bp.route("/login")
def login_page():
    if flask_session.get("username"):
        return redirect("/")
    return render_template_string(LOGIN_TEMPLATE)


@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    data     = request.get_json(force=True)
    username = data.get("username", "").strip().lower()
    password = data.get("password", "").strip()
    email    = data.get("email",    "").strip()
    if not username or not password:
        return jsonify({"error": "Nom d'utilisateur et mot de passe requis"}), 400
    if len(username) < 3:
        return jsonify({"error": "Le nom d'utilisateur doit faire au moins 3 caractères"}), 400
    if len(password) < 6:
        return jsonify({"error": "Le mot de passe doit faire au moins 6 caractères"}), 400
    with closing(get_db()) as db:
        if db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            return jsonify({"error": "Ce nom d'utilisateur est déjà pris"}), 409
        db.execute(
            "INSERT INTO users (username, password_hash, email, created_at) VALUES (?, ?, ?, ?)",
            (username, generate_password_hash(password), email, time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        db.commit()
    flask_session["username"] = username
    logger.info("New user registered: %s", username)
    return jsonify({"ok": True, "username": username})


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json(force=True)
    username = data.get("username", "").strip().lower()
    password = data.get("password", "").strip()
    if not username or not password:
        return jsonify({"error": "Identifiants manquants"}), 400
    with closing(get_db()) as db:
        row = db.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Identifiants incorrects"}), 401
    flask_session["username"] = username
    logger.info("User logged in: %s", username)
    return jsonify({"ok": True, "username": username})


@auth_bp.route("/api/auth/logout", methods=["POST"])
def logout():
    flask_session.clear()
    return jsonify({"ok": True})


@auth_bp.route("/api/auth/me", methods=["GET"])
def me():
    username = flask_session.get("username")
    if not username:
        return jsonify({"logged_in": False})
    with closing(get_db()) as db:
        row = db.execute(
            "SELECT username, email, created_at FROM users WHERE username = ?", (username,)
        ).fetchone()
    if not row:
        flask_session.clear()
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in":  True,
        "username":   row["username"],
        "email":      row["email"]      or "",
        "created_at": row["created_at"] or "",
    })


# ── CHAT ────────────────────────────────────────────────────────────────────

@chat_bp.route("/")
def index() -> str:
    return render_template_string(
        _load_html_template(), models=FREE_MODELS, default_model=DEFAULT_MODEL
    )


@chat_bp.route("/api/context", methods=["POST"])
def set_context():
    data     = request.get_json(force=True)
    username = flask_session.get("username", "__anon__")
    sess     = get_session()
    if "system_prompt" in data:
        sess["system_prompt"] = data["system_prompt"].strip()
    if "model" in data:
        sess["model"] = data["model"]
    if "skills" in data:
        sess["skills"] = [s for s in data["skills"] if isinstance(s, str)]
    if "lang" in data:
        sess["lang"] = data["lang"]
    if username != "__anon__":
        save_session_to_db(username, sess)
    return jsonify({"ok": True})


@chat_bp.route("/api/skills", methods=["POST"])
def set_skills():
    data     = request.get_json(force=True)
    username = flask_session.get("username", "__anon__")
    sess     = get_session()
    sess["skills"] = [s for s in data.get("skills", []) if isinstance(s, str)]
    if username != "__anon__":
        save_session_to_db(username, sess)
    return jsonify({"ok": True, "skills": sess["skills"]})


@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    username = flask_session.get("username", "__anon__")
    sess     = get_session()
    if request.is_json:
        data     = request.get_json()
        user_msg = data.get("message", "").strip()
    else:
        data     = None
        user_msg = request.form.get("message", "").strip()
    file_parts: list[str] = []
    for f in request.files.getlist("files"):
        if f.filename:
            try:
                content = f.read().decode("utf-8", errors="replace")
                file_parts.append(f"--- Fichier joint : {f.filename} ---\n{content}")
            except Exception as exc:
                logger.warning("Erreur lecture fichier %s : %s", f.filename, exc)
    full_msg = user_msg
    if file_parts:
        full_msg += "\n\n" + "\n\n".join(file_parts)
    if not full_msg:
        return jsonify({"error": "Message vide"}), 400
    lang = (data.get("lang") if data else None) or request.form.get("lang")
    if lang:
        sess["lang"] = lang
        if username != "__anon__":
            save_session_to_db(username, sess)
    selected_lang = sess.get("lang", "fr")
    lang_rule     = LANG_INSTRUCTIONS.get(selected_lang, LANG_INSTRUCTIONS["fr"])
    sys_content   = build_system_content(sess)
    tool_block    = (
        "\n\nOUTILS DISPONIBLES:\n"
        "Tu as accès à un outil de recherche internet. Pour l'utiliser, "
        "tape exactement: \\recherche <ta question>\n"
        "Utilise cet outil pour les questions factuelles ou à jour."
    )
    lang_block    = f"\n\nRÈGLE ABSOLUE DE LANGUE: {lang_rule}"
    full_sys      = (sys_content + tool_block + lang_block) if sys_content else (tool_block.strip() + lang_block)
    sess["messages"].append({"role": "user", "content": full_msg})
    payload_messages = [{"role": "system", "content": full_sys}]
    payload_messages.extend(sess["messages"])
    max_search_attempts = 5
    search_count        = 0
    final_reply         = None
    while search_count < max_search_attempts:
        payload = {"model": sess["model"], "messages": payload_messages, "max_tokens": 80000}
        try:
            response = http_requests.post(OPENROUTER_URL, headers=_get_headers(), json=payload, timeout=90)
            result   = response.json()
        except http_requests.RequestException as exc:
            logger.error("OpenRouter error: %s", exc)
            sess["messages"].pop()
            return jsonify({"error": "Problème de connexion au service AI"}), 502
        if "error" in result:
            err = result["error"]
            sess["messages"].pop()
            return jsonify({"error": f"IA indisponible ({err.get('code','?')}): {err.get('message','')}"}), 502
        reply: str = result["choices"][0]["message"]["content"]
        if reply.strip().startswith("\\recherche "):
            search_count += 1
            query = reply.strip()[len("\\recherche "):].strip()
            if not query:
                final_reply = reply
                payload_messages.append({"role": "assistant", "content": reply})
                break
            search_results = perform_search(query)
            payload_messages.append({"role": "assistant", "content": reply})
            payload_messages.append({
                "role": "system",
                "content": (f"RÉSULTATS DE RECHERCHE pour « {query} »:\n{search_results}\n\n"
                            "Rédige maintenant ta réponse finale basée sur ces résultats."),
            })
        else:
            final_reply = reply
            payload_messages.append({"role": "assistant", "content": reply})
            break
    if final_reply is None:
        return jsonify({"reply": "Limite de recherches atteinte. Réessaie."})
    sess["messages"] = [m for m in payload_messages if m["role"] in ("user", "assistant")]
    if username != "__anon__":
        save_session_to_db(username, sess)
    return jsonify({
        "reply":            final_reply,
        "total_messages":   len(sess["messages"]),
        "estimated_tokens": estimate_tokens(sess["messages"]),
    })


@chat_bp.route("/api/clear", methods=["POST"])
def clear():
    username = flask_session.get("username", "__anon__")
    sess     = get_session()
    sess["messages"] = []
    if username != "__anon__":
        save_session_to_db(username, sess)
    return jsonify({"ok": True})


@chat_bp.route("/api/history", methods=["GET"])
def history():
    sess = get_session()
    return jsonify({
        "messages":         sess["messages"],
        "system_prompt":    sess["system_prompt"],
        "model":            sess["model"],
        "skills":           sess.get("skills", []),
        "lang":             sess.get("lang", "fr"),
        "estimated_tokens": estimate_tokens(sess["messages"]),
    })


# ── WAR ROOM ────────────────────────────────────────────────────────────────

@warroom_bp.route("/api/warroom", methods=["POST"])
def warroom():
    ip = request.remote_addr or "unknown"
    if not _rate_check(ip):
        return jsonify({"error": "Rate limit dépassé."}), 429
    data = request.get_json(force=True, silent=True) or {}
    params, err = _parse_warroom_payload(data)
    if err:
        return jsonify({"error": err}), 400
    cache_key = _cache_key(params["query"], params["model"], params["selected_experts"])
    cached    = _cache_get(cache_key)
    if cached and not data.get("no_cache"):
        cached["meta"]["cache_hit"] = True
        return jsonify(cached)
    if params.get("webhook_url"):
        job_id = str(uuid.uuid4())
        _job_store[job_id] = {"status": "pending", "result": None, "created_at": time.time()}
        def _async_run():
            try:
                result = _run_pipeline(params)
                _job_store[job_id]["status"] = "done"
                _job_store[job_id]["result"] = result
                _cache_set(cache_key, result)
                try:
                    http_requests.post(params["webhook_url"], json={"job_id":job_id,"result":result}, timeout=15)
                except Exception as exc:
                    logger.warning("Webhook delivery failed : %s", exc)
            except Exception as exc:
                _job_store[job_id]["status"] = "error"
                _job_store[job_id]["error"]  = str(exc)
        threading.Thread(target=_async_run, daemon=True).start()
        return jsonify({"job_id": job_id, "status": "pending"}), 202
    result = _run_pipeline(params)
    result["meta"]["cache_hit"] = False
    _cache_set(cache_key, result)
    return jsonify(result)


@warroom_bp.route("/api/warroom/stream", methods=["POST"])
def warroom_stream():
    ip = request.remote_addr or "unknown"
    if not _rate_check(ip):
        def _err():
            yield "data: " + json.dumps({"event":"error","message":"Rate limit dépassé."}) + "\n\n"
        return Response(stream_with_context(_err()), mimetype="text/event-stream")
    data = request.get_json(force=True, silent=True) or {}
    params, err = _parse_warroom_payload(data)
    if err:
        def _err():
            yield "data: " + json.dumps({"event":"error","message":err}) + "\n\n"
        return Response(stream_with_context(_err()), mimetype="text/event-stream")

    def _sse(event: str, payload: dict) -> str:
        return "data: " + json.dumps({"event": event, **payload}) + "\n\n"

    @stream_with_context
    def _generate() -> Generator[str, None, None]:
        query     = params["query"]
        model     = params["model"]
        lang_rule = params["lang_rule"]
        stack     = params["stack"]
        pm_data, _ = _build_pm_data(query, model, lang_rule)
        yield _sse("pm_ready", {"pm": pm_data})
        experts, strategy = _select_experts(query, model, lang_rule, params["selected_experts"], params["num_experts"])
        proposals = _run_parallel_experts(query, model, lang_rule, experts, params["tokens_agent"])
        yield _sse("experts_ready", {
            "experts": experts, "strategy": strategy,
            "proposals": [{"expert":p["expert"],"text":p["proposal"],"error":p.get("error")} for p in proposals],
        })
        if params["debate_rounds"] > 0:
            proposals = _run_expert_debate(query, model, lang_rule, proposals, params["debate_rounds"])
            yield _sse("debate_ready", {
                "proposals": [{"expert":p["expert"],"text":p["proposal"]} for p in proposals],
            })
        synthesis, _ = _build_synthesis(query, pm_data, proposals, model, lang_rule)
        yield _sse("synthesis_ready", {"synthesis": synthesis})
        advocate_critique = None
        if params["advocate"]:
            advocate_critique = _run_advocate(query, synthesis, model, lang_rule)
            yield _sse("advocate_ready", {"advocate_critique": advocate_critique})
        programmer_prompt = _build_programmer_prompt(query, synthesis, lang_rule, advocate_critique, stack)
        try:
            programmer_output = call_llm_with_retry(
                [{"role":"user","content":programmer_prompt}], model=model,
                max_tokens=params["tokens_programmer"], step_name="Programmeur-stream",
            )
            programmer_output = _complete_generation_if_needed(programmer_output, programmer_prompt, model=model)
        except Exception as exc:
            programmer_output = f"Erreur : {exc}"
        files = _extract_files(programmer_output)
        yield _sse("code_ready", {
            "programmer_output": _clean_programmer_output(programmer_output),
            "files": files,
            "complete": FINAL_CODE_END_MARKER in programmer_output,
        })
        review_text = ""
        if params["reviewer"]:
            review_text, corrections = _run_code_review(query, programmer_output, model, lang_rule)
            if corrections:
                programmer_output = _apply_corrections_if_needed(
                    programmer_output, corrections, query, synthesis, model, lang_rule, advocate_critique, stack)
                files = _extract_files(programmer_output)
            yield _sse("review_ready", {
                "review_comments": review_text,
                "programmer_output": _clean_programmer_output(programmer_output),
                "files": files,
            })
        tests_output = ""
        if params["generate_tests"] and files:
            tests_output = _generate_tests(query, files, model, lang_rule)
            yield _sse("tests_ready", {
                "tests_output": _clean_programmer_output(tests_output),
                "test_files":   _extract_files(tests_output),
            })
        exec_sum = ""
        if params["executive_summary"]:
            exec_sum = _build_executive_summary(query, synthesis, programmer_output, model, lang_rule)
        yield _sse("done", {
            "executive_summary": exec_sum,
            "stack_detected":    stack,
            "complete":          FINAL_CODE_END_MARKER in programmer_output,
        })

    return Response(_generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@warroom_bp.route("/api/warroom/refine", methods=["POST"])
def warroom_refine():
    ip = request.remote_addr or "unknown"
    if not _rate_check(ip):
        return jsonify({"error": "Rate limit dépassé."}), 429
    data        = request.get_json(force=True, silent=True) or {}
    refinement  = data.get("refinement", "").strip()
    synthesis   = data.get("synthesis",  "").strip()
    query       = data.get("query",      "").strip()
    model       = data.get("model",      DEFAULT_MODEL)
    lang        = data.get("lang",       "fr")
    if not refinement or not synthesis or not query:
        return jsonify({"error": "Champs requis : query, synthesis, refinement"}), 400
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL
    lang_rule = LANG_RULES_SHORT.get(lang, LANG_RULES_SHORT["fr"])
    stack     = _detect_stack(query, data.get("stack"))
    refined_synthesis = synthesis + f"\n\nCONSIGNE DE RAFFINAGE :\n{refinement}"
    programmer_prompt = _build_programmer_prompt(query, refined_synthesis, lang_rule, stack=stack)
    t0 = time.time()
    try:
        programmer_output = call_llm_with_retry(
            [{"role":"user","content":programmer_prompt}], model=model,
            max_tokens=int(data.get("tokens_programmer", TOKENS_PROGRAMMER)),
            step_name="Programmeur-refine",
        )
        programmer_output = _complete_generation_if_needed(programmer_output, programmer_prompt, model=model)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502
    files = _extract_files(programmer_output)
    return jsonify({
        "programmer_output": _clean_programmer_output(programmer_output),
        "files":    files,
        "complete": FINAL_CODE_END_MARKER in programmer_output,
        "meta": {
            "refinement":   refinement,
            "duration_ms":  int((time.time() - t0) * 1000),
            "total_files":  len(files),
        },
    })


@warroom_bp.route("/api/warroom/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    job = _job_store.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable"}), 404
    response = {"job_id": job_id, "status": job["status"], "created_at": job["created_at"]}
    if job["status"] == "done":
        response["result"] = job["result"]
    elif job["status"] == "error":
        response["error"] = job.get("error", "Erreur inconnue")
    return jsonify(response)


@warroom_bp.route("/api/warroom/sessions/<session_id>", methods=["GET"])
def get_session_history(session_id: str):
    history = _session_store.get(session_id, [])
    return jsonify({"session_id": session_id, "count": len(history), "history": history})


@warroom_bp.route("/api/warroom/cache", methods=["DELETE"])
def clear_cache():
    count = len(_cache_store)
    _cache_store.clear()
    return jsonify({"cleared": count})


@warroom_bp.route("/api/experts", methods=["GET"])
def get_experts_route():
    return jsonify({
        "families": EXPERT_FAMILIES,
        "experts":  {k: {**v} for k, v in EXPERT_PROFILES.items()},
        "modes":    list(MODE_PRESETS.keys()),
    })


@warroom_bp.route("/api/advocate", methods=["POST"])
def advocate():
    ip = request.remote_addr or "unknown"
    if not _rate_check(ip):
        return jsonify({"error": "Rate limit dépassé."}), 429
    data  = request.get_json(force=True, silent=True) or {}
    topic = data.get("topic", "").strip()
    model = data.get("model", DEFAULT_MODEL)
    if not topic:
        return jsonify({"error": "Sujet vide"}), 400
    if len(topic) > MAX_QUERY_LENGTH:
        return jsonify({"error": f"Sujet trop long (max {MAX_QUERY_LENGTH} caractères)"}), 400
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL
    prompt = (
        "Tu es l'Avocat du Diable. Donne une critique structurée, concrète et concise.\n"
        "Structure :\n## Failles critiques\n## Risques majeurs\n## Points faibles\n"
        "## Hypothèses dangereuses\n## Scénarios catastrophe\n## Mitigations\n\n"
        f"Sujet : {topic}"
    )
    try:
        result = call_llm_with_retry(
            [{"role":"user","content":prompt}], model=model,
            max_tokens=3_000, step_name="Avocat-du-Diable",
        )
        return jsonify({"critique": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@warroom_bp.route("/api/warroom/feedback", methods=["POST"])
def submit_feedback():
    data    = request.get_json(force=True, silent=True) or {}
    score   = data.get("score")
    comment = data.get("comment", "")
    query   = data.get("query",   "")
    if score is None or not isinstance(score, (int, float)) or not (1 <= score <= 5):
        return jsonify({"error": "score requis entre 1 et 5"}), 400
    entry = {
        "score":     float(score),
        "comment":   str(comment)[:1000],
        "query":     str(query)[:500],
        "timestamp": time.time(),
        "ip":        request.remote_addr,
    }
    _feedback_store.append(entry)
    avg = sum(f["score"] for f in _feedback_store) / len(_feedback_store)
    return jsonify({"recorded": True, "avg_score": round(avg, 2), "total": len(_feedback_store)})


@warroom_bp.route("/api/warroom/feedback", methods=["GET"])
def get_feedback():
    if not _feedback_store:
        return jsonify({"avg_score": None, "total": 0, "recent": []})
    avg = sum(f["score"] for f in _feedback_store) / len(_feedback_store)
    return jsonify({"avg_score": round(avg, 2), "total": len(_feedback_store), "recent": _feedback_store[-10:]})


# ===========================================================================
# IDE PYTHON — Exécution sécurisée de code
# ===========================================================================

import subprocess
import sys
import tempfile

ide_bp = Blueprint("ide", __name__)

IDE_TIMEOUT_SECONDS = 15
IDE_MAX_CODE_LENGTH = 50_000
IDE_MAX_OUTPUT_LENGTH = 20_000

_IDE_BLOCKED_IMPORTS = [
    "os.system", "subprocess", "shutil.rmtree", "__import__('os').system",
    "eval(", "exec(", "open('/etc", "open('/proc", "open('/sys",
]


def _code_is_safe(code: str) -> tuple[bool, str]:
    """Vérifie basiquement que le code n'est pas dangereux."""
    lower = code.lower()
    danger_patterns = [
        ("import subprocess", "subprocess non autorisé"),
        ("import shutil", "shutil non autorisé (rm)"),
        ("os.system(", "os.system non autorisé"),
        ("os.popen(", "os.popen non autorisé"),
        ("__import__('os')", "__import__ os non autorisé"),
        ("open('/etc", "accès /etc non autorisé"),
        ("open('/proc", "accès /proc non autorisé"),
    ]
    for pattern, msg in danger_patterns:
        if pattern.lower() in lower:
            return False, msg
    return True, ""


@ide_bp.route("/api/execute", methods=["POST"])
def execute_code():
    """Exécute du code Python dans un subprocess isolé avec timeout."""
    data = request.get_json(force=True, silent=True) or {}
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "Code vide"}), 400
    if len(code) > IDE_MAX_CODE_LENGTH:
        return jsonify({"error": f"Code trop long (max {IDE_MAX_CODE_LENGTH} chars)"}), 400

    safe, reason = _code_is_safe(code)
    if not safe:
        return jsonify({"error": f"Code non autorisé : {reason}"}), 403

    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", encoding="utf-8", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True,
            timeout=IDE_TIMEOUT_SECONDS,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        duration_ms = int((time.time() - t0) * 1000)

        stdout = proc.stdout[:IDE_MAX_OUTPUT_LENGTH] if proc.stdout else ""
        stderr = proc.stderr[:IDE_MAX_OUTPUT_LENGTH] if proc.stderr else ""

        return jsonify({
            "stdout":      stdout,
            "stderr":      stderr,
            "returncode":  proc.returncode,
            "duration_ms": duration_ms,
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": f"Timeout : le code a dépassé {IDE_TIMEOUT_SECONDS}s d'exécution"}), 408
    except Exception as exc:
        logger.error("Erreur exécution IDE : %s", exc)
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            import os as _os
            _os.unlink(tmp_path)
        except Exception:
            pass


@ide_bp.route("/api/ide/ask", methods=["POST"])
def ide_ask():
    """Endpoint IA pour l'assistant IDE Python."""
    data = request.get_json(force=True, silent=True) or {}
    messages = data.get("messages", [])
    model    = data.get("model", DEFAULT_MODEL)
    if not messages:
        return jsonify({"error": "Messages vides"}), 400
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL
    try:
        reply = call_llm_with_retry(messages, model=model, max_tokens=4000, step_name="IDE-AI")
        return jsonify({"reply": reply})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


# ===========================================================================
# APPLICATION
# ===========================================================================

def create_app() -> Flask:
    application = Flask(__name__)
    application.config["MAX_CONTENT_LENGTH"]   = 50 * 1024 * 1024
    application.config["MAX_FORM_MEMORY_SIZE"] = 50 * 1024 * 1024
    application.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production-use-os-urandom")
    application.register_blueprint(auth_bp)
    application.register_blueprint(chat_bp)
    application.register_blueprint(warroom_bp)
    application.register_blueprint(ide_bp)
    return application


app = create_app()

if __name__ == "__main__":
    logger.info("NeuralChat — starting")
    init_db()
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
