from __future__ import annotations

import ast
import logging
import re
import time
import uuid
from typing import Any
import os
import requests
from flask import Flask, jsonify, render_template_string, request, session
import asyncio
from os import system
import threading
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL  = "minimax/minimax-m2.5:free"

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

if not API_KEY:
    logger.warning("⚠️  OPENROUTER_API_KEY non définie — les appels LLM échoueront.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]   = 50 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 50 * 1024 * 1024
# SECRET_KEY stable requis pour que les sessions Flask persistent entre redémarrages
app.secret_key = os.environ.get("SECRET_KEY") or os.environ.get("FLASK_SECRET_KEY") or "dev-only-secret-change-in-prod-42x"
app.config["PERMANENT_SESSION_LIFETIME"] = 86400 * 30  # 30 jours

# store[flask_session_id] = { "active_id": uuid, "sessions": { uuid: conv_data } }
store: dict[str, dict[str, Any]] = {}

FREE_MODELS: list[dict[str, str]] = [
    {"id": "minimax/minimax-m2.5:free",              "label": "MiniMax M2.5"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B"},
    {"id": "mistralai/mistral-7b-instruct:free",     "label": "Mistral 7B"},
    {"id": "google/gemma-3-4b-it:free",              "label": "Gemma 3 4B"},
    {"id": "deepseek/deepseek-r1:free",              "label": "DeepSeek R1"},
    {"id": "qwen/qwen3.6-plus-preview:free",         "label": "Qwen3.6"},
]

PING_URL = "https://neuralclaw.onrender.com"  # 🔁 mets ton site ici
async def ping():
    system(f"ping -c 4 {PING_URL}")
    await asyncio.sleep(10)

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = [
    {"icon": "🐍", "label": "Dev Python",       "prompt": "Tu es un expert Python senior. Réponds toujours avec du code PEP8 correct, des type hints, des docstrings et des exemples concrets. Propose des alternatives quand c'est pertinent."},
    {"icon": "⚛️", "label": "Expert React",     "prompt": "Tu es un expert React et TypeScript. Utilise les hooks modernes, les bonnes pratiques de performance, et fournis toujours du code fonctionnel prêt à l'emploi."},
    {"icon": "📊", "label": "Analyste Data",    "prompt": "Tu es un analyste de données senior. Tu maîtrises pandas, numpy, matplotlib, SQL. Explique tes analyses étape par étape et justifie tes choix méthodologiques."},
    {"icon": "✍️", "label": "Rédacteur Tech",   "prompt": "Tu es un rédacteur technique expert. Tu produis des documentations claires, structurées et accessibles. Tu adaptes ton niveau au public cible et évites le jargon inutile."},
    {"icon": "🔧", "label": "Expert Linux",     "prompt": "Tu es un expert Bash/Linux. Fournis des commandes précises avec des explications. Signale toujours les risques potentiels et propose des alternatives sûres."},
    {"icon": "🗄️", "label": "Expert SQL",       "prompt": "Tu es un expert bases de données SQL. Tu optimises les requêtes, expliques les index, les jointures et les performances. Tu travailles avec PostgreSQL, MySQL et SQLite."},
    {"icon": "🏗️", "label": "Architecte",       "prompt": "Tu es un architecte logiciel senior. Tu proposes des architectures scalables, tu discutes des trade-offs et tu justifies tes décisions avec des arguments concrets."},
    {"icon": "🤖", "label": "Prompt Engineer",  "prompt": "Tu es un expert en prompt engineering. Tu aides à concevoir des prompts efficaces pour les LLMs, tu expliques les techniques (few-shot, chain-of-thought, etc.) avec des exemples."},
]

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Chatbot + CLAW v3</title>
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

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── SIDEBAR ── */
#sidebar {
  width: 280px; min-width: 280px;
  background: var(--panel);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  overflow: hidden;
  transition: transform 0.3s ease;
}
.sidebar-brand {
  padding: 18px 16px 12px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.sidebar-brand h1 {
  font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.sidebar-brand span { display: block; font-size: 10px; color: var(--muted); margin-top: 2px; -webkit-text-fill-color: var(--muted); }

/* Sessions */
.sessions-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px 6px; flex-shrink: 0;
}
.sessions-header .section-label { margin: 0; }
#new-session-btn {
  background: none; border: none; color: var(--accent);
  cursor: pointer; font-size: 18px; padding: 2px 4px;
  border-radius: 6px; transition: all .2s; line-height: 1;
}
#new-session-btn:hover { background: rgba(59,130,246,.15); }

#sessions-list {
  flex: 1; overflow-y: auto; padding: 0 8px;
  display: flex; flex-direction: column; gap: 2px;
}

.session-item {
  display: flex; align-items: center; gap: 8px;
  padding: 9px 10px; border-radius: 8px;
  cursor: pointer; transition: all .15s;
  border: 1px solid transparent;
  position: relative; group: true;
}
.session-item:hover { background: rgba(255,255,255,.04); }
.session-item.active { background: rgba(59,130,246,.12); border-color: rgba(59,130,246,.3); }
.session-icon { font-size: 14px; flex-shrink: 0; }
.session-name {
  flex: 1; font-size: 12px; color: var(--text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  cursor: pointer;
}
.session-item.active .session-name { color: var(--accent); }
.session-name-input {
  flex: 1; background: none; border: none; outline: none;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  color: var(--accent); padding: 0;
}
.session-actions {
  display: none; gap: 2px; flex-shrink: 0;
}
.session-item:hover .session-actions { display: flex; }
.session-action-btn {
  background: none; border: none; color: var(--muted);
  cursor: pointer; font-size: 12px; padding: 2px 5px;
  border-radius: 4px; transition: all .2s;
}
.session-action-btn:hover { color: var(--danger); background: rgba(239,68,68,.1); }
.session-action-btn.edit:hover { color: var(--accent); background: rgba(59,130,246,.1); }
.session-date { font-size: 10px; color: var(--muted); flex-shrink: 0; }

.sidebar-config {
  border-top: 1px solid var(--border);
  overflow-y: auto;
  padding: 14px 14px 10px;
  display: flex; flex-direction: column; gap: 14px;
  flex-shrink: 0;
  max-height: 55vh;
}

.section-label {
  font-size: 10px; font-weight: 600; letter-spacing: 1.5px;
  text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
}
.field-group { display: flex; flex-direction: column; gap: 6px; }

select, textarea, input[type="text"] {
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  padding: 9px 11px; width: 100%;
  transition: border-color 0.2s; outline: none; resize: none;
}
select:focus, textarea:focus, input:focus { border-color: var(--accent); }
#system-prompt { min-height: 80px; line-height: 1.6; }

/* Slider */
.slider-row { display: flex; align-items: center; gap: 10px; }
.slider-row input[type="range"] {
  flex: 1; background: none; border: none; padding: 0;
  accent-color: var(--accent3); cursor: pointer;
}
.slider-val {
  min-width: 32px; text-align: right;
  font-size: 12px; color: var(--accent2);
}

.btn {
  display: flex; align-items: center; justify-content: center; gap: 6px;
  padding: 9px 14px; border-radius: 8px; border: none;
  cursor: pointer; font-family: 'Syne', sans-serif; font-weight: 600;
  font-size: 12px; transition: all 0.2s;
}
.btn-primary   { background: var(--accent);  color: #fff; }
.btn-primary:hover   { background: #2563eb; }
.btn-secondary { background: var(--accent3); color: #fff; }
.btn-secondary:hover { background: #7c3aed; }
.btn-danger    { background: transparent; color: var(--danger); border: 1px solid var(--danger); }
.btn-danger:hover    { background: rgba(239,68,68,.1); }
.btn-ghost     { background: transparent; color: var(--muted); border: 1px solid var(--border); }
.btn-ghost:hover     { border-color: var(--accent); color: var(--accent); }
.btn-tpl       { background: rgba(139,92,246,.1); color: var(--accent3); border: 1px solid rgba(139,92,246,.3); }
.btn-tpl:hover { background: rgba(139,92,246,.2); }
.btn-full  { width: 100%; }
.btn:disabled { opacity: .4; cursor: not-allowed; }

#stats {
  font-size: 11px; color: var(--muted);
  padding: 10px 12px; background: var(--bg);
  border-radius: 8px; border: 1px solid var(--border); line-height: 2;
}
#stats span { color: var(--accent2); }

/* Skills */
.skills-list { display: flex; flex-direction: column; gap: 5px; }
.skill-tag {
  display: flex; align-items: center; justify-content: space-between;
  padding: 7px 10px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; font-size: 11px; animation: fadeIn .3s ease;
}
@keyframes fadeIn { from { opacity:0; transform: translateX(-6px); } to { opacity:1; } }
.skill-tag .skill-name { color: var(--accent2); font-weight: 500; }
.skill-tag .skill-remove {
  background: none; border: none; color: var(--muted);
  cursor: pointer; font-size: 13px; padding: 1px 5px;
  border-radius: 4px; transition: all .2s;
}
.skill-tag .skill-remove:hover { color: var(--danger); background: rgba(239,68,68,.1); }

/* Sidebar bottom actions */
.sidebar-actions {
  padding: 10px 14px 14px;
  border-top: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 6px;
  flex-shrink: 0;
}

/* ── MODALS ── */
.modal-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,.75);
  display: flex; align-items: center; justify-content: center;
  z-index: 1000; opacity: 0; visibility: hidden; transition: all .3s;
}
.modal-overlay.active { opacity: 1; visibility: visible; }
.modal {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 16px; padding: 24px;
  width: 90%; max-width: 480px;
  transform: scale(.9); transition: transform .3s;
}
.modal-overlay.active .modal { transform: scale(1); }
.modal h2 { font-family: 'Syne', sans-serif; font-size: 18px; margin-bottom: 16px; }
.modal-input {
  width: 100%; padding: 11px 13px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 8px; color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  margin-bottom: 16px; outline: none;
}
.modal-input:focus { border-color: var(--accent3); }
.modal-buttons { display: flex; gap: 10px; justify-content: flex-end; }

/* Templates modal */
.tpl-modal { max-width: 620px; }
.tpl-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 16px;
}
.tpl-card {
  padding: 14px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 10px;
  cursor: pointer; transition: all .2s;
}
.tpl-card:hover { border-color: var(--accent3); background: rgba(139,92,246,.05); transform: translateY(-1px); }
.tpl-card.selected { border-color: var(--accent3); background: rgba(139,92,246,.1); }
.tpl-icon { font-size: 20px; margin-bottom: 6px; }
.tpl-label { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; color: var(--text); }
.tpl-preview { font-size: 10px; color: var(--muted); margin-top: 4px; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

/* ── APP WRAPPER ── */
#app-wrapper { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

/* ── TOPBAR ── */
#topbar {
  padding: 10px 20px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px; flex-shrink: 0;
}
.nav-tabs { display: flex; gap: 5px; }
.nav-tab {
  padding: 7px 14px; border-radius: 8px; border: 1px solid var(--border);
  background: transparent; color: var(--muted);
  font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 600;
  cursor: pointer; transition: all .2s;
}
.nav-tab:hover { border-color: var(--accent); color: var(--accent); }
.nav-tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.nav-tab.claw-tab.active { background: var(--accent3); border-color: var(--accent3); }

#topbar-right { margin-left: auto; display: flex; align-items: center; gap: 8px; }

.icon-btn {
  width: 34px; height: 34px; border-radius: 8px; border: 1px solid var(--border);
  background: transparent; color: var(--muted);
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  font-size: 15px; transition: all .2s;
}
.icon-btn:hover { border-color: var(--accent); color: var(--accent); background: rgba(59,130,246,.08); }

#model-badge {
  font-size: 10px; padding: 4px 9px; border-radius: 20px;
  background: rgba(59,130,246,.12); color: var(--accent);
  border: 1px solid rgba(59,130,246,.25);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 160px;
}
#context-indicator { font-size: 11px; color: var(--success); display: flex; align-items: center; gap: 4px; }
#context-indicator::before { content: '●'; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
#skills-indicator { font-size: 11px; color: var(--accent3); display: none; align-items: center; gap: 4px; }
#sidebar-toggle { display: none; background: none; border: none; cursor: pointer; color: var(--text); padding: 4px; }

/* ── SEARCH BAR ── */
#search-bar {
  display: none; align-items: center; gap: 8px;
  padding: 8px 20px; border-bottom: 1px solid var(--border);
  background: rgba(59,130,246,.05); flex-shrink: 0;
}
#search-bar.visible { display: flex; }
#search-input {
  flex: 1; background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  padding: 7px 12px; outline: none; transition: border-color .2s;
}
#search-input:focus { border-color: var(--accent); }
#search-count { font-size: 11px; color: var(--muted); min-width: 80px; }
.search-nav-btn {
  background: none; border: 1px solid var(--border); color: var(--muted);
  border-radius: 6px; cursor: pointer; padding: 5px 9px; font-size: 12px;
  transition: all .2s;
}
.search-nav-btn:hover { border-color: var(--accent); color: var(--accent); }
#search-close { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 16px; padding: 4px; transition: color .2s; }
#search-close:hover { color: var(--danger); }

/* ── VIEWS ── */
.view-panel { flex: 1; display: none; flex-direction: column; overflow: hidden; }
.view-panel.active { display: flex; }

/* ── MESSAGES ── */
#messages {
  flex: 1; overflow-y: auto; padding: 20px;
  display: flex; flex-direction: column; gap: 16px; scroll-behavior: smooth;
}

.msg-wrapper { display: flex; gap: 10px; max-width: 87%; animation: slideIn .22s ease; }
@keyframes slideIn { from { opacity:0; transform: translateY(6px); } to { opacity:1; } }
.msg-wrapper.user { flex-direction: row-reverse; align-self: flex-end; }
.msg-wrapper.bot  { align-self: flex-start; }

.avatar {
  width: 32px; height: 32px; border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; flex-shrink: 0; margin-top: 2px;
}
.avatar.user { background: var(--user-bg); }
.avatar.bot  { background: linear-gradient(135deg, var(--accent), var(--accent2)); }

.bubble {
  padding: 12px 16px; border-radius: var(--radius);
  font-size: 13px; line-height: 1.7; max-width: 100%;
  position: relative;
}
.bubble.user { background: var(--user-bg); border-top-right-radius: 4px; color: #c7d9f8; }
.bubble.bot  { background: var(--bot-bg); border-top-left-radius: 4px; border: 1px solid var(--border); }
.bubble.bot p { margin-bottom: 8px; }
.bubble.bot p:last-child { margin-bottom: 0; }
.bubble.bot h1,.bubble.bot h2,.bubble.bot h3 { font-family: 'Syne', sans-serif; margin: 12px 0 6px; color: #fff; }
.bubble.bot ul,.bubble.bot ol { padding-left: 18px; margin: 6px 0; }
.bubble.bot li { margin-bottom: 3px; }
.bubble.bot code:not(pre code) { background: rgba(59,130,246,.15); color: var(--accent2); padding: 2px 5px; border-radius: 4px; font-size: 11px; }
.bubble.bot pre { background: #0d1117; border-radius: 8px; padding: 12px; overflow-x: auto; margin: 8px 0; border: 1px solid var(--border); }
.bubble.bot pre code { font-size: 11px; }
.bubble.bot blockquote { border-left: 3px solid var(--accent); padding-left: 10px; color: var(--muted); margin: 6px 0; }
.bubble.bot table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 11px; }
.bubble.bot th { background: rgba(59,130,246,.12); padding: 7px 10px; text-align: left; }
.bubble.bot td { padding: 6px 10px; border-bottom: 1px solid var(--border); }

/* Search highlight */
.bubble.search-match { outline: 2px solid rgba(245,158,11,.4); }
.bubble.search-current { outline: 2px solid var(--warn) !important; background-color: rgba(245,158,11,.05) !important; }
mark.search-hl { background: rgba(245,158,11,.35); color: inherit; border-radius: 2px; padding: 0 1px; }

/* Copy button on messages */
.bubble-copy {
  position: absolute; top: 8px; right: 8px;
  background: rgba(30,45,69,.9); border: 1px solid var(--border);
  color: var(--muted); border-radius: 6px;
  padding: 3px 8px; font-size: 10px; cursor: pointer;
  opacity: 0; transition: opacity .2s;
}
.msg-wrapper:hover .bubble-copy { opacity: 1; }
.bubble-copy:hover { color: var(--accent2); border-color: var(--accent2); }

.msg-time { font-size: 10px; color: var(--muted); margin-top: 4px; padding: 0 3px; }

/* ── TYPING ── */
#typing { display: none; align-self: flex-start; gap: 10px; align-items: center; }
#typing.visible { display: flex; }
.typing-dots {
  background: var(--bot-bg); border: 1px solid var(--border);
  border-radius: var(--radius); border-top-left-radius: 4px;
  padding: 12px 16px; display: flex; gap: 4px; align-items: center;
}
.typing-dots span { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: bounce 1.2s infinite; }
.typing-dots span:nth-child(2) { animation-delay: .2s; }
.typing-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }

/* ── INPUT AREA ── */
#input-area {
  padding: 14px 20px 18px; border-top: 1px solid var(--border);
  display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; flex-shrink: 0;
}
#file-attach-area { width: 100%; display: none; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
#file-attach-area.visible { display: flex; }
.file-chip {
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 7px; padding: 5px 9px; font-size: 11px;
  display: flex; align-items: center; gap: 5px; color: var(--accent2);
}
.file-chip .remove { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 13px; padding: 0 2px; transition: color .2s; }
.file-chip .remove:hover { color: var(--danger); }

#message-input {
  flex: 1; min-height: 48px; max-height: 150px; padding: 13px 15px;
  background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  resize: none; outline: none; transition: border-color .2s; line-height: 1.5;
}
#message-input:focus { border-color: var(--accent); }
#message-input::placeholder { color: var(--muted); }

.action-btn {
  width: 48px; height: 48px; border-radius: 11px; border: none;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  transition: transform .15s, opacity .15s; flex-shrink: 0;
}
#attach-btn { background: var(--bg); color: var(--muted); }
#attach-btn:hover { color: var(--accent); transform: scale(1.05); }
#send-btn { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
#send-btn:hover:not(:disabled) { transform: scale(1.05); }
#send-btn:disabled { opacity: .4; cursor: not-allowed; }
.action-btn svg { width: 18px; height: 18px; fill: currentColor; }

/* ── WELCOME ── */
#welcome {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 10px; color: var(--muted); text-align: center; padding: 40px;
}
#welcome h2 { font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800; color: var(--text); }
#welcome p  { font-size: 12px; line-height: 1.8; max-width: 380px; }
#welcome .hint { font-size: 10px; color: #334155; margin-top: 10px; }

/* ── CLAW VIEW ── */
#claw-view { padding: 20px; gap: 16px; overflow-y: auto; }

.claw-header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.claw-header h2 {
  font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 800;
  background: linear-gradient(135deg, var(--accent3), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.badge {
  font-size: 10px; padding: 3px 9px; border-radius: 20px;
  background: rgba(139,92,246,.15); color: var(--accent3);
  border: 1px solid rgba(139,92,246,.3); font-family: 'JetBrains Mono', monospace;
}

.claw-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .claw-grid { grid-template-columns: 1fr; } }

.claw-card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 18px;
  display: flex; flex-direction: column; gap: 10px;
}
.claw-card label {
  font-size: 10px; font-weight: 600; letter-spacing: 1px;
  text-transform: uppercase; color: var(--muted);
}
#claw-obj {
  flex: 1; min-height: 110px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  padding: 11px; resize: vertical; outline: none; transition: border-color .2s;
}
#claw-obj:focus { border-color: var(--accent3); }

#claw-file-zone {
  border: 2px dashed var(--border); border-radius: 10px;
  padding: 18px; text-align: center; font-size: 12px; color: var(--muted);
  cursor: pointer; transition: all .2s; position: relative;
}
#claw-file-zone:hover, #claw-file-zone.drag-over { border-color: var(--accent3); color: var(--accent3); background: rgba(139,92,246,.04); }
#claw-file-zone input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; }
.file-list { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.file-tag {
  background: rgba(139,92,246,.1); border: 1px solid rgba(139,92,246,.3);
  color: var(--accent3); border-radius: 6px; padding: 3px 9px; font-size: 10px;
  display: flex; align-items: center; gap: 5px;
}
.file-tag span { cursor: pointer; color: var(--muted); font-size: 13px; transition: color .2s; }
.file-tag span:hover { color: var(--danger); }

.claw-logs {
  flex: 1; min-height: 260px; max-height: 380px;
  background: #060a14; border: 1px solid var(--border);
  border-radius: 8px; padding: 12px; overflow-y: auto;
  font-size: 11.5px; line-height: 1.8; font-family: 'JetBrains Mono', monospace;
}
.log-line { padding: 1px 0; }
.log-info  { color: #7dd3fc; }
.log-ok    { color: var(--success); }
.log-err   { color: var(--danger); }
.log-warn  { color: var(--warn); }
.log-dim   { color: #334155; }
.log-phase { color: var(--accent3); font-weight: 600; }
.log-retry { color: var(--warn); font-style: italic; }

.claw-progress { height: 3px; border-radius: 2px; background: var(--border); overflow: hidden; }
.claw-progress-bar {
  height: 100%; border-radius: 2px;
  background: linear-gradient(90deg, var(--accent3), var(--accent2));
  width: 0%; transition: width .4s ease;
}

#claw-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }

.claw-output {
  background: var(--panel); border: 1px solid var(--success);
  border-radius: var(--radius); overflow: hidden; animation: fadeIn .4s ease;
}
.claw-output-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 11px 14px; border-bottom: 1px solid var(--border); font-size: 12px;
}
.claw-output-header span { color: var(--success); font-weight: 600; }
.claw-output-header-btns { display: flex; gap: 8px; }
.claw-output-meta {
  padding: 7px 14px; border-bottom: 1px solid var(--border);
  font-size: 10px; color: var(--muted); display: flex; gap: 14px; flex-wrap: wrap;
}
.claw-output-meta em { color: var(--accent2); font-style: normal; }
.code-action-btn {
  padding: 5px 12px; border-radius: 6px; border: 1px solid; font-size: 10px;
  cursor: pointer; transition: all .2s; font-family: 'JetBrains Mono', monospace;
}
.copy-btn  { background: rgba(34,197,94,.08);  border-color: rgba(34,197,94,.3);  color: var(--success); }
.copy-btn:hover  { background: rgba(34,197,94,.18); }
.dl-btn    { background: rgba(59,130,246,.08);  border-color: rgba(59,130,246,.3);  color: var(--accent); }
.dl-btn:hover    { background: rgba(59,130,246,.18); }
#claw-code {
  padding: 14px !important; margin: 0 !important;
  border-radius: 0 !important; font-size: 11px !important;
  max-height: 460px; overflow-y: auto;
}

/* ── TOAST ── */
#toast {
  position: fixed; bottom: 28px; right: 28px;
  background: var(--panel); border: 1px solid var(--border);
  color: var(--text); padding: 11px 18px; border-radius: 10px;
  font-size: 12px; transform: translateY(70px); opacity: 0;
  transition: all .3s; z-index: 9999; pointer-events: none;
}
#toast.show { transform: none; opacity: 1; }
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

<!-- ── SKILL MODAL ── -->
<div class="modal-overlay" id="skill-modal">
  <div class="modal">
    <h2>✨ Ajouter un Skill</h2>
    <input type="text" class="modal-input" id="skill-input" placeholder="Ex: Python, Rédaction, SQL…"/>
    <div class="modal-buttons">
      <button class="btn btn-ghost" onclick="closeModal('skill-modal')">Annuler</button>
      <button class="btn btn-secondary" onclick="addSkill()">Ajouter</button>
    </div>
  </div>
</div>

<!-- ── TEMPLATES MODAL ── -->
<div class="modal-overlay" id="tpl-modal">
  <div class="modal tpl-modal">
    <h2>🎭 Choisir un template</h2>
    <div class="tpl-grid" id="tpl-grid"></div>
    <div class="modal-buttons">
      <button class="btn btn-ghost" onclick="closeModal('tpl-modal')">Annuler</button>
      <button class="btn btn-secondary" onclick="applyTemplate()">Appliquer</button>
    </div>
  </div>
</div>

<!-- ── SIDEBAR ── -->
<aside id="sidebar">
  <div class="sidebar-brand">
    <h1>AI Chat + CLAW<span>v3 · OpenRouter</span></h1>
  </div>

  <!-- Sessions -->
  <div class="sessions-header">
    <div class="section-label">Conversations</div>
    <button id="new-session-btn" onclick="createSession()" title="Nouvelle conversation">+</button>
  </div>
  <div id="sessions-list"></div>

  <!-- Config -->
  <div class="sidebar-config">

    <div>
      <div class="section-label">Modèle IA</div>
      <select id="model-select">
        {% for m in models %}
        <option value="{{ m.id }}" {% if m.id == default_model %}selected{% endif %}>{{ m.label }}</option>
        {% endfor %}
      </select>
    </div>

    <div>
      <div class="section-label">Paramètres LLM</div>
      <div class="field-group">
        <label class="section-label" style="text-transform:none;letter-spacing:0;font-size:11px;margin-bottom:2px">
          Température &nbsp;<span id="temp-label" style="color:var(--accent2)">0.7</span>
        </label>
        <div class="slider-row">
          <input type="range" id="temperature" min="0" max="1" step="0.05" value="0.7"
                 oninput="document.getElementById('temp-label').textContent=parseFloat(this.value).toFixed(2)"/>
        </div>
        <label class="section-label" style="text-transform:none;letter-spacing:0;font-size:11px;margin-bottom:2px;margin-top:4px">Max tokens</label>
        <select id="max-tokens">
          <option value="1024">1 024</option>
          <option value="2048">2 048</option>
          <option value="4096" selected>4 096</option>
          <option value="8192">8 192</option>
          <option value="16384">16 384</option>
          <option value="65000">65 000 (max)</option>
        </select>
      </div>
    </div>

    <div>
      <div class="section-label">Contexte / Persona</div>
      <div class="field-group">
        <button class="btn btn-tpl btn-full" onclick="openModal('tpl-modal')">🎭 Choisir un template</button>
        <textarea id="system-prompt" placeholder="Ex: Tu es un expert Python…"></textarea>
        <button class="btn btn-primary btn-full" onclick="applyContext()">✓ Appliquer</button>
      </div>
    </div>

    <div>
      <div class="section-label">Skills & Expertise</div>
      <div class="field-group">
        <div class="skills-list" id="skills-list"></div>
        <button class="btn btn-secondary btn-full" onclick="openModal('skill-modal');setTimeout(()=>document.getElementById('skill-input').focus(),100)">➕ Ajouter un skill</button>
      </div>
    </div>

    <div>
      <div class="section-label">Stats de session</div>
      <div id="stats">
        Messages : <span id="stat-msgs">0</span><br/>
        Tokens estimés : <span id="stat-tokens">0</span><br/>
        Contexte : <span id="stat-ctx">Non</span><br/>
        Skills : <span id="stat-skills">0</span>
      </div>
    </div>

  </div>

  <div class="sidebar-actions">
    <button class="btn btn-ghost btn-full" onclick="exportChat()">⬇ Exporter conv.</button>
    <button class="btn btn-danger btn-full" onclick="clearChat()">✕ Effacer mémoire</button>
  </div>
</aside>

<!-- ── APP WRAPPER ── -->
<div id="app-wrapper">

  <div id="topbar">
    <button id="sidebar-toggle" onclick="document.getElementById('sidebar').classList.toggle('open')">☰</button>
    <div class="nav-tabs">
      <button class="nav-tab active" data-view="chat" onclick="switchView('chat')">💬 Chat</button>
      <button class="nav-tab claw-tab" data-view="claw" onclick="switchView('claw')">🐾 CLAW</button>
    </div>
    <div id="topbar-right">
      <button class="icon-btn" onclick="toggleSearch()" title="Rechercher (Ctrl+F)">🔍</button>
      <span id="skills-indicator">✨ Skills</span>
      <span id="context-indicator" style="display:none">Ctx actif</span>
      <span id="model-badge">{{ default_model }}</span>
    </div>
  </div>

  <!-- Search bar -->
  <div id="search-bar">
    <input type="text" id="search-input" placeholder="Rechercher dans la conversation…"
           oninput="doSearch()" onkeydown="searchKeyNav(event)"/>
    <span id="search-count"></span>
    <button class="search-nav-btn" onclick="navigateSearch(-1)" title="Précédent">↑</button>
    <button class="search-nav-btn" onclick="navigateSearch(1)"  title="Suivant">↓</button>
    <button id="search-close" onclick="toggleSearch()">✕</button>
  </div>

  <!-- ── CHAT VIEW ── -->
  <div id="chat-view" class="view-panel active">
    <div id="messages">
      <div id="welcome">
        <h2>Bonjour 👋</h2>
        <p>Assistant IA avec multi-sessions, templates, paramètres LLM et recherche intégrée.</p>
        <div class="hint">Ctrl+F pour rechercher · Entrée pour envoyer · Shift+Entrée pour saut de ligne</div>
      </div>
      <div id="typing">
        <div class="avatar bot">🤖</div>
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    </div>

    <div id="input-area">
      <div id="file-attach-area"></div>
      <button id="attach-btn" class="action-btn" onclick="document.getElementById('file-input').click()">
        <svg viewBox="0 0 24 24"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
      </button>
      <input type="file" id="file-input" multiple style="display:none" onchange="handleFileSelect(event)"/>
      <textarea id="message-input" placeholder="Envoie un message…" rows="1"></textarea>
      <button id="send-btn" class="action-btn" onclick="sendMessage()">
        <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    </div>
  </div>

  <!-- ── CLAW VIEW ── -->
  <div id="claw-view" class="view-panel">
    <div class="claw-header">
      <h2>🐾 CLAW Engine v2</h2>
      <span class="badge">Impl. → Debug → Optimisation · Validation syntaxique</span>
    </div>

    <div class="claw-grid">
      <div class="claw-card">
        <label>🎯 Objectif</label>
        <textarea id="claw-obj" placeholder="Ex: Convertis ce script en FastAPI avec tests Pytest…"></textarea>
        <label>📎 Fichiers sources</label>
        <div id="claw-file-zone"
             ondragover="event.preventDefault();this.classList.add('drag-over')"
             ondragleave="this.classList.remove('drag-over')"
             ondrop="handleClawDrop(event)">
          <input type="file" id="claw-files" multiple onchange="handleClawFiles(this.files)"/>
          📂 Glisse tes fichiers ici ou clique
          <div class="file-list" id="claw-file-list"></div>
        </div>
        <div id="claw-actions">
          <button class="btn btn-ghost" onclick="clearClaw()">↺ Reset</button>
          <button id="claw-run-btn" class="btn btn-secondary" onclick="runClaw()">🚀 Lancer CLAW</button>
        </div>
      </div>

      <div class="claw-card">
        <label>📜 Journal d'exécution</label>
        <div class="claw-progress" id="claw-progress-wrap" style="display:none">
          <div class="claw-progress-bar" id="claw-progress-bar"></div>
        </div>
        <div class="claw-logs" id="claw-logs">
          <span class="log-dim">En attente de lancement…</span>
        </div>
      </div>
    </div>

    <div class="claw-output" id="claw-output-area" style="display:none">
      <div class="claw-output-header">
        <span>✅ Code Final Généré</span>
        <div class="claw-output-header-btns">
          <button class="code-action-btn copy-btn" onclick="copyClawCode()">📋 Copier</button>
          <button class="code-action-btn dl-btn"   onclick="downloadClawCode()">⬇ Télécharger</button>
        </div>
      </div>
      <div class="claw-output-meta" id="claw-output-meta"></div>
      <pre><code id="claw-code" class="hljs"></code></pre>
    </div>
  </div>
  <!-- end claw view -->

</div><!-- end app-wrapper -->

<div id="toast"></div>

<script>
/* =====================================================
   GLOBALS
   ===================================================== */
let skills       = [];
let attachedFiles= [];
let clawFiles    = [];
let clawRunning  = false;
let selectedTpl  = null;

// Search state
let searchActive  = false;
let searchMatches = [];
let searchIdx     = 0;

/* =====================================================
   TEMPLATES
   ===================================================== */
const TEMPLATES = {{ templates|tojson }};

function buildTemplateGrid() {
  const grid = document.getElementById('tpl-grid');
  grid.innerHTML = TEMPLATES.map((t, i) => `
    <div class="tpl-card" data-idx="${i}" onclick="selectTemplate(${i})">
      <div class="tpl-icon">${t.icon}</div>
      <div class="tpl-label">${escHtml(t.label)}</div>
      <div class="tpl-preview">${escHtml(t.prompt)}</div>
    </div>`).join('');
}

function selectTemplate(idx) {
  selectedTpl = idx;
  document.querySelectorAll('.tpl-card').forEach((c, i) =>
    c.classList.toggle('selected', i === idx));
}

function applyTemplate() {
  if (selectedTpl === null) return showToast('❌ Sélectionne un template', 'error');
  document.getElementById('system-prompt').value = TEMPLATES[selectedTpl].prompt;
  closeModal('tpl-modal');
  showToast(`✓ Template "${TEMPLATES[selectedTpl].label}" appliqué`, 'success');
}

/* =====================================================
   SESSIONS
   ===================================================== */
async function loadSessions() {
  try {
    const res  = await fetch('/api/sessions');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    renderSessions(data.sessions, data.active_id);
  } catch(e) {
    console.error('loadSessions error:', e);
    showToast('❌ Impossible de charger les sessions', 'error');
  }
}

function renderSessions(sessions, activeId) {
  const list = document.getElementById('sessions-list');
  if (!sessions.length) { list.innerHTML = '<div style="padding:8px 10px;font-size:11px;color:var(--muted)">Aucune session</div>'; return; }
  list.innerHTML = sessions.map(s => `
    <div class="session-item ${s.id === activeId ? 'active' : ''}" onclick="switchSession('${s.id}')">
      <span class="session-icon">💬</span>
      <span class="session-name" id="sname-${s.id}" ondblclick="renameSession('${s.id}')">${escHtml(s.name)}</span>
      <div class="session-actions">
        <button class="session-action-btn edit" onclick="event.stopPropagation();renameSession('${s.id}')" title="Renommer">✏️</button>
        ${sessions.length > 1 ? `<button class="session-action-btn" onclick="event.stopPropagation();deleteSession('${s.id}')" title="Supprimer">✕</button>` : ''}
      </div>
    </div>`).join('');
}

async function createSession() {
  try {
    const res  = await fetch('/api/sessions', { method: 'POST' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    await loadSessionData();
    showToast('✓ Nouvelle conversation créée', 'success');
  } catch(e) {
    console.error('createSession error:', e);
    showToast('❌ Erreur création session', 'error');
  }
}

async function switchSession(id) {
  try {
    const res = await fetch(`/api/sessions/${id}/activate`, { method: 'POST' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    await loadSessionData();
  } catch(e) {
    console.error('switchSession error:', e);
    showToast('❌ Erreur changement de session', 'error');
  }
}

async function deleteSession(id) {
  if (!confirm('Supprimer cette conversation ?')) return;
  try {
    const res = await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    await loadSessionData();
    showToast('✕ Conversation supprimée', 'success');
  } catch(e) {
    console.error('deleteSession error:', e);
    showToast('❌ Erreur suppression session', 'error');
  }
}

function renameSession(id) {
  const nameEl = document.getElementById(`sname-${id}`);
  if (!nameEl) return;
  const current = nameEl.textContent;
  nameEl.innerHTML = `<input class="session-name-input" value="${escHtml(current)}"
    onblur="commitRename('${id}', this.value)"
    onkeydown="if(event.key==='Enter')this.blur();if(event.key==='Escape'){this.value='${escHtml(current)}';this.blur()}"
    onclick="event.stopPropagation()"/>`;
  const inp = nameEl.querySelector('input');
  inp.focus(); inp.select();
}

async function commitRename(id, newName) {
  newName = newName.trim() || 'Sans titre';
  await fetch(`/api/sessions/${id}/name`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: newName })
  });
  await loadSessions();
}

/* =====================================================
   LOAD SESSION DATA (messages + config)
   ===================================================== */
async function loadSessionData() {
  // Clear messages
  const msgs = document.getElementById('messages');
  msgs.innerHTML = `
    <div id="welcome">
      <h2>Bonjour 👋</h2>
      <p>Assistant IA avec multi-sessions, templates, paramètres LLM et recherche intégrée.</p>
      <div class="hint">Ctrl+F pour rechercher · Entrée pour envoyer · Shift+Entrée pour saut de ligne</div>
    </div>
    <div id="typing">
      <div class="avatar bot">🤖</div>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;

  const res  = await fetch('/api/history');
  const data = await res.json();

  if (data.messages?.length) {
    document.getElementById('welcome')?.remove();
    data.messages.forEach(m => appendMessage(m.role, m.content, false));
    updateStats(data.messages.length, data.estimated_tokens);
  }
  if (data.system_prompt) {
    document.getElementById('system-prompt').value = data.system_prompt;
    document.getElementById('context-indicator').style.display = 'flex';
    document.getElementById('stat-ctx').textContent = 'Oui';
  } else {
    document.getElementById('system-prompt').value = '';
    document.getElementById('context-indicator').style.display = 'none';
    document.getElementById('stat-ctx').textContent = 'Non';
  }
  if (data.skills?.length) { skills = data.skills; renderSkills(); }
  else { skills = []; renderSkills(); }
  if (data.model) {
    document.getElementById('model-select').value      = data.model;
    document.getElementById('model-badge').textContent = data.model;
  }
  if (data.temperature !== undefined) {
    document.getElementById('temperature').value = data.temperature;
    document.getElementById('temp-label').textContent = parseFloat(data.temperature).toFixed(2);
  }
  if (data.max_tokens !== undefined) {
    document.getElementById('max-tokens').value = data.max_tokens;
  }
  await loadSessions();
}

/* =====================================================
   SWITCH VIEW
   ===================================================== */
function switchView(view) {
  document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(view + '-view').classList.add('active');
  document.querySelectorAll('.nav-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.view === view));
}

/* =====================================================
   TEXTAREA AUTO-RESIZE + ENTER
   ===================================================== */
const msgInput = document.getElementById('message-input');
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 150) + 'px';
});
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

/* =====================================================
   CONTEXT + LLM PARAMS
   ===================================================== */
async function applyContext() {
  const prompt      = document.getElementById('system-prompt').value.trim();
  const model       = document.getElementById('model-select').value;
  const temperature = parseFloat(document.getElementById('temperature').value);
  const max_tokens  = parseInt(document.getElementById('max-tokens').value);

  const res = await fetch('/api/context', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system_prompt: prompt, model, skills, temperature, max_tokens })
  });
  if (res.ok) {
    const hasCtx = !!prompt;
    document.getElementById('context-indicator').style.display = hasCtx ? 'flex' : 'none';
    document.getElementById('stat-ctx').textContent = hasCtx ? 'Oui' : 'Non';
    document.getElementById('model-badge').textContent = model;
    showToast('✓ Paramètres appliqués', 'success');
  } else {
    showToast('❌ Erreur lors de l\'application', 'error');
  }
}

/* =====================================================
   SKILLS
   ===================================================== */
async function addSkill() {
  const name = document.getElementById('skill-input').value.trim();
  if (!name) return showToast('❌ Veuillez entrer un nom', 'error');
  if (skills.includes(name)) return showToast('❌ Skill déjà présent', 'error');
  skills.push(name);
  renderSkills();
  closeModal('skill-modal');
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
      <button class="skill-remove" onclick="removeSkill('${escHtml(s)}')">✕</button>
    </div>`).join('');
  const has = skills.length > 0;
  document.getElementById('skills-indicator').style.display = has ? 'flex' : 'none';
  document.getElementById('stat-skills').textContent = skills.length;
}

/* =====================================================
   FILE ATTACH
   ===================================================== */
function handleFileSelect(event) {
  Array.from(event.target.files).forEach(f => {
    if (f.size > 5 * 1024 * 1024) return showToast(`❌ ${f.name} dépasse 5MB`, 'error');
    if (attachedFiles.find(x => x.name === f.name)) return;
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

/* =====================================================
   SEND CHAT MESSAGE
   ===================================================== */
async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text && attachedFiles.length === 0) return;

  msgInput.value = '';
  msgInput.style.height = 'auto';
  document.getElementById('welcome')?.remove();

  let display = text || '[Fichier joint]';
  if (attachedFiles.length) display += `\n📎 ${attachedFiles.map(f=>f.name).join(', ')}`;
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
    // Auto-rename session from first user message
    if (data.total_messages <= 2 && text) {
      const name = text.slice(0, 40) + (text.length > 40 ? '…' : '');
      await fetch(`/api/sessions/active/name`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      loadSessions();
    }
  } catch {
    setTyping(false);
    showToast('❌ Erreur réseau', 'error');
  }
}

/* =====================================================
   RENDER MESSAGE
   ===================================================== */
function appendMessage(role, text, scroll = true) {
  // Normalise le rôle : le backend stocke "assistant", le frontend utilise "bot"
  const displayRole = (role === 'assistant') ? 'bot' : role;
  const container = document.getElementById('messages');
  const typing    = document.getElementById('typing');

  const wrapper = document.createElement('div');
  wrapper.className = `msg-wrapper ${displayRole}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${displayRole}`;
  avatar.textContent = displayRole === 'user' ? '🧑' : '🤖';

  const inner  = document.createElement('div');
  const bubble = document.createElement('div');
  bubble.className = `bubble ${displayRole}`;
  bubble.innerHTML = displayRole === 'bot' ? marked.parse(text) : escHtml(text).replace(/\n/g,'<br>');
  if (displayRole === 'bot') bubble.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));

  // Copy button
  const copyBtn = document.createElement('button');
  copyBtn.className = 'bubble-copy';
  copyBtn.textContent = '📋';
  copyBtn.title = 'Copier';
  copyBtn.onclick = () => {
    navigator.clipboard.writeText(text).then(() => showToast('📋 Copié !', 'success'));
  };
  bubble.appendChild(copyBtn);

  const time = document.createElement('div');
  time.className = 'msg-time';
  time.textContent = new Date().toLocaleTimeString('fr-FR', {hour:'2-digit',minute:'2-digit'});
  if (displayRole === 'user') time.style.textAlign = 'right';

  inner.append(bubble, time);
  if (displayRole === 'user') wrapper.append(inner, avatar);
  else                        wrapper.append(avatar, inner);

  container.insertBefore(wrapper, typing);
  if (scroll) container.scrollTop = container.scrollHeight;
}

function setTyping(on) {
  document.getElementById('typing').className = on ? 'visible' : '';
  document.getElementById('send-btn').disabled = on;
  if (on) document.getElementById('messages').scrollTop = 9999;
}
function updateStats(msgs, tokens) {
  document.getElementById('stat-msgs').textContent   = msgs   ?? '–';
  document.getElementById('stat-tokens').textContent = tokens ?? '–';
}

/* =====================================================
   SEARCH
   ===================================================== */
function toggleSearch() {
  searchActive = !searchActive;
  const bar = document.getElementById('search-bar');
  if (searchActive) {
    bar.classList.add('visible');
    document.getElementById('search-input').focus();
  } else {
    bar.classList.remove('visible');
    clearSearch();
    document.getElementById('search-input').value = '';
  }
}

function doSearch() {
  const query = document.getElementById('search-input').value.trim().toLowerCase();
  clearSearch();
  if (!query || query.length < 2) {
    document.getElementById('search-count').textContent = '';
    return;
  }
  const bubbles = document.querySelectorAll('.bubble');
  bubbles.forEach(bubble => {
    if (bubble.textContent.toLowerCase().includes(query)) {
      bubble.classList.add('search-match');
      searchMatches.push(bubble);
    }
  });
  const count = searchMatches.length;
  document.getElementById('search-count').textContent =
    count ? `${searchIdx + 1}/${count}` : '0 résultat';
  if (count) {
    searchMatches[0].classList.add('search-current');
    searchMatches[0].scrollIntoView({behavior: 'smooth', block: 'center'});
  }
}

function navigateSearch(dir) {
  if (!searchMatches.length) return;
  searchMatches[searchIdx].classList.remove('search-current');
  searchIdx = (searchIdx + dir + searchMatches.length) % searchMatches.length;
  searchMatches[searchIdx].classList.add('search-current');
  searchMatches[searchIdx].scrollIntoView({behavior: 'smooth', block: 'center'});
  document.getElementById('search-count').textContent =
    `${searchIdx + 1}/${searchMatches.length}`;
}

function searchKeyNav(e) {
  if (e.key === 'Enter')  { e.preventDefault(); navigateSearch(e.shiftKey ? -1 : 1); }
  if (e.key === 'Escape') { toggleSearch(); }
}

function clearSearch() {
  document.querySelectorAll('.search-match').forEach(el =>
    el.classList.remove('search-match', 'search-current'));
  searchMatches = [];
  searchIdx = 0;
}

/* =====================================================
   CLEAR & EXPORT
   ===================================================== */
async function clearChat() {
  if (!confirm('Effacer toute la mémoire de cette conversation ?')) return;
  await fetch('/api/clear', { method: 'POST' });
  const m = document.getElementById('messages');
  m.innerHTML = `
    <div id="welcome"><h2>Effacé ✓</h2><p>Mémoire réinitialisée.</p></div>
    <div id="typing"><div class="avatar bot">🤖</div>
    <div class="typing-dots"><span></span><span></span><span></span></div></div>`;
  updateStats(0, 0);
  showToast('Mémoire effacée', 'success');
}

async function exportChat() {
  const res  = await fetch('/api/history');
  const data = await res.json();
  if (!data.messages?.length) return showToast('Aucun message à exporter', 'error');
  let md = '# Conversation — ' + new Date().toLocaleString('fr-FR') + '\n\n';
  if (data.system_prompt) md += '> **Contexte :** ' + data.system_prompt + '\n\n';
  data.messages.forEach(m => {
    md += `**${m.role === 'user' ? 'Vous' : 'IA'}**\n\n${m.content}\n\n---\n\n`;
  });
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([md], {type:'text/markdown'})),
    download: 'conversation_' + Date.now() + '.md'
  });
  a.click();
  showToast('✓ Exporté', 'success');
}

/* =====================================================
   CLAW — FILE HANDLING
   ===================================================== */
function handleClawFiles(files) {
  Array.from(files).forEach(f => {
    if (!clawFiles.find(x => x.name === f.name)) clawFiles.push(f);
  });
  renderClawFiles();
  document.getElementById('claw-files').value = '';
}
function handleClawDrop(e) {
  e.preventDefault();
  document.getElementById('claw-file-zone').classList.remove('drag-over');
  handleClawFiles(e.dataTransfer.files);
}
function renderClawFiles() {
  document.getElementById('claw-file-list').innerHTML =
    clawFiles.map((f, i) => `
      <div class="file-tag">📄 ${escHtml(f.name)}
        <span onclick="removeClawFile(${i})">✕</span>
      </div>`).join('');
}
function removeClawFile(i) { clawFiles.splice(i, 1); renderClawFiles(); }

/* =====================================================
   CLAW — MAIN LOOP
   ===================================================== */
function clawLog(msg, type = 'info') {
  const logs = document.getElementById('claw-logs');
  const line = document.createElement('div');
  line.className = 'log-line log-' + type;
  line.textContent = msg;
  logs.appendChild(line);
  logs.scrollTop = logs.scrollHeight;
}
function setClawProgress(pct) {
  document.getElementById('claw-progress-bar').style.width = pct + '%';
}

async function runClaw() {
  if (clawRunning) return;
  const obj = document.getElementById('claw-obj').value.trim();
  if (!obj) return showToast('❌ Objectif manquant', 'error');

  clawRunning = true;
  const runBtn = document.getElementById('claw-run-btn');
  runBtn.disabled = true; runBtn.textContent = '⏳ En cours…';

  document.getElementById('claw-logs').innerHTML = '';
  document.getElementById('claw-output-area').style.display = 'none';
  document.getElementById('claw-progress-wrap').style.display = 'block';
  setClawProgress(0);

  const fd = new FormData();
  fd.append('objective', obj);
  clawFiles.forEach(f => fd.append('claw_files', f));

  try {
    const res  = await fetch('/api/claw/process', { method: 'POST', body: fd });
    const data = await res.json();

    if (data.error) { clawLog('❌ ' + data.error, 'err'); showToast('❌ ' + data.error, 'error'); return; }

    (data.logs || []).forEach(line => {
      let type = 'info';
      if (!line.trim()) { clawLog('', 'dim'); return; }
      if (line.includes('❌'))            type = 'err';
      else if (line.includes('✅') || line.includes('🎯') || line.includes('🏁')) type = 'ok';
      else if (line.includes('⚠️') || line.includes('🔄')) type = 'warn';
      else if (line.includes('══'))       type = 'phase';
      else if (line.includes('↩️'))       type = 'retry';
      else if (line.startsWith('   '))    type = 'dim';
      clawLog(line, type);
    });

    setClawProgress(100);

    if (data.final_code) {
      const codeEl = document.getElementById('claw-code');
      codeEl.textContent = data.final_code;
      hljs.highlightElement(codeEl);

      const meta = document.getElementById('claw-output-meta');
      meta.innerHTML = [
        data.char_count  ? `<span>Taille : <em>${data.char_count.toLocaleString()} caractères</em></span>` : '',
        data.validated   ? `<span>Syntaxe : <em>✅ valide</em></span>` : `<span>Syntaxe : <em>⚠️ non validée</em></span>`,
        data.retries > 0 ? `<span>Retentatives : <em>${data.retries}</em></span>` : '',
        `<span>Langage : <em>${detectLanguage(data.final_code).label}</em></span>`,
      ].filter(Boolean).join('');

      document.getElementById('claw-output-area').style.display = 'block';
      showToast('✅ CLAW terminé !', 'success');
    }
  } catch(e) {
    clawLog('❌ Erreur réseau : ' + e.message, 'err');
    showToast('❌ Erreur réseau', 'error');
  } finally {
    clawRunning = false;
    runBtn.disabled = false;
    runBtn.textContent = '🚀 Lancer CLAW';
  }
}

function clearClaw() {
  document.getElementById('claw-obj').value = '';
  clawFiles = []; renderClawFiles();
  document.getElementById('claw-logs').innerHTML = '<span class="log-dim">En attente…</span>';
  document.getElementById('claw-output-area').style.display = 'none';
  document.getElementById('claw-progress-wrap').style.display = 'none';
  setClawProgress(0);
  showToast('🔄 Réinitialisé', 'success');
}

function copyClawCode() {
  navigator.clipboard.writeText(document.getElementById('claw-code').textContent)
    .then(() => showToast('📋 Code copié !', 'success'));
}

/* =====================================================
   CLAW — DOWNLOAD WITH LANGUAGE DETECTION
   ===================================================== */
function detectLanguage(code) {
  const c = code.slice(0, 600).toLowerCase();
  if (c.includes('#!/usr/bin/env python') || /\ndef \w+\(/.test(code.slice(0,200)) || c.includes('import os') || c.includes('from flask')) return {ext: '.py',  label: 'Python'};
  if (c.includes('<!doctype html') || c.includes('<html'))                             return {ext: '.html', label: 'HTML'};
  if (c.includes('#!/bin/bash') || c.includes('#!/bin/sh'))                           return {ext: '.sh',   label: 'Bash'};
  if (c.includes('package main') && c.includes('func main'))                          return {ext: '.go',   label: 'Go'};
  if (c.includes('import react') || c.includes('from react'))                         return {ext: '.jsx',  label: 'React/JSX'};
  if (c.includes('interface ') && c.includes(': string') || c.includes(': number'))   return {ext: '.ts',   label: 'TypeScript'};
  if (c.includes('const ') && c.includes('=>') || c.includes('require('))             return {ext: '.js',   label: 'JavaScript'};
  if (c.includes('#[derive') || c.includes('fn main()'))                              return {ext: '.rs',   label: 'Rust'};
  if (c.includes('public class') || c.includes('public static void main'))            return {ext: '.java', label: 'Java'};
  if (c.includes('using system;') || c.includes('namespace '))                        return {ext: '.cs',   label: 'C#'};
  if (c.includes('#include') && c.includes('int main'))                               return {ext: '.cpp',  label: 'C++'};
  if (c.includes('select ') && c.includes('from ') && c.includes('where '))          return {ext: '.sql',  label: 'SQL'};
  return {ext: '.txt', label: 'Texte'};
}

function downloadClawCode() {
  const code = document.getElementById('claw-code').textContent;
  const {ext, label} = detectLanguage(code);
  const filename = `claw_output_${Date.now()}${ext}`;
  const blob = new Blob([code], {type: 'text/plain'});
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob), download: filename
  });
  a.click();
  showToast(`⬇ Téléchargé : ${filename}`, 'success');
}

/* =====================================================
   MODALS
   ===================================================== */
function openModal(id) {
  document.getElementById(id).classList.add('active');
  if (id === 'tpl-modal') buildTemplateGrid();
}
function closeModal(id) {
  document.getElementById(id).classList.remove('active');
}

/* =====================================================
   KEYBOARD SHORTCUTS
   ===================================================== */
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
    if (searchActive) toggleSearch();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
    e.preventDefault();
    if (document.getElementById('chat-view').classList.contains('active')) toggleSearch();
  }
});

/* =====================================================
   UTILS
   ===================================================== */
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

/* =====================================================
   INIT
   ===================================================== */
(async () => {
  try {
    await loadSessionData();
  } catch(e) {
    console.error('Init error:', e);
  }
})();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _new_conversation(name: str = "Nouvelle conversation") -> dict[str, Any]:
    cid = str(uuid.uuid4())
    return {
        "id":            cid,
        "name":          name,
        "messages":      [],
        "system_prompt": "",
        "model":         DEFAULT_MODEL,
        "skills":        [],
        "temperature":   0.7,
        "max_tokens":    4096,
        "created_at":    time.time(),
    }


def _get_user_store() -> dict[str, Any]:
    """Return the top-level user store, creating it if absent."""
    session.permanent = True  # sessions persistent (30 jours)
    fid = session.get("id")
    if not fid or fid not in store:
        fid = str(uuid.uuid4())
        session["id"] = fid
        first_conv = _new_conversation("Conversation 1")
        store[fid]  = {"active_id": first_conv["id"], "sessions": {first_conv["id"]: first_conv}}
    return store[fid]


def _get_session() -> dict[str, Any]:
    """Return the currently active conversation."""
    us  = _get_user_store()
    aid = us["active_id"]
    # Robustesse : si l'active_id pointe vers une session inexistante
    if aid not in us["sessions"]:
        if us["sessions"]:
            aid = next(iter(us["sessions"]))
            us["active_id"] = aid
        else:
            conv = _new_conversation("Conversation 1")
            us["sessions"][conv["id"]] = conv
            us["active_id"] = conv["id"]
            aid = conv["id"]
    return us["sessions"][aid]


def _build_system_content(sess: dict) -> str:
    parts: list[str] = []
    if sess.get("skills"):
        parts.append(f"Skills and expertise: {', '.join(sess['skills'])}.")
    if sess.get("system_prompt"):
        parts.append(sess["system_prompt"])
    return "\n\n".join(parts)


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


# ---------------------------------------------------------------------------
# LLM helpers (same as v2)
# ---------------------------------------------------------------------------

_APP_URL = (
    os.environ.get("RENDER_EXTERNAL_URL")
    or os.environ.get("APP_URL")
    or "https://localhost"
)
_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer":  _APP_URL,
    "X-Title":       "AI Chatbot CLAW v3",
    "Content-Type":  "application/json",
}

_PLACEHOLDER_PATTERNS = re.compile(
    r"(#\s*(TODO|FIXME|\.\.\.)|pass\s*#\s*(implement|todo)|\.{3}\s*\n|"
    r"# rest of (the )?(code|implementation)|# similar to above|"
    r"# \.\.\. existing code|\[suite du code\]|\[contenu identique\])",
    re.IGNORECASE,
)


def _call_llm_raw(messages: list[dict], model: str, temperature: float = 0.7,
                  max_tokens: int = 4096, timeout: int = 90) -> str | None:
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp    = requests.post(OPENROUTER_URL, headers=_HEADERS, json=payload, timeout=timeout)
        resp.raise_for_status()
        data    = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content")
        return content.strip() if content else None
    except Exception as exc:
        logger.error("LLM call error: %s", exc)
        return None


def _call_llm_with_retry(messages: list[dict], model: str,
                          temperature: float = 0.7, max_tokens: int = 4096,
                          max_retries: int = 2, timeout: int = 90) -> tuple[str | None, int]:
    retries = 0
    for attempt in range(max_retries + 1):
        result = _call_llm_raw(messages, model, temperature, max_tokens, timeout)
        if result:
            return result, retries
        retries += 1
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    return None, retries


def _call_llm(messages: list[dict], model: str,
               temperature: float = 0.7, max_tokens: int = 4096) -> str:
    result, _ = _call_llm_with_retry(messages, model, temperature, max_tokens)
    return result or "ℹ️ Aucune réponse générée. Réessaie."


def _clean_code_block(text: str) -> str:
    blocks = re.findall(r"```(?:\w*)\n?([\s\S]*?)```", text)
    if not blocks:
        return text.strip()
    return max(blocks, key=len).strip()


def _validate_python_syntax(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"Ligne {exc.lineno}: {exc.msg}"
    except Exception as exc:
        return False, str(exc)


def _has_placeholders(code: str) -> bool:
    return bool(_PLACEHOLDER_PATTERNS.search(code))


def _is_code_shorter(new_code: str, old_code: str, threshold: float = 0.70) -> bool:
    if not old_code or len(old_code) < 200:
        return False
    return len(new_code) < len(old_code) * threshold


# ---------------------------------------------------------------------------
# Routes — sessions
# ---------------------------------------------------------------------------

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    us = _get_user_store()
    sessions_list = sorted(
        [{"id": s["id"], "name": s["name"], "created_at": s["created_at"],
          "msg_count": len(s["messages"])}
         for s in us["sessions"].values()],
        key=lambda x: x["created_at"]
    )
    return jsonify({"sessions": sessions_list, "active_id": us["active_id"]})


@app.route("/api/sessions", methods=["POST"])
def create_session():
    us   = _get_user_store()
    conv = _new_conversation(f"Conversation {len(us['sessions']) + 1}")
    us["sessions"][conv["id"]] = conv
    us["active_id"] = conv["id"]
    return jsonify({"ok": True, "id": conv["id"]})


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def delete_session(sid: str):
    us = _get_user_store()
    if sid not in us["sessions"]:
        return jsonify({"error": "Session introuvable"}), 404
    if len(us["sessions"]) <= 1:
        return jsonify({"error": "Impossible de supprimer la dernière session"}), 400
    del us["sessions"][sid]
    if us["active_id"] == sid:
        us["active_id"] = next(iter(us["sessions"]))
    return jsonify({"ok": True})


@app.route("/api/sessions/<sid>/activate", methods=["POST"])
def activate_session(sid: str):
    us = _get_user_store()
    if sid not in us["sessions"]:
        return jsonify({"error": "Session introuvable"}), 404
    us["active_id"] = sid
    return jsonify({"ok": True})


@app.route("/api/sessions/<sid>/name", methods=["PATCH"])
@app.route("/api/sessions/active/name", methods=["PATCH"])
def rename_session(sid: str = "active"):
    us   = _get_user_store()
    data = request.get_json(force=True)
    name = data.get("name", "").strip() or "Sans titre"
    target_id = us["active_id"] if sid == "active" else sid
    if target_id not in us["sessions"]:
        return jsonify({"error": "Session introuvable"}), 404
    us["sessions"][target_id]["name"] = name
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes — config / context
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    return render_template_string(
        HTML_TEMPLATE, models=FREE_MODELS,
        default_model=DEFAULT_MODEL, templates=PROMPT_TEMPLATES
    )


@app.route("/health")
def health():
    """Endpoint de santé pour Render keep-alive."""
    return jsonify({"status": "ok", "api_key_set": bool(API_KEY)}), 200


@app.errorhandler(500)
def internal_error(exc):
    logger.error("500 Internal Error: %s", exc)
    return jsonify({"error": "Erreur serveur interne. Réessaie."}), 500


@app.errorhandler(404)
def not_found(exc):
    return jsonify({"error": "Route introuvable."}), 404


@app.route("/api/context", methods=["POST"])
def set_context():
    data = request.get_json(force=True)
    sess = _get_session()
    sess["system_prompt"] = data.get("system_prompt", "").strip()
    if data.get("model"):
        sess["model"] = data["model"]
    if "skills" in data:
        sess["skills"] = [s for s in data["skills"] if isinstance(s, str)]
    if "temperature" in data:
        sess["temperature"] = max(0.0, min(1.0, float(data["temperature"])))
    if "max_tokens" in data:
        sess["max_tokens"] = int(data["max_tokens"])
    return jsonify({"ok": True})


@app.route("/api/skills", methods=["POST"])
def set_skills():
    data = request.get_json(force=True)
    sess = _get_session()
    sess["skills"] = [s for s in data.get("skills", []) if isinstance(s, str)]
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes — chat
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():
    if request.content_type and "multipart/form-data" in request.content_type:
        user_msg = request.form.get("message", "").strip()
        files    = request.files.getlist("files")
    else:
        body     = request.get_json(force=True)
        user_msg = body.get("message", "").strip()
        files    = []

    if not user_msg and not files:
        return jsonify({"error": "Message vide"}), 400

    sess = _get_session()

    file_blocks: list[str] = []
    for f in files:
        if not f.filename:
            continue
        try:
            content = f.read(250_000).decode("utf-8", errors="replace")
            file_blocks.append(f"📄 **{f.filename}**:\n```\n{content}\n```")
        except Exception as exc:
            logger.warning("File read error %s: %s", f.filename, exc)

    full_msg = user_msg
    if file_blocks:
        full_msg += "\n\n" + "\n\n".join(file_blocks)

    sess["messages"].append({"role": "user", "content": full_msg})

    payload_messages: list[dict] = []
    sys_content = _build_system_content(sess)
    if sys_content:
        payload_messages.append({"role": "system", "content": sys_content})
    payload_messages.extend(sess["messages"])

    payload = {
        "model":       sess["model"],
        "messages":    payload_messages,
        "max_tokens":  sess.get("max_tokens", 4096),
        "temperature": sess.get("temperature", 0.7),
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=_HEADERS, json=payload, timeout=90)
        result   = response.json()
    except requests.Timeout:
        sess["messages"].pop()
        return jsonify({"error": "Délai d'attente dépassé. Réessaie ou choisis un modèle plus rapide."}), 504
    except requests.RequestException as exc:
        sess["messages"].pop()
        return jsonify({"error": f"Erreur réseau: {exc}"}), 502

    if "error" in result:
        sess["messages"].pop()
        return jsonify({"error": result["error"].get("message", "Erreur inconnue")}), 502

    choices = result.get("choices") or []
    if not choices:
        sess["messages"].pop()
        return jsonify({"error": "Aucune réponse du modèle. Réessaie ou change de modèle."}), 502

    reply: str = choices[0].get("message", {}).get("content") or ""
    if not reply:
        sess["messages"].pop()
        return jsonify({"error": "Réponse vide reçue du modèle."}), 502
    sess["messages"].append({"role": "assistant", "content": reply})

    return jsonify({
        "reply":            reply,
        "total_messages":   len(sess["messages"]),
        "estimated_tokens": _estimate_tokens(sess["messages"]),
    })


@app.route("/api/clear", methods=["POST"])
def clear():
    _get_session()["messages"] = []
    return jsonify({"ok": True})


@app.route("/api/history", methods=["GET"])
def history():
    sess = _get_session()
    return jsonify({
        "messages":         sess["messages"],
        "system_prompt":    sess["system_prompt"],
        "model":            sess["model"],
        "skills":           sess.get("skills", []),
        "temperature":      sess.get("temperature", 0.7),
        "max_tokens":       sess.get("max_tokens", 4096),
        "estimated_tokens": _estimate_tokens(sess["messages"]),
    })


# ---------------------------------------------------------------------------
# CLAW ENGINE v2 (unchanged logic, uses session rature)
# ---------------------------------------------------------------------------

_ITER_FOCUS = {
    1: "implémentation initiale complète et fonctionnelle",
    2: "correction des bugs, cas limites et robustesse",
    3: "optimisation, lisibilité et qualité finale",
}
_SYS_ARCHITECT = (
    "Tu es un architecte logiciel senior. Produis un plan d'action PRÉCIS et STRUCTURÉ "
    "pour atteindre l'objectif. Liste modules, fonctions clés, dépendances et cas limites. "
    "Sois concis mais complet — maximum 30 lignes."
)
_SYS_CRITIC = (
    "Tu es un expert en revue de code et architecture. Identifie les failles du plan : "
    "cas limites oubliés, performance, sécurité, logique. "
    "Retourne UNIQUEMENT une liste numérotée de problèmes concrets. Maximum 15 points."
)
_SYS_DEV = (
    "Tu es un développeur expert. Génère UNIQUEMENT le code complet et fonctionnel.\n"
    "RÈGLES ABSOLUES :\n"
    "1. Inclus la totalité du code — aucun placeholder, aucun '# TODO', aucun '...'\n"
    "2. Si tu modifies du code existant, reproduis l'intégralité du fichier\n"
    "3. Ne résume jamais une section existante\n"
    "4. Utilise des blocs ```python ... ``` pour délimiter le code\n"
    "5. Le code doit être prêt à l'exécution sans modification"
)
_SYS_FIX_SYNTAX = (
    "Tu es un développeur expert. Le code fourni contient des erreurs de syntaxe. "
    "Corrige-les et retourne UNIQUEMENT le code corrigé complet dans un bloc ```python. "
    "Ne supprime aucune fonctionnalité."
)
_SYS_IMPROVE = (
    "Tu es un expert en qualité logicielle. Produis une liste COURTE (max 10 points) "
    "d'améliorations concrètes pour la prochaine itération. Format : numéroté, une ligne par point."
)


@app.route("/api/claw/process", methods=["POST"])
def claw_process():
    sess      = _get_session()
    objective = request.form.get("objective", "").strip()
    if not objective:
        return jsonify({"error": "Veuillez définir un objectif clair."}), 400

    model       = sess["model"]
    temperature = sess.get("temperature", 0.7)

    files: list[str] = []
    for f in request.files.getlist("claw_files"):
        if not f.filename:
            continue
        try:
            content = f.read().decode("utf-8", errors="replace")
            if len(content) > 30_000:
                content = content[:30_000] + "\n\n…[tronqué]"
            files.append(f"### {f.filename}\n{content}")
        except Exception as exc:
            logger.warning("CLAW file read error: %s", exc)

    current_code: str = (
        "\n\n".join(files) if files
        else "# Aucun code initial — génère depuis zéro."
    )

    logs: list[str]         = []
    improvements: list[str] = []
    total_retries: int      = 0
    validated: bool         = False
    success: bool           = True

    def log(msg: str = "") -> None:
        logs.append(msg)

    for iteration in range(1, 4):
        focus = _ITER_FOCUS[iteration]
        log()
        log(f"══ Itération {iteration}/3 — {focus} ══")

        # ── 1. Plan ──────────────────────────────────────────
        log("   📝 Planification…")
        improvement_ctx = "\n".join(f"  - {p}" for p in improvements) or "Aucune"
        plan, r = _call_llm_with_retry([
            {"role": "system", "content": _SYS_ARCHITECT},
            {"role": "user",   "content": (
                f"OBJECTIF: {objective}\n\nFOCUS: {focus}\n\n"
                f"CODE ACTUEL ({len(current_code)} chars):\n{current_code}\n\n"
                f"POINTS À AMÉLIORER:\n{improvement_ctx}"
            )},
        ], model, temperature)
        total_retries += r
        if not plan:
            log("   ❌ Planification échouée"); success = False; break
        log("   ✅ Plan établi")

        # ── 2. Critique ──────────────────────────────────────
        log("   🔍 Critique du plan…")
        critique, r = _call_llm_with_retry([
            {"role": "system", "content": _SYS_CRITIC},
            {"role": "user",   "content": f"OBJECTIF: {objective}\n\nPLAN:\n{plan}\n\nCODE:\n{current_code}"},
        ], model, temperature)
        total_retries += r
        log("   ✅ Critique prête" if critique else "   ⚠️  Critique indisponible")
        if not critique:
            critique = "(aucune critique)"

        # ── 3. Implement ─────────────────────────────────────
        log("   ⚙️  Implémentation…")
        raw_code, r = _call_llm_with_retry([
            {"role": "system", "content": _SYS_DEV},
            {"role": "user",   "content": (
                f"OBJECTIF: {objective}\n\nFOCUS: {focus}\n\n"
                f"PLAN:\n{plan}\n\nCRITIQUES:\n{critique}\n\n"
                f"CODE ACTUEL ({len(current_code)} chars):\n{current_code}\n\n"
                "RAPPEL: Génère la totalité du code, sans troncature."
            )},
        ], model, temperature)
        total_retries += r
        if not raw_code:
            log("   ❌ Implémentation échouée"); success = False; break

        candidate = _clean_code_block(raw_code)

        if _is_code_shorter(candidate, current_code):
            log(f"   ⚠️  Code trop court ({len(candidate)} vs {len(current_code)}) — version précédente conservée")
        else:
            if _has_placeholders(candidate):
                log("   ⚠️  Placeholders détectés — correction…")
                fixed, r2 = _call_llm_with_retry([
                    {"role": "system", "content": _SYS_DEV},
                    {"role": "user",   "content": f"Complète ce code incomplet (objectif: {objective}):\n{candidate}"},
                ], model, temperature)
                total_retries += r2
                if fixed and not _is_code_shorter(_clean_code_block(fixed), current_code):
                    candidate = _clean_code_block(fixed)
                    log("   🔄 Placeholders corrigés")
            current_code = candidate
            log(f"   🎯 Code généré ({len(current_code):,} chars)")

        # Syntax validation
        validated, syntax_err = _validate_python_syntax(current_code)
        if validated:
            log("   ✅ Syntaxe Python valide")
        else:
            log(f"   ⚠️  Erreur syntaxe : {syntax_err} — correction…")
            fixed_s, r3 = _call_llm_with_retry([
                {"role": "system", "content": _SYS_FIX_SYNTAX},
                {"role": "user",   "content": f"ERREUR: {syntax_err}\n\nCODE:\n{current_code}"},
            ], model, temperature)
            total_retries += r3
            if fixed_s:
                c2 = _clean_code_block(fixed_s)
                ok2, _ = _validate_python_syntax(c2)
                if ok2:
                    current_code = c2; validated = True
                    log("   ✅ Syntaxe corrigée")
                else:
                    log("   ❌ Correction syntaxique échouée")

        # ── 4. Improvement analysis ───────────────────────────
        log("   🔬 Analyse pour itération suivante…")
        improve_raw, r = _call_llm_with_retry([
            {"role": "system", "content": _SYS_IMPROVE},
            {"role": "user",   "content": f"OBJECTIF: {objective}\n\nCODE:\n{current_code}"},
        ], model, temperature)
        total_retries += r
        if improve_raw:
            improvements = [
                ln.lstrip("0123456789.-) ").strip()
                for ln in improve_raw.splitlines()
                if ln.strip() and ln.strip()[0].isdigit()
            ][:10]
            if improvements:
                log(f"   🧠 {len(improvements)} amélioration(s) identifiée(s)")

        log(f"✅ Itération {iteration}/3 terminée")

    log()
    if success:
        log("🏁 CLAW terminé avec succès !")
        log(f"📦 Code final : {len(current_code):,} caractères")
        if validated: log("🟢 Syntaxe Python validée")
        if total_retries: log(f"🔄 Retentatives : {total_retries}")
    else:
        log("⚠️  CLAW interrompu suite à une erreur.")

    final_code = (
        current_code
        if current_code and "Aucun code initial" not in current_code
        else None
    )
    return jsonify({
        "logs":       logs,
        "final_code": final_code,
        "success":    success,
        "char_count": len(final_code) if final_code else 0,
        "validated":  validated,
        "retries":    total_retries,
    })
def start_async_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ping())
def start_background_ping():
    thread = threading.Thread(target=start_async_loop, daemon=True)
    thread.start()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 AI Chatbot + CLAW v3")
    # Render donne un port dans la variable PORT, sinon on prend 5000 en local
    port = int(os.environ.get("PORT", 5000))
    start_background_ping()
    app.run(host="0.0.0.0", port=port)
