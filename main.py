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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENROUTER_API_KEY")

OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL   = "minimax/minimax-m2.5:free"

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]   = 50 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 50 * 1024 * 1024
app.secret_key = "change-me-in-production-use-os-urandom"

store: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Available free models on OpenRouter
# ---------------------------------------------------------------------------
FREE_MODELS: list[dict[str, str]] = [
    {"id": "minimax/minimax-m2.5:free",              "label": "MiniMax M2.5"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B"},
    {"id": "mistralai/mistral-7b-instruct:free",     "label": "Mistral 7B"},
    {"id": "google/gemma-3-4b-it:free",              "label": "Gemma 3 4B"},
    {"id": "deepseek/deepseek-r1:free",              "label": "DeepSeek R1"},
    {"id": "qwen/qwen3.6-plus-preview:free",         "label": "Qwen3.6"},
]

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Chatbot + CLAW</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/tokyo-night-dark.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<style>
:root {
  --bg:       #0a0e1a;
  --panel:    #111827;
  --border:   #1e2d45;
  --accent:   #3b82f6;
  --accent2:  #06b6d4;
  --accent3:  #8b5cf6;
  --user-bg:  #1d3461;
  --bot-bg:   #131f35;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --danger:   #ef4444;
  --success:  #22c55e;
  --warn:     #f59e0b;
  --radius:   12px;
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

#sidebar {
  width: 300px;
  min-width: 300px;
  background: var(--panel);
  border-right: 1px solid var(--border);
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
  background: linear-gradient(135deg, var(--accent), var(--accent2));
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

select, textarea, input[type="text"], input[type="password"] {
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
.btn-primary   { background: var(--accent);  color: #fff; }
.btn-primary:hover   { background: #2563eb; }
.btn-secondary { background: var(--accent3); color: #fff; }
.btn-secondary:hover { background: #7c3aed; }
.btn-danger    { background: transparent; color: var(--danger); border: 1px solid var(--danger); }
.btn-danger:hover    { background: rgba(239,68,68,.1); }
.btn-ghost     { background: transparent; color: var(--muted); border: 1px solid var(--border); }
.btn-ghost:hover     { border-color: var(--accent); color: var(--accent); }
.btn-warn      { background: var(--warn); color: #000; }
.btn-warn:hover      { background: #d97706; }
.btn-full  { width: 100%; }
.btn:disabled { opacity: .4; cursor: not-allowed; }

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

#app-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

#topbar {
  padding: 12px 24px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}
.nav-tabs { display: flex; gap: 6px; }
.nav-tab {
  padding: 7px 16px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  font-family: 'Syne', sans-serif;
  font-size: 13px; font-weight: 600;
  cursor: pointer; transition: all .2s;
}
.nav-tab:hover { border-color: var(--accent); color: var(--accent); }
.nav-tab.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.nav-tab.claw-tab.active {
  background: var(--accent3);
  border-color: var(--accent3);
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
  display: flex; align-items: center; gap: 5px;
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
}

.view-panel {
  flex: 1;
  display: none;
  flex-direction: column;
  overflow: hidden;
}
.view-panel.active { display: flex; }

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
.avatar.bot  { background: linear-gradient(135deg, var(--accent), var(--accent2)); }

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
  border: 1px solid var(--border);
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
  overflow-x: auto; margin: 10px 0; border: 1px solid var(--border);
}
.bubble.bot pre code { font-size: 12px; }
.bubble.bot blockquote {
  border-left: 3px solid var(--accent);
  padding-left: 12px; color: var(--muted); margin: 8px 0;
}
.bubble.bot table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }
.bubble.bot th { background: rgba(59,130,246,.15); padding: 8px 12px; text-align: left; }
.bubble.bot td { padding: 7px 12px; border-bottom: 1px solid var(--border); }

.msg-time { font-size: 10px; color: var(--muted); margin-top: 5px; padding: 0 4px; }

#typing { display: none; align-self: flex-start; gap: 12px; align-items: center; }
#typing.visible { display: flex; }
.typing-dots {
  background: var(--bot-bg); border: 1px solid var(--border);
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

#input-area {
  padding: 16px 24px 20px;
  border-top: 1px solid var(--border);
  display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap;
  flex-shrink: 0;
}
#file-attach-area {
  width: 100%; display: none; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;
}
#file-attach-area.visible { display: flex; }
.file-chip {
  background: var(--bg); border: 1px solid var(--border);
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
  background: var(--panel); border: 1px solid var(--border);
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
#send-btn { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
#send-btn:hover:not(:disabled) { transform: scale(1.05); }
#send-btn:disabled { opacity: .4; cursor: not-allowed; }
.action-btn svg { width: 20px; height: 20px; fill: currentColor; }

#welcome {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 12px; color: var(--muted); text-align: center; padding: 40px;
}
#welcome h2 { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: var(--text); }
#welcome p  { font-size: 13px; line-height: 1.8; max-width: 400px; }
#welcome .hint { font-size: 11px; color: #334155; margin-top: 12px; }

#claw-view {
  padding: 24px;
  gap: 20px;
  overflow-y: auto;
}

.claw-header {
  display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
}
.claw-header h2 {
  font-family: 'Syne', sans-serif;
  font-size: 22px; font-weight: 800;
  background: linear-gradient(135deg, var(--accent3), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.badge {
  font-size: 11px; padding: 4px 10px; border-radius: 20px;
  background: rgba(139,92,246,.15); color: var(--accent3);
  border: 1px solid rgba(139,92,246,.3);
  font-family: 'JetBrains Mono', monospace;
}

.claw-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
@media (max-width: 900px) { .claw-grid { grid-template-columns: 1fr; } }

.claw-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  display: flex; flex-direction: column; gap: 12px;
}
.claw-card label {
  font-size: 11px; font-weight: 600; letter-spacing: 1px;
  text-transform: uppercase; color: var(--muted);
}
#claw-obj {
  flex: 1; min-height: 120px;
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; color: var(--text);
  font-family: 'JetBrains Mono', monospace; font-size: 13px;
  padding: 12px; resize: vertical; outline: none; transition: border-color .2s;
}
#claw-obj:focus { border-color: var(--accent3); }

#claw-file-zone {
  border: 2px dashed var(--border); border-radius: 10px;
  padding: 20px; text-align: center; font-size: 12px; color: var(--muted);
  cursor: pointer; transition: all .2s; position: relative;
}
#claw-file-zone:hover, #claw-file-zone.drag-over {
  border-color: var(--accent3); color: var(--accent3);
  background: rgba(139,92,246,.05);
}
#claw-file-zone input[type="file"] {
  position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%;
}
.file-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.file-tag {
  background: rgba(139,92,246,.1); border: 1px solid rgba(139,92,246,.3);
  color: var(--accent3); border-radius: 6px;
  padding: 4px 10px; font-size: 11px;
  display: flex; align-items: center; gap: 6px;
}
.file-tag span {
  cursor: pointer; color: var(--muted); font-size: 14px; transition: color .2s;
}
.file-tag span:hover { color: var(--danger); }

.claw-logs {
  flex: 1; min-height: 280px; max-height: 420px;
  background: #060a14; border: 1px solid var(--border);
  border-radius: 8px; padding: 14px;
  overflow-y: auto; font-size: 12px; line-height: 1.8;
  font-family: 'JetBrains Mono', monospace;
}
.log-line { padding: 1px 0; }
.log-info { color: #7dd3fc; }
.log-ok   { color: var(--success); }
.log-err  { color: var(--danger); }
.log-warn { color: var(--warn); }
.log-dim  { color: #334155; }
.log-phase { color: var(--accent3); font-weight: 600; }
.log-retry { color: var(--warn); font-style: italic; }

.claw-progress {
  height: 4px; border-radius: 2px;
  background: var(--border); overflow: hidden;
}
.claw-progress-bar {
  height: 100%; border-radius: 2px;
  background: linear-gradient(90deg, var(--accent3), var(--accent2));
  width: 0%; transition: width .4s ease;
}

#claw-actions {
  display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
}

.claw-output {
  background: var(--panel);
  border: 1px solid var(--success);
  border-radius: var(--radius);
  overflow: hidden;
  animation: fadeIn .4s ease;
}
.claw-output-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}
.claw-output-header span { color: var(--success); font-weight: 600; }
.claw-output-meta {
  padding: 8px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  color: var(--muted);
  display: flex; gap: 16px;
}
.claw-output-meta em { color: var(--accent2); font-style: normal; }
.copy-btn {
  padding: 6px 14px; border-radius: 6px;
  background: rgba(34,197,94,.1); border: 1px solid rgba(34,197,94,.3);
  color: var(--success); font-size: 11px;
  cursor: pointer; transition: all .2s; font-family: 'JetBrains Mono', monospace;
}
.copy-btn:hover { background: rgba(34,197,94,.2); }
#claw-code {
  padding: 16px !important; margin: 0 !important;
  border-radius: 0 !important; font-size: 12px !important;
  max-height: 500px; overflow-y: auto;
}

#toast {
  position: fixed; bottom: 30px; right: 30px;
  background: var(--panel); border: 1px solid var(--border);
  color: var(--text); padding: 12px 20px; border-radius: 10px;
  font-size: 13px; transform: translateY(80px); opacity: 0;
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
    <h1>AI Chat<span>Powered by OpenRouter</span></h1>
  </div>

  <div>
    <div class="section-label">Modèle IA</div>
    <select id="model-select">
      {% for m in models %}
      <option value="{{ m.id }}" {% if m.id == default_model %}selected{% endif %}>{{ m.label }}</option>
      {% endfor %}
    </select>
  </div>

  <div>
    <div class="section-label">Contexte / Persona</div>
    <div class="field-group">
      <label class="field-label">Comportement de l'IA</label>
      <textarea id="system-prompt" placeholder="Ex: Tu es un expert Python. Réponds avec des exemples concrets…"></textarea>
      <button class="btn btn-primary btn-full" onclick="applyContext()">✓ Appliquer le contexte</button>
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
    <div class="nav-tabs">
      <button class="nav-tab active" data-view="chat" onclick="switchView('chat')">💬 Chat</button>
      <button class="nav-tab claw-tab" data-view="claw" onclick="switchView('claw')">🐾 CLAW</button>
    </div>
    <div id="topbar-right">
      <span id="skills-indicator">✨ Skills actifs</span>
      <span id="context-indicator" style="display:none">Contexte actif</span>
      <span id="model-badge">{{ default_model }}</span>
    </div>
  </div>

  <!-- CHAT VIEW -->
  <div id="chat-view" class="view-panel active">
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

  <!-- CLAW VIEW -->
  <div id="claw-view" class="view-panel">

    <div class="claw-header">
      <h2>🐾 CLAW Engine v2</h2>
      <span class="badge">Implémentation → Débogage → Optimisation + Validation syntaxique</span>
    </div>

    <div class="claw-grid">
      <div class="claw-card">
        <label>🎯 Objectif</label>
        <textarea id="claw-obj"
          placeholder="Ex: Convertis ce script Python en FastAPI avec gestion d'erreurs, Swagger et tests Pytest."></textarea>

        <label>📎 Fichiers sources</label>
        <div id="claw-file-zone"
             ondragover="event.preventDefault();this.classList.add('drag-over')"
             ondragleave="this.classList.remove('drag-over')"
             ondrop="handleClawDrop(event)">
          <input type="file" id="claw-files" multiple onchange="handleClawFiles(this.files)"/>
          📂 Glisse tes fichiers ici ou clique pour en sélectionner
          <div class="file-list" id="claw-file-list"></div>
        </div>

        <div id="claw-actions">
          <button class="btn btn-ghost" onclick="clearClaw()">↺ Réinitialiser</button>
          <button id="claw-run-btn" class="btn btn-secondary" onclick="runClaw()">
            🚀 Lancer la boucle CLAW
          </button>
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
        <button class="copy-btn" onclick="copyClawCode()">📋 Copier</button>
      </div>
      <div class="claw-output-meta" id="claw-output-meta"></div>
      <pre><code id="claw-code" class="hljs"></code></pre>
    </div>

  </div>

</div>

<div id="toast"></div>

<script>
/* =========================================================
   GLOBALS
   ========================================================= */
let hasContext   = false;
let skills       = [];
let attachedFiles = [];
let clawFiles    = [];
let clawRunning  = false;

/* =========================================================
   VIEW SWITCHING
   ========================================================= */
function switchView(view) {
  document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(view + '-view').classList.add('active');
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.view === view);
  });
}

/* =========================================================
   TEXTAREA AUTO-RESIZE + ENTER
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
   CONTEXT
   ========================================================= */
async function applyContext() {
  const prompt = document.getElementById('system-prompt').value.trim();
  const model  = document.getElementById('model-select').value;
  const res = await fetch('/api/context', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system_prompt: prompt, model, skills })
  });
  if (res.ok) {
    hasContext = !!prompt;
    document.getElementById('context-indicator').style.display = hasContext ? 'flex' : 'none';
    document.getElementById('stat-ctx').textContent = hasContext ? 'Oui' : 'Non';
    document.getElementById('model-badge').textContent = model;
    showToast(hasContext ? '✓ Contexte appliqué' : '✓ Contexte effacé', 'success');
  } else {
    showToast('❌ Erreur lors de l\'application', 'error');
  }
}

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
   CHAT FILE HANDLING
   ========================================================= */
function handleFileSelect(event) {
  Array.from(event.target.files).forEach(f => {
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
   SEND CHAT MESSAGE
   ========================================================= */
async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text && attachedFiles.length === 0) return;

  msgInput.value = '';
  msgInput.style.height = 'auto';
  document.getElementById('welcome')?.remove();

  let display = text || '[Fichier joint sans texte]';
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
  time.textContent = new Date().toLocaleTimeString('fr-FR', { hour:'2-digit', minute:'2-digit' });
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
    href: URL.createObjectURL(new Blob([md], {type:'text/markdown'})),
    download: 'conversation_' + Date.now() + '.md'
  });
  a.click();
  showToast('✓ Exporté', 'success');
}

/* =========================================================
   CLAW — FILE HANDLING
   ========================================================= */
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
      <div class="file-tag">
        📄 ${escHtml(f.name)}
        <span onclick="removeClawFile(${i})" title="Supprimer">✕</span>
      </div>`).join('');
}
function removeClawFile(i) {
  clawFiles.splice(i, 1);
  renderClawFiles();
}

/* =========================================================
   CLAW — MAIN LOOP
   ========================================================= */
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
  runBtn.disabled = true;
  runBtn.textContent = '⏳ En cours…';

  const logsEl = document.getElementById('claw-logs');
  logsEl.innerHTML = '';
  document.getElementById('claw-output-area').style.display = 'none';
  document.getElementById('claw-progress-wrap').style.display = 'block';
  setClawProgress(0);

  const fd = new FormData();
  fd.append('objective', obj);
  clawFiles.forEach(f => fd.append('claw_files', f));

  try {
    const res  = await fetch('/api/claw/process', { method: 'POST', body: fd });
    const data = await res.json();

    if (data.error) {
      clawLog('❌ ' + data.error, 'err');
      showToast('❌ Erreur CLAW : ' + data.error, 'error');
      return;
    }

    // Render logs with richer type detection
    (data.logs || []).forEach(line => {
      let type = 'info';
      if (!line.trim()) { clawLog('', 'dim'); return; }
      if (line.includes('❌'))            type = 'err';
      else if (line.includes('✅') || line.includes('🎯') || line.includes('🏁')) type = 'ok';
      else if (line.includes('⚠️') || line.includes('🔄')) type = 'warn';
      else if (line.includes('══'))       type = 'phase';
      else if (line.includes('↩️') || line.includes('retry')) type = 'retry';
      else if (line.startsWith('   '))    type = 'dim';
      clawLog(line, type);
    });

    setClawProgress(100);

    if (data.final_code) {
      const codeEl = document.getElementById('claw-code');
      codeEl.textContent = data.final_code;
      hljs.highlightElement(codeEl);

      // Show metadata
      const meta = document.getElementById('claw-output-meta');
      meta.innerHTML = [
        data.iterations  ? `<span>Itérations : <em>${data.iterations}</em></span>` : '',
        data.char_count  ? `<span>Taille : <em>${data.char_count.toLocaleString()} caractères</em></span>` : '',
        data.validated   ? `<span>Syntaxe : <em>✅ valide</em></span>` : `<span>Syntaxe : <em>⚠️ non validée</em></span>`,
        data.retries > 0 ? `<span>Retentatives : <em>${data.retries}</em></span>` : '',
      ].filter(Boolean).join('');

      document.getElementById('claw-output-area').style.display = 'block';
      showToast('✅ Boucle CLAW terminée !', 'success');
    } else {
      showToast('⚠️ Boucle terminée sans code final', 'warn');
    }

  } catch (e) {
    clawLog('❌ Erreur réseau : ' + e.message, 'err');
    showToast('❌ Erreur réseau', 'error');
  } finally {
    clawRunning = false;
    runBtn.disabled = false;
    runBtn.textContent = '🚀 Lancer la boucle CLAW';
  }
}

function clearClaw() {
  document.getElementById('claw-obj').value = '';
  clawFiles = [];
  renderClawFiles();
  document.getElementById('claw-logs').innerHTML = '<span class="log-dim">En attente de lancement…</span>';
  document.getElementById('claw-output-area').style.display = 'none';
  document.getElementById('claw-progress-wrap').style.display = 'none';
  setClawProgress(0);
  showToast('🔄 Réinitialisé', 'success');
}

function copyClawCode() {
  navigator.clipboard.writeText(document.getElementById('claw-code').textContent)
    .then(() => showToast('📋 Code copié !', 'success'))
    .catch(() => showToast('❌ Impossible de copier', 'error'));
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
      document.getElementById('context-indicator').style.display = 'flex';
      document.getElementById('stat-ctx').textContent = 'Oui';
      hasContext = true;
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


# ---------------------------------------------------------------------------
# Helpers — session
# ---------------------------------------------------------------------------

def _get_session() -> dict[str, Any]:
    """Return (and create if needed) the store entry for the current session."""
    sid = session.get("id")
    if not sid or sid not in store:
        sid = str(uuid.uuid4())
        session["id"] = sid
        store[sid] = {
            "messages":      [],
            "system_prompt": "",
            "model":         DEFAULT_MODEL,
            "skills":        [],
        }
    return store[sid]


def _build_system_content(sess: dict) -> str:
    """Build the full system prompt, prepending skills if any."""
    parts: list[str] = []
    if sess.get("skills"):
        parts.append(f"Skills and areas of expertise: {', '.join(sess['skills'])}.")
    if sess.get("system_prompt"):
        parts.append(sess["system_prompt"])
    return "\n\n".join(parts)


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate (1 token ≈ 4 chars)."""
    return sum(len(m.get("content", "")) for m in messages) // 4


# ---------------------------------------------------------------------------
# Helpers — LLM calls
# ---------------------------------------------------------------------------

_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer":  "https://localhost",
    "X-Title":       "AI Chatbot CLAW",
    "Content-Type":  "application/json",
}

# Placeholder patterns that indicate the model truncated its output
_PLACEHOLDER_PATTERNS = re.compile(
    r"(#\s*(TODO|FIXME|\.\.\.)|"
    r"pass\s*#\s*(implement|todo)|"
    r"\.{3}\s*\n|"
    r"# rest of (the )?(code|implementation)|"
    r"# similar to above|"
    r"# \.\.\. existing code|"
    r"\[suite du code\]|"
    r"\[contenu identique\])",
    re.IGNORECASE,
)


def _call_llm_raw(messages: list[dict], model: str, timeout: int = 90) -> str | None:
    """Single LLM call. Returns content string or None on failure."""
    payload = {"model": model, "messages": messages, "max_tokens": 65000}
    try:
        resp = requests.post(OPENROUTER_URL, headers=_HEADERS, json=payload, timeout=timeout)
        resp.raise_for_status()
        data    = resp.json()
        choices = data.get("choices", [])
        if not choices:
            logger.warning("No choices in response: %s", data)
            return None
        content = choices[0].get("message", {}).get("content")
        return content.strip() if content else None
    except requests.exceptions.Timeout:
        logger.error("LLM call timed out")
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("LLM request error: %s", exc)
        return None
    except Exception as exc:
        logger.error("Unexpected LLM error: %s", exc)
        return None


def _call_llm_with_retry(
    messages: list[dict],
    model: str,
    max_retries: int = 2,
    timeout: int = 90,
) -> tuple[str | None, int]:
    """
    Call the LLM with automatic retry on empty / failed responses.
    Returns (content, retries_used).
    """
    retries = 0
    for attempt in range(max_retries + 1):
        result = _call_llm_raw(messages, model, timeout)
        if result:
            return result, retries
        retries += 1
        if attempt < max_retries:
            logger.info("Retry %d/%d after failed LLM call", attempt + 1, max_retries)
            time.sleep(1.5 * (attempt + 1))
    return None, retries


def _call_llm(messages: list[dict], model: str) -> str:
    """Convenience wrapper used by chat endpoint — single call, safe fallback."""
    result, _ = _call_llm_with_retry(messages, model, max_retries=2)
    if result:
        return result
    return "ℹ️ Le modèle n'a généré aucun texte. Réessaie."


# ---------------------------------------------------------------------------
# Helpers — code utilities
# ---------------------------------------------------------------------------

def _clean_code_block(text: str) -> str:
    """
    Extract code from markdown fences intelligently.
    - Prefers the largest single code block (most likely the full implementation).
    - Falls back to concatenation if multiple blocks of similar size.
    - Returns raw text if no fences found.
    """
    blocks = re.findall(r"```(?:\w*)\n?([\s\S]*?)```", text)
    if not blocks:
        return text.strip()
    if len(blocks) == 1:
        return blocks[0].strip()
    # Return the largest block (most likely the full implementation)
    return max(blocks, key=len).strip()


def _validate_python_syntax(code: str) -> tuple[bool, str]:
    """
    Try to parse code as Python AST.
    Returns (is_valid, error_message).
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"Ligne {exc.lineno}: {exc.msg}"
    except Exception as exc:
        return False, str(exc)


def _has_placeholders(code: str) -> bool:
    """Return True if the code contains obvious truncation placeholders."""
    return bool(_PLACEHOLDER_PATTERNS.search(code))


def _is_code_shorter(new_code: str, old_code: str, threshold: float = 0.70) -> bool:
    """
    Return True if new_code is suspiciously shorter than old_code.
    Threshold: new must be at least 70 % of old length to be accepted.
    """
    if not old_code or len(old_code) < 200:
        return False
    return len(new_code) < len(old_code) * threshold


# ---------------------------------------------------------------------------
# Routes — shared
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    return render_template_string(HTML_TEMPLATE, models=FREE_MODELS, default_model=DEFAULT_MODEL)


@app.route("/api/context", methods=["POST"])
def set_context():
    data = request.get_json(force=True)
    sess = _get_session()
    sess["system_prompt"] = data.get("system_prompt", "").strip()
    if data.get("model"):
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
            file_blocks.append(f"📄 **{f.filename}**: [Erreur de lecture]")

    full_msg = user_msg
    if file_blocks:
        full_msg += "\n\n" + "\n\n".join(file_blocks)

    sess["messages"].append({"role": "user", "content": full_msg})

    payload_messages: list[dict] = []
    sys_content = _build_system_content(sess)
    if sys_content:
        payload_messages.append({"role": "system", "content": sys_content})
    payload_messages.extend(sess["messages"])

    payload = {"model": sess["model"], "messages": payload_messages, "max_tokens": 65000}
    try:
        response = requests.post(OPENROUTER_URL, headers=_HEADERS, json=payload, timeout=90)
        result   = response.json()
    except requests.RequestException as exc:
        logger.error("OpenRouter request failed: %s", exc)
        sess["messages"].pop()
        return jsonify({"error": "Erreur réseau lors de l'appel à l'API"}), 502

    if "error" in result:
        msg = result["error"].get("message", "Erreur inconnue")
        sess["messages"].pop()
        return jsonify({"error": msg}), 502

    reply: str = result["choices"][0]["message"]["content"]
    sess["messages"].append({"role": "assistant", "content": reply})

    return jsonify({
        "reply":            reply,
        "total_messages":   len(sess["messages"]),
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
        "messages":         sess["messages"],
        "system_prompt":    sess["system_prompt"],
        "model":            sess["model"],
        "skills":           sess.get("skills", []),
        "estimated_tokens": _estimate_tokens(sess["messages"]),
    })


# ---------------------------------------------------------------------------
# CLAW ENGINE v2 — improved pipeline
# ---------------------------------------------------------------------------

# Each iteration has a clearly specialised focus
_ITER_FOCUS = {
    1: "implémentation initiale complète et fonctionnelle",
    2: "correction des bugs, cas limites et robustesse",
    3: "optimisation, lisibilité et qualité finale",
}

# System prompts per role
_SYS_ARCHITECT = (
    "Tu es un architecte logiciel senior. "
    "Produis un plan d'action PRÉCIS et STRUCTURÉ pour atteindre l'objectif. "
    "Liste les modules, fonctions clés, dépendances et cas limites à gérer. "
    "Sois concis mais complet — maximum 30 lignes."
)

_SYS_CRITIC = (
    "Tu es un expert en revue de code et architecture. "
    "Identifie les failles du plan fourni : cas limites oubliés, "
    "problèmes de performance, failles de sécurité, erreurs de logique. "
    "Retourne UNIQUEMENT une liste numérotée de problèmes concrets à corriger. "
    "Maximum 15 points."
)

_SYS_DEV = (
    "Tu es un développeur expert. Génère UNIQUEMENT le code complet et fonctionnel. "
    "RÈGLES ABSOLUES :\n"
    "1. Inclus la totalité du code — aucun placeholder, aucun '# TODO', aucun '...'\n"
    "2. Si tu modifies du code existant, reproduis l'intégralité du fichier\n"
    "3. Ne résume jamais une section existante\n"
    "4. Utilise des blocs ```python ... ``` pour délimiter le code\n"
    "5. Le code doit être prêt à l'exécution sans modification"
)

_SYS_FIX_SYNTAX = (
    "Tu es un développeur expert. Le code fourni contient des erreurs de syntaxe. "
    "Corrige-les et retourne UNIQUEMENT le code corrigé et complet dans un bloc ```python. "
    "Ne supprime aucune fonctionnalité existante."
)

_SYS_IMPROVE = (
    "Tu es un expert senior en qualité logicielle. "
    "Analyse ce code et produis une liste COURTE (max 10 points) "
    "d'améliorations concrètes et prioritaires pour la prochaine itération. "
    "Format : numéroté, une ligne par point."
)


@app.route("/api/claw/process", methods=["POST"])
def claw_process():
    """
    CLAW v2 — Smarter 3-iteration loop:
      Iter 1 → Initial implementation
      Iter 2 → Bug fixing & edge cases
      Iter 3 → Optimisation & polish
    Each iteration: Plan → Critique → Implement (with syntax validation + retry) → Improve
    """
    sess      = _get_session()
    objective = request.form.get("objective", "").strip()
    if not objective:
        return jsonify({"error": "Veuillez définir un objectif clair."}), 400

    model = sess["model"]

    # ── Load uploaded files ──────────────────────────────────
    files: list[str] = []
    for f in request.files.getlist("claw_files"):
        if not f.filename:
            continue
        try:
            content = f.read().decode("utf-8", errors="replace")
            if len(content) > 30_000:
                content = content[:30_000] + "\n\n…[tronqué à 30 000 caractères]"
            files.append(f"### {f.filename}\n{content}")
        except Exception as exc:
            logger.warning("CLAW file read error: %s", exc)

    current_code: str = (
        "\n\n".join(files)
        if files
        else "# Aucun code initial — génère la structure complète depuis zéro."
    )

    # ── State ────────────────────────────────────────────────
    logs: list[str]         = []
    improvements: list[str] = []   # structured feedback from previous iteration
    total_retries: int      = 0
    validated: bool         = False
    success: bool           = True

    def log(msg: str = "") -> None:
        logs.append(msg)
        logger.info("CLAW | %s", msg)

    # ── Main loop ────────────────────────────────────────────
    for iteration in range(1, 4):
        focus = _ITER_FOCUS[iteration]
        log()
        log(f"══ Itération {iteration}/3 — {focus} ══")

        # ── 1. Plan ──────────────────────────────────────────
        log("   📝 Planification…")
        improvement_context = (
            "\n".join(f"  - {p}" for p in improvements)
            if improvements else "Aucune (première itération)"
        )
        plan, retries = _call_llm_with_retry([
            {"role": "system", "content": _SYS_ARCHITECT},
            {"role": "user", "content": (
                f"OBJECTIF: {objective}\n\n"
                f"FOCUS DE CETTE ITÉRATION: {focus}\n\n"
                f"CODE ACTUEL ({len(current_code)} caractères):\n{current_code}\n\n"
                f"POINTS À CORRIGER/AMÉLIORER (itérations précédentes):\n{improvement_context}"
            )},
        ], model)
        total_retries += retries

        if not plan:
            log("   ❌ Planification échouée après plusieurs tentatives")
            success = False
            break
        log("   ✅ Plan établi")
        log()

        # ── 2. Critique ──────────────────────────────────────
        log("   🔍 Critique du plan…")
        critique, retries = _call_llm_with_retry([
            {"role": "system", "content": _SYS_CRITIC},
            {"role": "user", "content": (
                f"OBJECTIF: {objective}\n\n"
                f"PLAN À CRITIQUER:\n{plan}\n\n"
                f"CODE ACTUEL:\n{current_code}"
            )},
        ], model)
        total_retries += retries

        if not critique:
            log("   ⚠️  Critique indisponible — on continue avec le plan tel quel")
            critique = "(pas de critique disponible)"
        else:
            log("   ✅ Critique prête")

        # ── 3. Implement + validate ───────────────────────────
        log("   ⚙️  Implémentation du code…")
        raw_code, retries = _call_llm_with_retry([
            {"role": "system", "content": _SYS_DEV},
            {"role": "user", "content": (
                f"OBJECTIF: {objective}\n\n"
                f"FOCUS: {focus}\n\n"
                f"PLAN VALIDÉ:\n{plan}\n\n"
                f"CRITIQUES À INTÉGRER:\n{critique}\n\n"
                f"CODE ACTUEL À AMÉLIORER/COMPLÉTER ({len(current_code)} caractères):\n"
                f"{current_code}\n\n"
                f"RAPPEL: Génère la totalité du code, sans troncature ni placeholder."
            )},
        ], model)
        total_retries += retries

        if not raw_code:
            log("   ❌ Implémentation échouée après plusieurs tentatives")
            success = False
            break

        candidate = _clean_code_block(raw_code)

        # Guard: reject suspiciously short output (regression)
        if _is_code_shorter(candidate, current_code):
            log("   ⚠️  Code généré trop court — probable troncature, on conserve la version précédente")
            log(f"      (ancien: {len(current_code)} chars / nouveau: {len(candidate)} chars)")
        else:
            # Guard: detect placeholder truncation
            if _has_placeholders(candidate):
                log("   ⚠️  Placeholders détectés (code incomplet) — tentative de correction…")
                fixed, retries2 = _call_llm_with_retry([
                    {"role": "system", "content": _SYS_DEV},
                    {"role": "user", "content": (
                        f"Le code suivant contient des placeholders comme '# TODO', '...', '# rest of code'.\n"
                        f"Remplace-les par une implémentation réelle complète.\n\n"
                        f"OBJECTIF: {objective}\n\n"
                        f"CODE INCOMPLET:\n{candidate}"
                    )},
                ], model)
                total_retries += retries2
                if fixed and not _is_code_shorter(_clean_code_block(fixed), current_code):
                    candidate = _clean_code_block(fixed)
                    log("   🔄 Code complété après correction des placeholders")

            current_code = candidate
            log(f"   🎯 Code généré ({len(current_code):,} caractères)")

        # Syntax validation (Python only)
        validated, syntax_error = _validate_python_syntax(current_code)
        if validated:
            log("   ✅ Syntaxe Python valide")
        else:
            log(f"   ⚠️  Erreur de syntaxe : {syntax_error} — tentative de correction…")
            fixed_syntax, retries3 = _call_llm_with_retry([
                {"role": "system", "content": _SYS_FIX_SYNTAX},
                {"role": "user", "content": (
                    f"ERREUR: {syntax_error}\n\n"
                    f"CODE À CORRIGER:\n{current_code}"
                )},
            ], model)
            total_retries += retries3
            if fixed_syntax:
                candidate2 = _clean_code_block(fixed_syntax)
                ok2, _ = _validate_python_syntax(candidate2)
                if ok2:
                    current_code = candidate2
                    validated    = True
                    log("   ✅ Syntaxe corrigée avec succès")
                else:
                    log("   ❌ Correction syntaxique échouée — on conserve le code actuel")
            else:
                log("   ❌ Impossible d'obtenir une correction syntaxique")

        # ── 4. Improvement analysis ───────────────────────────
        log("   🔬 Analyse des améliorations pour la prochaine itération…")
        improve_raw, retries = _call_llm_with_retry([
            {"role": "system", "content": _SYS_IMPROVE},
            {"role": "user", "content": (
                f"OBJECTIF: {objective}\n\n"
                f"CODE (itération {iteration}):\n{current_code}"
            )},
        ], model)
        total_retries += retries

        if improve_raw:
            # Parse numbered list into structured items
            improvements = [
                line.lstrip("0123456789.-) ").strip()
                for line in improve_raw.splitlines()
                if line.strip() and line.strip()[0].isdigit()
            ][:10]
            if improvements:
                log(f"   🧠 {len(improvements)} point(s) d'amélioration identifiés")
            else:
                log("   ⚠️  Suggestions non structurées (ignorées)")
        else:
            log("   ⚠️  Aucune suggestion d'amélioration obtenue")

        log(f"✅ Itération {iteration}/3 terminée")

    # ── Final result ─────────────────────────────────────────
    log()
    if success:
        log("🏁 Boucle CLAW v2 terminée avec succès !")
        log(f"📦 Code final : {len(current_code):,} caractères")
        if validated:
            log("🟢 Syntaxe Python validée")
        if total_retries:
            log(f"🔄 Retentatives totales : {total_retries}")
    else:
        log("⚠️  Boucle CLAW interrompue suite à une erreur.")

    final_code = (
        current_code
        if current_code and "Aucun code initial" not in current_code
        else None
    )

    return jsonify({
        "logs":       logs,
        "final_code": final_code,
        "success":    success,
        "iterations": 3 if success else None,
        "char_count": len(final_code) if final_code else 0,
        "validated":  validated,
        "retries":    total_retries,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 AI Chatbot + CLAW v3")
    # Render donne un port dans la variable PORT, sinon on prend 5000 en local
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
