from __future__ import annotations

import ast
import json
import re
import logging
from typing import List, Dict
import time
import uuid
import zipfile
import io
from collections import Counter
from typing import Any
import os
import requests
from flask import Flask, jsonify, render_template_string, request, session, send_file, Response
from datetime import datetime
import urllib.parse
import threading

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = "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"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "minimax/minimax-m2.5:free"

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 50 * 1024 * 1024
app.secret_key = "change-me-in-production-use-os-urandom"

store: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Available free models on OpenRouter
# ---------------------------------------------------------------------------
FREE_MODELS: list[dict[str, str]] = [
    {"id": "minimax/minimax-m2.5:free", "label": "MiniMax M2.5"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B"},
    {"id": "stepfun/step-3.5-flash:free", "label": "step 3.5"},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free", "label": "nemotron 3"},
]

# ---------------------------------------------------------------------------
# Système de tâches CLAW avec SSE
# ---------------------------------------------------------------------------
claw_tasks = {}  # Dictionnaire global pour stocker les tâches
claw_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Language Detection & Mapping
# ---------------------------------------------------------------------------
LANGUAGE_EXTENSIONS = {
    '.py': 'python', '.pyw': 'python', '.ipynb': 'python',
    '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
    '.ts': 'typescript', '.tsx': 'typescript',
    '.java': 'java',
    '.c': 'c', '.h': 'c',
    '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp', '.hxx': 'cpp',
    '.cs': 'csharp',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    '.m': 'objective-c', '.mm': 'objective-c',
    '.dart': 'dart',
    '.lua': 'lua',
    '.r': 'r',
    '.jl': 'julia',
    '.pl': 'perl',
    '.pm': 'perl',
    '.sh': 'bash', '.bash': 'bash',
    '.ps1': 'powershell',
    '.sql': 'sql',
    '.html': 'html', '.htm': 'html',
    '.css': 'css',
    '.scss': 'scss', '.sass': 'sass',
    '.less': 'less',
    '.xml': 'xml',
    '.json': 'json',
    '.yaml': 'yaml', '.yml': 'yaml',
    '.toml': 'toml',
    '.ini': 'ini',
    '.cfg': 'cfg',
    '.md': 'markdown',
    '.rst': 'rst',
    '.tex': 'latex',
    '.dockerfile': 'dockerfile',
    '.make': 'makefile', '.mk': 'makefile',
    '.gradle': 'gradle',
}

LANGUAGE_KEYWORDS = {
    'python': ['def ', 'class ', 'import ', 'from ', 'async ', 'await ', 'with '],
    'javascript': ['function ', 'const ', 'let ', 'var ', '=>', 'async ', 'await '],
    'typescript': ['interface ', 'type ', 'enum ', 'implements '],
    'java': ['public class ', 'private ', 'protected ', 'static ', 'void '],
    'cpp': ['#include', 'namespace ', 'template ', 'typename '],
    'csharp': ['using ', 'namespace ', 'public ', 'private ', 'class '],
    'go': ['package ', 'func ', 'import ', 'type '],
    'rust': ['fn ', 'struct ', 'enum ', 'impl ', 'use '],
    'ruby': ['def ', 'class ', 'module ', 'require '],
    'php': ['<?php', 'function ', 'class ', 'namespace '],
    'sql': ['SELECT ', 'INSERT ', 'UPDATE ', 'DELETE ', 'CREATE TABLE'],
    'html': ['<!DOCTYPE', '<html', '<head', '<body', '<div'],
    'css': [':root', '{', '}', 'margin', 'padding', 'color'],
    'bash': ['#!/bin/bash', 'echo ', 'if [', 'for ', 'while '],
}


def _detect_language(files: list[str], objective: str = "") -> str:
    """Détecte le langage principal basé sur les extensions et mots-clés."""
    # Compter les extensions
    ext_counter = Counter()
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in LANGUAGE_EXTENSIONS:
            ext_counter[ext] += 1

    if ext_counter:
        most_common_ext = ext_counter.most_common(1)[0][0]
        return LANGUAGE_EXTENSIONS[most_common_ext]

    # Fallback: analyser l'objectif et le code
    objective_lower = objective.lower()
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in objective_lower:
                return lang

    # Chercher dans le code fourni
    all_content = " ".join(files).lower()
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in all_content:
                return lang

    return "python"  # Défaut


def _get_file_extension(language: str) -> str:
    """Retourne l'extension appropriée pour le langage."""
    ext_map = {
        'python': '.py',
        'javascript': '.js',
        'typescript': '.ts',
        'java': '.java',
        'c': '.c',
        'cpp': '.cpp',
        'csharp': '.cs',
        'go': '.go',
        'rust': '.rs',
        'ruby': '.rb',
        'php': '.php',
        'swift': '.swift',
        'kotlin': '.kt',
        'scala': '.scala',
        'objective-c': '.m',
        'dart': '.dart',
        'lua': '.lua',
        'r': '.r',
        'julia': '.jl',
        'perl': '.pl',
        'bash': '.sh',
        'powershell': '.ps1',
        'sql': '.sql',
        'html': '.html',
        'css': '.css',
        'scss': '.scss',
        'sass': '.sass',
        'less': '.less',
        'xml': '.xml',
        'json': '.json',
        'yaml': '.yaml',
        'toml': '.toml',
        'ini': '.ini',
        'markdown': '.md',
        'latex': '.tex',
        'dockerfile': 'Dockerfile',
        'makefile': 'Makefile',
        'gradle': 'build.gradle',
    }
    return ext_map.get(language, '.txt')


# ---------------------------------------------------------------------------
# System Prompts Adaptés par Langage
# ---------------------------------------------------------------------------
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

_SYS_DEV_TEMPLATE = (
    "Tu es un développeur expert en {language}. Génère UNIQUEMENT le code complet et fonctionnel. "
    "RÈGLES ABSOLUES :\n"
    "1. Inclus la totalité du code — aucun placeholder, aucun 'TODO', aucun '...'\n"
    "2. Si tu modifies du code existant, reproduis l'intégralité du fichier\n"
    "3. Ne résume jamais une section existante\n"
    "4. Utilise des blocs ```{language} ... ``` pour délimiter le code\n"
    "5. Le code doit être prêt à l'exécution sans modification\n"
    "6. Respecte les conventions de style du langage ({style_guide})\n"
    "7. Ajoute des docstrings/commentaires appropriés\n"
    "8. Gère les erreurs et cas limites\n"
    "9. Inclus les imports/dépendances nécessaires\n"
    "10. Structure le code de manière modulaire\n"
    "11. ⚠️ INTERDICTION STRICTE : Ne supprime JAMAIS de code fonctionnel existant. "
    "Si tu ajoutes des fonctionnalités, le code peut s'allonger. Si tu dois remplacer une section, "
    "assure-toi que la nouvelle version est équivalente ou supérieure en fonctionnalités. "
    "La suppression de code n'est autorisée que si elle est compensée par une implémentation "
    "équivalente ou meilleure dans la même passe.\n"
)

_SYS_DEV = {
    'python': _SYS_DEV_TEMPLATE.format(
        language='python',
        style_guide='PEP 8, type hints, context managers, dataclasses'
    ),
    'javascript': _SYS_DEV_TEMPLATE.format(
        language='javascript',
        style_guide='ES6+, const/let, arrow functions, async/await'
    ),
    'typescript': _SYS_DEV_TEMPLATE.format(
        language='typescript',
        style_guide='strict mode, interfaces, type safety'
    ),
    'java': _SYS_DEV_TEMPLATE.format(
        language='java',
        style_guide=' encapsulation, exceptions, Javadoc'
    ),
    'cpp': _SYS_DEV_TEMPLATE.format(
        language='cpp',
        style_guide='RAII, move semantics, header files'
    ),
    'csharp': _SYS_DEV_TEMPLATE.format(
        language='csharp',
        style_guide='async/await, nullable reference types, LINQ'
    ),
    'go': _SYS_DEV_TEMPLATE.format(
        language='go',
        style_guide='idiomatic Go, error handling, interfaces'
    ),
    'rust': _SYS_DEV_TEMPLATE.format(
        language='rust',
        style_guide='ownership, borrowing, Result/Option'
    ),
}

_SYS_FIX_SYNTAX = (
    "Tu es un développeur expert. Le code fourni contient des erreurs de syntaxe. "
    "Corrige-les et retourne UNIQUEMENT le code corrigé et complet dans un bloc ```{language}. "
    "Ne supprime aucune fonctionnalité existante. Respecte les conventions du langage."
)

_SYS_IMPROVE = (
    "Tu es un expert senior en qualité logicielle. "
    "Analyse ce code et produis une liste COURTE (max 10 points) "
    "d'améliorations concrètes et prioritaires pour la prochaine itération. "
    "Format : numéroté, une ligne par point."
)


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
            "messages": [],
            "system_prompt": "",
            "model": DEFAULT_MODEL,
            "skills": [],
            "claw_versions": [],  # Stocke les versions du code CLAW
            "claw_language": None,
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
    "HTTP-Referer": "https://localhost",
    "X-Title": "AI Chatbot CLAW",
    "Content-Type": "application/json",
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


def _call_llm_raw(messages: list[dict], model: str, timeout: int = 180) -> str | None:
    """Single LLM call. Returns content string or None on failure."""
    payload = {"model": model, "messages": messages, "max_tokens": 65000}
    try:
        resp = requests.post(OPENROUTER_URL, headers=_HEADERS, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
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
logger = logging.getLogger(__name__)


def perform_search(query: str, max_results: int = 5) -> str:
    """
    Effectue une recherche sur Internet (DuckDuckGo) et retourne les résultats.
    L'IA peut appeler cet outil avec \recherche <query>.

    Parameters
    ----------
    query : str
        Terme de recherche.
    max_results : int, optional
        Nombre maximal de résultats à renvoyer (défaut : 5).

    Returns
    -------
    str
        Texte formaté contenant les titres, URLs et extraits des résultats.
    """
    try:
        # ---------- Pré‑préparation de la requête ----------
        encoded_query = urllib.parse.quote(query)
        # DuckDuckGo Instant Answer API (JSON, sans HTML)
        ddg_url = "https://api.duckduckgo.com/"
        params: Dict[str, str] = {
            "q": encoded_query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }

        # ---------- Appel à l’API ----------
        response = requests.get(ddg_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # ---------- Construction du texte de sortie ----------
        results: List[str] = []

        # Informations globales (nombre approximatif de résultats)
        # DuckDuckGo ne renvoie pas toujours un compte exact ; on utilise AbstractText quand disponible.
        abstract = data.get("AbstractText", "").strip()
        abstract_src = data.get("AbstractSource", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()

        if abstract:
            results.append(
                f"🔎 **Résultat instantané** (source : {abstract_src or 'DuckDuckGo'})\n"
                f"{abstract}\n"
                f"🔗 {abstract_url}\n"
            )
        else:
            results.append(f"🔍 Recherche Internet pour « {query} » :\n")

        # Résultats classiques (RelatedTopics) – chaque entrée peut contenir un texte et une URL
        related: List[Dict] = data.get("RelatedTopics", [])
        count = 0

        for item in related:
            if count >= max_results:
                break

            # Certaines entrées sont simplement des catégories (sans 'FirstURL')
            if isinstance(item, dict) and "FirstURL" in item:
                title = item.get("Text", "Titre inconnu").split(" – ")[0]  # on garde le texte avant le tiret
                snippet = re.sub(r"<[^>]+>", "", item.get("Text", ""))  # nettoyage éventuel de balises
                url = item.get("FirstURL", "#")
                results.append(
                    f"{count + 1}. **{title}**\n"
                    f"{snippet[:200]}…\n"
                    f"🔗 {url}\n"
                )
                count += 1
            # Dans le cas où le sujet est une sous‑liste (ex. « RelatedTopics » contenant d’autres dicts)
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
        logger.error(f"Erreur lors de la recherche Internet : {e}")
        return f"❌ Erreur de connexion : {str(e)}"
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la recherche Internet : {e}")
        return f"❌ Erreur inattendue : {str(e)}"


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


def _is_code_shorter(new_code: str, old_code: str, threshold: float = 0.90) -> bool:
    """
    Return True if new_code is suspiciously shorter than old_code.
    Threshold: new must be at least 90% of old length to be accepted.
    """
    if not old_code or len(old_code) < 200:
        return False
    return len(new_code) < len(old_code) * threshold


def _compute_diff(old_code: str, new_code: str) -> str:
    """Compute a simple diff between two code versions."""
    import difflib
    diff = difflib.unified_diff(
        old_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile='previous',
        tofile='current',
        lineterm=''
    )
    return ''.join(diff)


def _generate_project_structure(code: str, language: str, objective: str) -> dict:
    """Generate a complete project structure with proper files."""
    ext = _get_file_extension(language)
    files = {}

    # Main file
    main_filename = f"main{ext}"
    files[main_filename] = code

    # Requirements/package files based on language
    if language == 'python':
        files['requirements.txt'] = _generate_requirements(code)
        files['README.md'] = _generate_readme(objective, language, code)
        files['.gitignore'] = "*.pyc\n__pycache__/\nvenv/\n.env\n"
    elif language in ['javascript', 'typescript']:
        files['package.json'] = _generate_package_json(code, language)
        files['README.md'] = _generate_readme(objective, language, code)
        files['.gitignore'] = "node_modules/\n.env\ndist/\nbuild/\n"
    elif language == 'java':
        files['pom.xml'] = _generate_maven_pom(code)
        files['README.md'] = _generate_readme(objective, language, code)
        files['.gitignore'] = "target/\n.idea/\n*.iml\n.vscode/\n"
    elif language == 'go':
        files['go.mod'] = _generate_go_mod(code)
        files['README.md'] = _generate_readme(objective, language, code)
        files['.gitignore'] = "bin/\nvendor/\n"

    return files


def _generate_requirements(code: str) -> str:
    """Generate Python requirements.txt from code imports."""
    imports = re.findall(r'^(?:from|import)\s+([\w\.]+)', code, re.MULTILINE)
    packages = set()
    for imp in imports:
        pkg = imp.split('.')[0]
        if pkg not in ['os', 'sys', 'json', 're', 'datetime', 'typing', 'collections', 'itertools', 'functools',
                       'pathlib']:
            packages.add(pkg)

    if not packages:
        return "# No external dependencies detected\n"

    return '\n'.join(sorted(packages)) + '\n'


def _generate_package_json(code: str, language: str) -> str:
    """Generate package.json for JS/TS projects."""
    pkg = {
        "name": "claw-generated-project",
        "version": "1.0.0",
        "description": "Generated by CLAW Engine",
        "main": f"main.{'js' if language == 'javascript' else 'ts'}",
        "scripts": {
            "start": f"node main.{'js' if language == 'javascript' else 'ts'}",
            "test": "echo \"Error: no test specified\" && exit 1"
        },
        "keywords": ["claw", "generated"],
        "author": "CLAW Engine",
        "license": "MIT"
    }
    return json.dumps(pkg, indent=2) + '\n'


def _generate_maven_pom(code: str) -> str:
    """Generate basic pom.xml for Java."""
    return """<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.claw</groupId>
  <artifactId>claw-project</artifactId>
  <version>1.0.0</version>
  <properties>
    <maven.compiler.source>11</maven.compiler.source>
    <maven.compiler.target>11</maven.compiler.target>
  </properties>
</project>
"""


def _generate_go_mod(code: str) -> str:
    """Generate go.mod for Go projects."""
    return """module claw-project

go 1.21
"""


def _generate_readme(objective: str, language: str, code: str) -> str:
    """Generate README.md for the project."""
    lines = [
        f"# CLAW Generated Project",
        "",
        f"**Language:** {language}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Objective",
        objective,
        "",
        "## Description",
        "This code was generated by CLAW Engine v2 through an iterative process:",
        "1. Initial implementation",
        "2. Bug fixing & edge cases",
        "3. Optimization & polish",
        "",
        "## Usage",
        f"See the main file (`main{_get_file_extension(language)}`) for the implementation.",
        "",
        "## Notes",
        f"- Total characters: {len(code):,}",
        f"- Lines of code: {len(code.splitlines()):,}",
    ]
    return '\n'.join(lines) + '\n'


def _create_zip(files: dict[str, str]) -> bytes:
    """Create a ZIP file from a dictionary of filename->content."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename, content in files.items():
            zipf.writestr(filename, content)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


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

    # Met à jour seulement les champs fournis
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

    # Récupération du message
    if request.is_json:
        data = request.get_json()
        user_msg = data.get("message", "").strip()
    else:
        user_msg = request.form.get("message", "").strip()

    # Gestion des fichiers joints
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

    # Construire le contexte système avec mention de l'outil de recherche
    sys_content = _build_system_content(sess)

    #  système qui présente l'outil de recherche à l'IA
    tool_instruction = (
        "\n\nOUTILS DISPONIBLES:\n"
        "Tu as accès à un outil de recherche internet . Pour l'utiliser, "
        "tappe exactement: \\recherche <ta question>\n"
        "Exemple: \\recherche comment créer une classe en Python\n"
        "Le système remplacera cette commande par les résultats de recherche "
        "pour que tu puisses rédiger une réponse complète et précise.\n"
        "IMPORTANT: Utilise cet outil pour les questions factuelles, techniques, "
        "ou qui nécessitent des informations à jour que tu ne connaisrais pas."
    )

    if sys_content:
        full_sys_content = sys_content + tool_instruction
    else:
        full_sys_content = tool_instruction.strip()

    # Ajouter le message utilisateur à l'historique
    sess["messages"].append({"role": "user", "content": full_msg})

    # Construire les messages pour l'API
    payload_messages = [{"role": "system", "content": full_sys_content}]
    payload_messages.extend(sess["messages"])

    # --- BOUCLE DE RECHERCHE ---
    # L'IA peut demander plusieurs recherches successives
    max_search_attempts = 5
    search_count = 0
    final_reply = None

    while search_count < max_search_attempts:
        # Appel à l'IA
        payload = {"model": sess["model"], "messages": payload_messages, "max_tokens": 65000}
        try:
            response = requests.post(OPENROUTER_URL, headers=_HEADERS, json=payload, timeout=90)
            result = response.json()
        except requests.RequestException as exc:
            logger.error(f"❌ Erreur de connexion OpenRouter: {exc}")
            sess["messages"].pop()  # Retirer le message utilisateur
            return jsonify({"error": "Problème de connexion au service AI"}), 502

        if "error" in result:
            err_data = result["error"]
            detailed_msg = err_data.get("message", "Erreur inconnue")
            err_code = err_data.get("code", "N/A")
            logger.error(f"🚨 OpenRouter Error [{err_code}]: {detailed_msg}")
            if "metadata" in err_data:
                logger.error(f"Metadata: {err_data['metadata']}")
            sess["messages"].pop()
            return jsonify({"error": f"IA indisponible ({err_code}): {detailed_msg}"}), 502

        reply: str = result["choices"][0]["message"]["content"]

        # Vérifier si l'IA demande une recherche
        if reply.strip().startswith('\\recherche '):
            search_count += 1
            query = reply.strip()[len('\\recherche '):].strip()

            if not query:
                # Si pas de requête, on continue sans faire de recherche
                payload_messages.append({"role": "assistant", "content": reply})
                continue

            logger.info(f"🔍 Recherche demandée par l'IA: {query}")

            # Effectuer la recherche
            search_results = perform_search(query)

            # Ajouter la commande de recherche à l'historique
            payload_messages.append({"role": "assistant", "content": reply})

            # Ajouter les résultats comme message système
            payload_messages.append({
                "role": "system",
                "content": f"RÉSULTATS DE RECHERCHE pour '{query}':\n{search_results}\n\nTu peux maintenant rédiger ta réponse finale basée sur ces résultats."
            })

            # Continuer la boucle pour que l'IA rédige sa réponse
            continue
        else:
            # Réponse finale (pas de recherche demandée)
            final_reply = reply
            payload_messages.append({"role": "assistant", "content": reply})
            break

    if final_reply is None:
        # Si on a atteint le maximum de recherches sans réponse finale
        if search_count >= max_search_attempts:
            error_msg = "⚠️ Limite de recherches atteinte. L'IA n'a pas fourni de réponse finale."
        else:
            error_msg = "❌ Erreur lors de la génération de la réponse."
        return jsonify({"reply": error_msg})

    # Mettre à jour l'historique de la session
    # On filtre pour ne garder que les messages user/assistant (pas les systèmes intermédiaires)
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
# CLAW ENGINE v3 — Enhanced with language detection, versioning, export, SSE
# ---------------------------------------------------------------------------

@app.route("/api/claw/process", methods=["POST"])
def claw_process():
    """
    Démarre une tâche CLAW en arrière-plan et retourne un task_id.
    """
    objective = request.form.get("objective", "").strip()
    if not objective:
        return jsonify({"error": "Veuillez définir un objectif clair."}), 400

    model = _get_session()["model"]

    # Charger les fichiers
    files = []
    file_names = []
    for f in request.files.getlist("claw_files"):
        if not f.filename:
            continue
        try:
            content = f.read().decode("utf-8", errors="replace")
            files.append(f"### {f.filename}\n{content}")
            file_names.append(f.filename)
        except Exception as exc:
            logger.warning("CLAW file read error: %s", exc)

    # Créer un task_id
    task_id = str(uuid.uuid4())
    with claw_lock:
        claw_tasks[task_id] = {
            'logs': [],
            'status': 'running',
            'progress': 0,
            'result': None,
            'error': None
        }

    # Lancer la tâche dans un thread séparé
    thread = threading.Thread(
        target=run_claw_task,
        args=(task_id, objective, files, file_names, model)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"task_id": task_id})


def run_claw_task(task_id, objective, files, file_names, model):
    """
    Exécute la boucle CLAW et stocke les résultats dans claw_tasks[task_id].
    """
    try:
        # Détection du langage
        language = _detect_language(file_names, objective)
        logger.info(f"CLAW: Detected language = {language}")

        current_code = "\n\n".join(
            files) if files else "# Aucun code initial — génère la structure complète depuis zéro."

        logs = []
        improvements = []
        total_retries = 0
        validated = False
        success = True
        versions = [current_code]

        def log(msg=""):
            logs.append(msg)
            # Stocker le log dans la tâche
            with claw_lock:
                claw_tasks[task_id]['logs'].append(msg)
                claw_tasks[task_id]['progress'] = min(99, int((len(versions) / 10) * 100))

        # Get language-specific system prompt
        sys_dev = _SYS_DEV.get(language, _SYS_DEV['python'])

        # ── Main loop ────────────────────────────────────────────
        max_iterations = 10  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Determine focus for this iteration
            if iteration == 1:
                focus = "implémentation initiale complète et fonctionnelle"
            elif iteration == 2:
                focus = "correction des bugs, cas limites et robustesse"
            elif iteration == 3:
                focus = "optimisation, lisibilité et qualité finale"
            else:
                focus = f"itération supplémentaire d'optimisation ({iteration})"

            log()
            log(f"══ Itération {iteration} — {focus} ══")

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
                    f"LANGAGE: {language}\n\n"
                    f"CODE ACTUEL ({len(current_code)} caractères):\n{current_code}\n\n"
                    f"POINTS À CORRIGER/AMÉLIORER (itérations précédentes):\n{improvement_context}\n\n"
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
                    f"LANGAGE: {language}\n\n"
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
                {"role": "system", "content": sys_dev},
                {"role": "user", "content": (
                    f"OBJECTIF: {objective}\n\n"
                    f"LANGAGE: {language}\n\n"
                    f"FOCUS: {focus}\n\n"
                    f"PLAN VALIDÉ:\n{plan}\n\n"
                    f"CRITIQUES À INTÉGRER:\n{critique}\n\n"
                    "‼️ RÈGLE CRITIQUE : NE SUPPRIME PAS DE CODE FONCTIONNEL EXISTANT.\n"
                    "Si le code actuel fonctionne, toute modification doit préserver toutes ses fonctionnalités.\n"
                    "Tu peux ajouter, refactoriser ou améliorer, mais jamais supprimer sans remplacement équivalent.\n"
                    "Le code final doit être AU MOINS AUSSI COMPLET que le code actuel.\n\n"
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
                        {"role": "system", "content": sys_dev},
                        {"role": "user", "content": (
                            f"Le code suivant contient des placeholders comme '# TODO', '...', '# rest of code'.\n"
                            f"Remplace-les par une implémentation réelle complète.\n\n"
                            f"OBJECTIF: {objective}\n\n"
                            f"LANGAGE: {language}\n\n"
                            f"CODE INCOMPLET:\n{candidate}"
                        )},
                    ], model)
                    total_retries += retries2
                    if fixed and not _is_code_shorter(_clean_code_block(fixed), current_code):
                        candidate = _clean_code_block(fixed)
                        log("   🔄 Code complété après correction des placeholders")

                current_code = candidate
                versions.append(current_code)
                log(f"   🎯 Code généré ({len(current_code):,} caractères)")

            # Syntax validation (Python only)
            if language == 'python':
                validated, syntax_error = _validate_python_syntax(current_code)
                if validated:
                    log("   ✅ Syntaxe Python valide")
                else:
                    log(f"   ⚠️  Erreur de syntaxe : {syntax_error} — tentative de correction…")
                    fixed_syntax, retries3 = _call_llm_with_retry([
                        {"role": "system", "content": _SYS_FIX_SYNTAX.format(language=language)},
                        {"role": "user", "content": (
                            f"ERREUR: {syntax_error}\n\n"
                            f"LANGAGE: {language}\n\n"
                            f"CODE À CORRIGER:\n{current_code}\n\n"
                            "corrige ce programmme pour ne plus avoir cette erreur"
                        )},
                    ], model)
                    total_retries += retries3
                    if fixed_syntax:
                        candidate2 = _clean_code_block(fixed_syntax)
                        ok2, _ = _validate_python_syntax(candidate2)
                        if ok2:
                            current_code = candidate2
                            versions[-1] = current_code  # Update last version
                            validated = True
                            log("   ✅ Syntaxe corrigée avec succès")
                        else:
                            log("   ❌ Correction syntaxique échouée — on conserve le code actuel")
                    else:
                        log("   ❌ Impossible d'obtenir une correction syntaxique")
            else:
                validated = False  # Non-Python languages not validated yet
                log(f"   ℹ️  Validation syntaxique non disponible pour {language}")

            # ── 4. Improvement analysis ───────────────────────────
            log("   🔬 Analyse des améliorations pour la prochaine itération…")
            improve_raw, retries = _call_llm_with_retry([
                {"role": "system", "content": _SYS_IMPROVE},
                {"role": "user", "content": (
                    f"OBJECTIF: {objective}\n\n"
                    f"LANGAGE: {language}\n\n"
                    f"CODE (itération {iteration}):\n{current_code}"
                )},
            ], model)
            total_retries += retries

            if improve_raw:
                improvements = [
                                   line.lstrip("0123456789.-) ").strip()
                                   for line in improve_raw.splitlines()
                                   if line.strip() and line.strip()[0].isdigit()
                               ][:10]
                if improvements:
                    log(f"   🧠 {len(improvements)} point(s) d'amélioration identifiés")
                else:
                    log("   ⚠️  Aucune amélioration identifiée — la boucle s'arrête.")
                    improvements = []  # Ensure empty
                    break  # Exit loop because no improvements
            else:
                log("   ⚠️  Aucune suggestion d'amélioration obtenue — la boucle s'arrête.")
                improvements = []
                break

            log(f"✅ Itération {iteration} terminée")

        # ── Final result ─────────────────────────────────────────
        log()
        if success and iteration > 0:
            log("🏁 Boucle CLAW v3 terminée avec succès !")
            log(f"📦 Code final : {len(current_code):,} caractères")
            log(f"🔢 Itérations effectuées : {iteration}")
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

        result = {
            "logs": logs,
            "final_code": final_code,
            "success": success,
            "iterations": iteration if success else None,
            "char_count": len(final_code) if final_code else 0,
            "validated": validated,
            "retries": total_retries,
            "versions": versions,
            "language": language,
            "file_extension": _get_file_extension(language),
        }
        with claw_lock:
            claw_tasks[task_id]['result'] = result
            claw_tasks[task_id]['status'] = 'complete' if success else 'error'
            claw_tasks[task_id]['progress'] = 100

    except Exception as e:
        logger.error(f"Erreur dans run_claw_task: {e}")
        with claw_lock:
            claw_tasks[task_id]['status'] = 'error'
            claw_tasks[task_id]['error'] = str(e)


@app.route("/api/claw/stream")
def claw_stream():
    """
    Stream les logs de la tâche CLAW en cours via SSE.
    Le client doit fournir le task_id en query parameter.
    """
    task_id = request.args.get("task_id")
    if not task_id or task_id not in claw_tasks:
        return jsonify({"error": "Tâche inconnue"}), 404

    task = claw_tasks[task_id]

    def generate():
        # Envoyer les logs existants d'abord
        for log in task['logs']:
            yield f"data: {json.dumps({'type': 'log', 'message': log})}\n\n"

        # Puis suivre les nouveaux logs
        last_index = len(task['logs'])
        while task['status'] == 'running':
            # Vérifier les nouveaux logs
            with claw_lock:
                current_logs = task['logs'][last_index:]
                if current_logs:
                    for log in current_logs:
                        yield f"data: {json.dumps({'type': 'log', 'message': log})}\n\n"
                    last_index += len(current_logs)
                # Envoyer la progression
                progress = task.get('progress', 0)
                yield f"data: {json.dumps({'type': 'progress', 'value': progress})}\n\n"
            time.sleep(0.5)

        # Envoyer le résultat final
        if task['status'] == 'complete' and task.get('result'):
            yield f"data: {json.dumps({'type': 'complete', 'result': task['result']})}\n\n"
        elif task['status'] == 'error':
            yield f"data: {json.dumps({'type': 'error', 'message': task.get('error', 'Erreur inconnue')})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/claw/export", methods=["POST"])
def claw_export():
    """Export the generated code as a ZIP project structure."""
    data = request.get_json()
    version_index = data.get("version_index", -1)  # -1 = latest
    language = data.get("language", "python")
    objective = data.get("objective", "")

    # Get versions from session or request
    sess = _get_session()
    versions = sess.get('claw_versions', [])

    # If versions not in session, try from request
    if not versions and 'versions' in data:
        versions = data['versions']

    if not versions:
        return jsonify({"error": "Aucune version disponible"}), 400

    # Select version
    if version_index < 0:
        version_index = len(versions) - 1
    if version_index >= len(versions):
        version_index = len(versions) - 1

    code = versions[version_index]

    # Generate project structure
    files = _generate_project_structure(code, language, objective)

    # Create ZIP
    zip_data = _create_zip(files)

    # Determine filename
    ext = _get_file_extension(language)
    main_file = f"main{ext}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"claw_project_{timestamp}.zip"

    return send_file(
        io.BytesIO(zip_data),
        mimetype='application/zip',
        as_attachment=True,
        download_name=filename
    )


@app.route("/api/claw/diff", methods=["POST"])
def claw_diff():
    """Compute diff between two versions."""
    data = request.get_json()
    version1_idx = data.get("version1", 0)
    version2_idx = data.get("version2", 1)

    sess = _get_session()
    versions = sess.get('claw_versions', [])

    if len(versions) < 2:
        return jsonify({"error": "Pas assez de versions pour comparer"}), 400

    if version1_idx >= len(versions) or version2_idx >= len(versions):
        return jsonify({"error": "Index de version invalide"}), 400

    old_code = versions[version1_idx]
    new_code = versions[version2_idx]

    diff = _compute_diff(old_code, new_code)

    return jsonify({
        "diff": diff,
        "version1_size": len(old_code),
        "version2_size": len(new_code),
        "added_lines": sum(1 for line in diff.split('\n') if line.startswith('+')),
        "removed_lines": sum(1 for line in diff.split('\n') if line.startswith('-')),
    })


# ---------------------------------------------------------------------------
# HTML Template (Modified for SSE real-time display)
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
  display: none; align-items: center; gap: 5px;
}
#context-indicator::before { content: '●'; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

#skills-indicator {
  font-size: 11px; color: var(--accent3);
  display: none; align-items: center; gap: 5px;
}
#language-badge {
  font-size: 11px; padding: 4px 10px;
  border-radius: 20px;
  background: rgba(139,92,246,.15);
  color: var(--accent3);
  border: 1px solid rgba(139,92,246,.3);
  font-family: 'JetBrains Mono', monospace;
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
  font-family: 'JetBrains+Mono', monospace; font-size: 13px;
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
  font-family: 'JetBrains+Mono', monospace;
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
.claw-output-header-left {
  display: flex; align-items: center; gap: 12px;
}
.version-selector {
  background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 8px; font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
}
.claw-output-header span { color: var(--success); font-weight: 600; }
.claw-output-meta {
  padding: 8px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  color: var(--muted);
  display: flex; gap: 16px; flex-wrap: wrap;
}
.claw-output-meta em { color: var(--accent2); font-style: normal; }
.copy-btn, .export-btn, .diff-btn {
  padding: 6px 14px; border-radius: 6px;
  font-size: 11px; cursor: pointer; transition: all .2s;
  font-family: 'JetBrains Mono', monospace;
  border: 1px solid;
}
.copy-btn {
  background: rgba(34,197,94,.1); border-color: rgba(34,197,94,.3);
  color: var(--success);
}
.copy-btn:hover { background: rgba(34,197,94,.2); }
.export-btn {
  background: rgba(59,130,246,.1); border-color: rgba(59,130,246,.3);
  color: var(--accent);
}
.export-btn:hover { background: rgba(59,130,246,.2); }
.diff-btn {
  background: rgba(245,158,11,.1); border-color: rgba(245,158,11,.3);
  color: var(--warn);
}
.diff-btn:hover { background: rgba(245,158,11,.2); }
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

/* Diff viewer styles */
#diff-modal {
  position: fixed; inset: 0;
  background: rgba(0,0,0,.85);
  display: none; align-items: center; justify-content: center;
  z-index: 2000;
}
#diff-modal.active { display: flex; }
.diff-content {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  width: 90%; max-width: 1200px;
  max-height: 90vh;
  display: flex; flex-direction: column;
}
.diff-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
}
.diff-header h3 {
  font-family: 'Syne', sans-serif;
  color: var(--text); margin: 0;
}
.diff-close {
  background: none; border: none; color: var(--muted);
  font-size: 24px; cursor: pointer; padding: 4px;
}
.diff-body {
  flex: 1; overflow: auto;
  padding: 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; line-height: 1.5;
}
.diff-line {
  padding: 2px 0;
  white-space: pre;
}
.diff-add { background: rgba(34,197,94,.1); color: var(--success); }
.diff-del { background: rgba(239,68,68,.1); color: var(--danger); }
.diff-hdr { color: var(--muted); font-weight: bold; }

@media (max-width: 768px) {
  #sidebar { position: absolute; z-index: 50; height: 100%; transform: translateX(-100%); }
  #sidebar.open { transform: none; }
  #sidebar-toggle { display: block; }
  .claw-grid { grid-template-columns: 1fr; }
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
    <h1>NeuralClaw<span>Powered by OpenRouter</span></h1>
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
    <div class="section-label">Contexte / Persona <span style="color:var(--success); font-size:10px;">(auto)</span></div>
    <div class="field-group">
      <label class="field-label">Comportement de l'IA</label>
      <textarea id="system-prompt" placeholder="Ex: Tu es un expert Python. Réponds avec des exemples concrets…" oninput="debouncedSaveContext()"></textarea>
      <small style="color:var(--muted); font-size:11px; margin-top:4px;">
        Sauvegardé automatiquement
      </small>
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
      <button class="nav-tab claw-tab" data-view="claw" onclick="switchView('claw')">🐾 CLAW v3</button>
    </div>
    <div id="topbar-right">
      <span id="skills-indicator">✨ Skills actifs</span>
      <span id="context-indicator" style="display:none">Contexte actif</span>
      <span id="model-badge">{{ default_model }}</span>
      <span id="language-badge" style="display:none">Python</span>
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
      <h2>🐾 CLAW Engine v3</h2>
      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
        <span class="badge">Détection auto • Validation syntaxique • Export ZIP • Diff visuel • SSE temps réel</span>
        <span class="badge" id="claw-language-badge" style="display:none">Python</span>
      </div>
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
        <label>📜 Journal d'exécution (temps réel)</label>
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
        <div class="claw-output-header-left">
          <span>✅ Code Généré</span>
          <select id="claw-version-select" class="version-selector" onchange="changeClawVersion()">
            <option value="0">Version initiale</option>
            <option value="1">Après itération 1</option>
            <option value="2" selected>Après itération 2</option>
            <option value="3">Version finale</option>
          </select>
          <span id="claw-lang-display" style="color:var(--accent2); font-size:11px;"></span>
        </div>
        <div style="display:flex; gap:8px;">
          <button class="diff-btn" onclick="showDiff()">🔄 Diff</button>
          <button class="export-btn" onclick="exportClawCode()">📦 Exporter</button>
          <button class="copy-btn" onclick="copyClawCode()">📋 Copier</button>
        </div>
      </div>
      <div class="claw-output-meta" id="claw-output-meta"></div>
      <pre><code id="claw-code" class="hljs"></code></pre>
    </div>

  </div>

</div>

<div id="toast"></div>

<!-- Diff Modal -->
<div id="diff-modal">
  <div class="diff-content">
    <div class="diff-header">
      <h3> Comparaison des versions</h3>
      <button class="diff-close" onclick="closeDiffModal()">×</button>
    </div>
    <div class="diff-body" id="diff-body"></div>
  </div>
</div>

<script>
/* =========================================================
   GLOBALS
   ========================================================= */
let hasContext   = false;
let skills       = [];
let attachedFiles = [];
let clawFiles    = [];
let clawRunning  = false;
let clawVersions = [];
let clawLanguage = '';
let currentClawVersionIndex = 3;
let contextSaveTimeout;
let eventSource = null; // Pour pouvoir fermer la connexion SSE

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
   CONTEXT AUTO-SAVE
   ========================================================= */
function saveContext() {
  const prompt = document.getElementById('system-prompt').value.trim();
  const model = document.getElementById('model-select').value;

  fetch('/api/context', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system_prompt: prompt, model: model })
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

// Sauvegarde automatique du modèle aussi
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
   CLAW — SSE REAL-TIME STREAMING
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
    // 1. Démarrer la tâche
    const res = await fetch('/api/claw/process', { method: 'POST', body: fd });
    const data = await res.json();

    if (data.error) {
      clawLog('❌ ' + data.error, 'err');
      showToast('❌ Erreur CLAW : ' + data.error, 'error');
      clawRunning = false;
      runBtn.disabled = false;
      runBtn.textContent = '🚀 Lancer la boucle CLAW';
      return;
    }

    const task_id = data.task_id;

    // Fermer toute connexion SSE précédente
    if (eventSource) {
      eventSource.close();
    }

    // 2. Connecter au stream SSE
    eventSource = new EventSource(`/api/claw/stream?task_id=${task_id}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
          // Analyser le type de log pour la coloration
          let type = 'info';
          if (!data.message.trim()) { clawLog('', 'dim'); return; }
          if (data.message.includes('❌')) type = 'err';
          else if (data.message.includes('✅') || data.message.includes('🎯') || data.message.includes('🏁')) type = 'ok';
          else if (data.message.includes('⚠️') || data.message.includes('🔄')) type = 'warn';
          else if (data.message.includes('══')) type = 'phase';
          else if (data.message.includes('↩️') || data.message.toLowerCase().includes('retry')) type = 'retry';
          else if (data.message.startsWith('   ')) type = 'dim';
          clawLog(data.message, type);
        } else if (data.type === 'progress') {
          setClawProgress(data.value);
        } else if (data.type === 'complete') {
          eventSource.close();
          handleClawComplete(data.result);
          clawRunning = false;
          runBtn.disabled = false;
          runBtn.textContent = '🚀 Lancer la boucle CLAW';
        } else if (data.type === 'error') {
          eventSource.close();
          clawLog('❌ ' + data.message, 'err');
          showToast('❌ Erreur tâche: ' + data.message, 'error');
          clawRunning = false;
          runBtn.disabled = false;
          runBtn.textContent = '🚀 Lancer la boucle CLAW';
        }
      } catch (e) {
        console.error('Erreur parsing SSE:', e);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      showToast('❌ Connexion SSE interrompue', 'error');
      clawRunning = false;
      runBtn.disabled = false;
      runBtn.textContent = '🚀 Lancer la boucle CLAW';
    };

  } catch (e) {
    clawLog('❌ Erreur réseau : ' + e.message, 'err');
    showToast('❌ Erreur réseau', 'error');
    clawRunning = false;
    runBtn.disabled = false;
    runBtn.textContent = '🚀 Lancer la boucle CLAW';
  }
}

function handleClawComplete(result) {
  if (result.final_code) {
    clawVersions = result.versions || [];
    clawLanguage = result.language || 'python';
    currentClawVersionIndex = clawVersions.length - 1;

    // Mettre à jour le badge langue
    const langBadge = document.getElementById('language-badge');
    const clawLangBadge = document.getElementById('claw-language-badge');
    if (clawLanguage) {
      langBadge.textContent = clawLanguage;
      langBadge.style.display = 'inline-flex';
      clawLangBadge.textContent = clawLanguage;
      clawLangBadge.style.display = 'inline-flex';
    }

    updateClawCodeDisplay();

    // Afficher les métadonnées
    const meta = document.getElementById('claw-output-meta');
    meta.innerHTML = [
      `Langage : <em>${clawLanguage}</em>`,
      result.iterations ? `<span>Itérations : <em>${result.iterations}</em></span>` : '',
      result.char_count ? `<span>Taille : <em>${result.char_count.toLocaleString()} caractères</em></span>` : '',
      result.validated ? `<span>Syntaxe : <em>✅ valide</em></span>` : `<span>Syntaxe : <em>⚠️ non validée</em></span>`,
      result.retries > 0 ? `<span>Retentatives : <em>${result.retries}</em></span>` : '',
      clawVersions.length > 1 ? `<span>Versions : <em>${clawVersions.length}</em></span>` : '',
    ].filter(Boolean).join('');

    // Mettre à jour le sélecteur de version
    const versionSelect = document.getElementById('claw-version-select');
    versionSelect.innerHTML = '';
    clawVersions.forEach((v, i) => {
      const option = document.createElement('option');
      option.value = i;
      let label = i === 0 ? 'Version initiale' : `Après itération ${i}`;
      if (i === clawVersions.length - 1) label += ' (finale)';
      option.textContent = label;
      versionSelect.appendChild(option);
    });
    versionSelect.value = currentClawVersionIndex;

    document.getElementById('claw-output-area').style.display = 'block';
    showToast('✅ Boucle CLAW terminée !', 'success');
  } else {
    showToast('⚠️ Boucle terminée sans code final', 'warn');
  }
}

function changeClawVersion() {
  const select = document.getElementById('claw-version-select');
  currentClawVersionIndex = parseInt(select.value);
  updateClawCodeDisplay();
}

function updateClawCodeDisplay() {
  if (clawVersions.length === 0) return;

  const code = clawVersions[currentClawVersionIndex] || '';
  const codeEl = document.getElementById('claw-code');
  codeEl.textContent = code;
  hljs.highlightElement(codeEl);

  // Update language display if we have language info
  if (clawLanguage) {
    document.getElementById('claw-lang-display').textContent = 
      `(${clawLanguage})`;
  }
}

async function showDiff() {
  if (clawVersions.length < 2) {
    return showToast('Pas assez de versions pour comparer', 'warn');
  }

  const idx1 = currentClawVersionIndex;
  const idx2 = currentClawVersionIndex > 0 ? currentClawVersionIndex - 1 : 1;

  try {
    const res = await fetch('/api/claw/diff', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        version1: idx1,
        version2: idx2
      })
    });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    const diffBody = document.getElementById('diff-body');
    const lines = data.diff.split('\n');
    let html = '';

    lines.forEach(line => {
      let className = 'diff-line';
      if (line.startsWith('+') && !line.startsWith('+++')) {
        className += ' diff-add';
      } else if (line.startsWith('-') && !line.startsWith('---')) {
        className += ' diff-del';
      } else if (line.startsWith('@@')) {
        className += ' diff-hdr';
      }
      html += `<div class="${className}">${escHtml(line)}</div>`;
    });

    diffBody.innerHTML = html;
    document.getElementById('diff-modal').classList.add('active');
  } catch (e) {
    showToast('❌ Erreur diff: ' + e.message, 'error');
  }
}

function closeDiffModal() {
  document.getElementById('diff-modal').classList.remove('active');
}

function clearClaw() {
  // Fermer la connexion SSE si elle est ouverte
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  document.getElementById('claw-obj').value = '';
  clawFiles = [];
  renderClawFiles();
  document.getElementById('claw-logs').innerHTML = '<span class="log-dim">En attente de lancement…</span>';
  document.getElementById('claw-output-area').style.display = 'none';
  document.getElementById('claw-progress-wrap').style.display = 'none';
  setClawProgress(0);
  clawVersions = [];
  clawLanguage = '';
  document.getElementById('language-badge').style.display = 'none';
  document.getElementById('claw-language-badge').style.display = 'none';
  showToast('🔄 Réinitialisé', 'success');
}

function copyClawCode() {
  const code = clawVersions[currentClawVersionIndex] || '';
  if (!code) return showToast('Aucun code à copier', 'error');

  navigator.clipboard.writeText(code)
    .then(() => showToast('📋 Code copié !', 'success'))
    .catch(() => showToast('❌ Impossible de copier', 'error'));
}

async function exportClawCode() {
  if (clawVersions.length === 0) return showToast('Aucun code à exporter', 'error');

  const code = clawVersions[currentClawVersionIndex];
  const objective = document.getElementById('claw-obj').value;

  try {
    const res = await fetch('/api/claw/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code: code,
        language: clawLanguage,
        objective: objective,
        version_index: currentClawVersionIndex
      })
    });

    if (!res.ok) throw new Error('Export failed');

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `claw_project_${Date.now()}.zip`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('📦 Projet exporté !', 'success');
  } catch (e) {
    showToast('❌ Erreur export: ' + e.message, 'error');
  }
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
  if (e.key === 'Escape') {
    closeSkillModal();
    closeDiffModal();
  }
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
    logger.info("🚀 AI Chatbot + CLAW v3")
    # Render donne un port dans la variable PORT, sinon on prend 5000 en local
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
