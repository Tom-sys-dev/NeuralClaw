from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Generator

import requests as http_requests
from flask import Blueprint, Response, jsonify, request, stream_with_context

from config.py import DEFAULT_MODEL, logger
from llm.py import call_llm
from session_helpers.py import LANG_RULES_SHORT

# ---------------------------------------------------------------------------
# Retry / timeouts / budgets
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_DELAY = 1.5
RETRY_BACKOFF = 2.0

# Budgets généreux pour des réponses complètes et de haute qualité.
TOKENS_PM = 2_500  # Cadrage PM détaillé
TOKENS_ORCHESTRATOR = 2_000  # Sélection d'experts
TOKENS_AGENT = 2_000  # Chaque expert peut développer son analyse
TOKENS_DEBATE = 4_000  # Réponse de débat inter-experts (plus courte)
TOKENS_ADVOCATE = 3_500  # Critique de l'avocat du diable
TOKENS_SYNTHESIS = 10_000  # Synthèse riche pour alimenter le programmeur
TOKENS_PROGRAMMER = 64_000  # Programmeur : sortie code complète sans troncature
TOKENS_REVIEWER = 8_000  # Reviewer de code post-génération
TOKENS_TESTS = 12_000  # Génération des tests unitaires
TOKENS_CONTINUATION = 32_000  # Continuation large si troncature détectée
TOKENS_SUMMARY = 1_500  # Résumé exécutif court

# Timeout par expert (secondes).
EXPERTS_TIMEOUT_SECONDS = 300

# Marqueur de complétude pour le code final.
FINAL_CODE_END_MARKER = "__WARROOM_COMPLETE__"

# Taille max de la requête utilisateur.
MAX_QUERY_LENGTH = 8_000

# Modèles autorisés (allowlist sécurité).
ALLOWED_MODELS: set[str] = {
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-6",
    DEFAULT_MODEL,
}

# Presets de mode (fast / balanced / full).
MODE_PRESETS: dict[str, dict] = {
    "fast": {
        "num_experts": 1,
        "debate_rounds": 0,
        "advocate": False,
        "reviewer": False,
        "generate_tests": False,
        "executive_summary": False,
        "tokens_agent": 3_000,
        "tokens_programmer": 16_000,
    },
    "balanced": {
        "num_experts": 3,
        "debate_rounds": 0,
        "advocate": True,
        "reviewer": True,
        "generate_tests": False,
        "executive_summary": True,
        "tokens_agent": 8_000,
        "tokens_programmer": 64_000,
    },
    "full": {
        "num_experts": 5,
        "debate_rounds": 1,
        "advocate": True,
        "reviewer": True,
        "generate_tests": True,
        "executive_summary": True,
        "tokens_agent": 8_000,
        "tokens_programmer": 64_000,
    },
}

# ---------------------------------------------------------------------------
# In-memory stores (Redis-compatible interface — swap _store_* calls for Redis)
# ---------------------------------------------------------------------------

_cache_store: dict[str, dict] = {}  # key -> {value, expires_at}
_session_store: dict[str, list] = {}  # session_id -> [warroom_result, ...]
_job_store: dict[str, dict] = {}  # job_id -> {status, result, created_at}
_rate_store: dict[str, list] = defaultdict(list)  # ip -> [timestamps]

CACHE_TTL_SECONDS = 3600
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW = 60


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
    """Retourne True si la requête est autorisée, False si rate-limitée."""
    now = time.time()
    timestamps = _rate_store[ip]
    _rate_store[ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_store[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_store[ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Helpers LLM
# ---------------------------------------------------------------------------

def call_llm_with_retry(
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4_000,
        step_name: str = "LLM",
) -> str:
    """Appel LLM avec retry et backoff exponentiel."""
    delay = RETRY_DELAY
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = call_llm(messages, model=model, max_tokens=max_tokens)
            if not isinstance(result, str) or not result.strip():
                raise ValueError("Réponse LLM vide")
            if attempt > 1:
                logger.info("[%s] Succès tentative %d/%d", step_name, attempt, MAX_RETRIES)
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                logger.warning(
                    "[%s] Tentative %d/%d échouée : %s — retry dans %.1fs",
                    step_name, attempt, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                delay *= RETRY_BACKOFF
            else:
                logger.error("[%s] Échec définitif (%d/%d) : %s", step_name, attempt, MAX_RETRIES, exc)

    raise last_exc  # type: ignore[misc]


def _clean_markdown_fences(raw: str) -> str:
    return re.sub(r"```(?:json|python|javascript|ts|tsx|js|html|bash)?|```", "", raw).strip()


def _parse_json_llm(raw: str) -> dict:
    """Nettoie les balises markdown et extrait le premier objet JSON valide."""
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
    suspicious_endings = (
        "```", "def ", "class ", "return", "except", "finally",
        "else:", "elif ", "for ", "while ", "if ", "{", "[", "(", ",", ":",
    )
    if any(stripped.endswith(x) for x in suspicious_endings):
        return True
    if stripped.count("```") % 2 != 0:
        return True
    return FINAL_CODE_END_MARKER not in stripped


def _safe_excerpt(text: str, max_chars: int = 8_000) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[tronqué]"


def _normalize_expert_payload(expert: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(expert.get("role", "Expert")),
        "emoji": str(expert.get("emoji", "🔹")),
        "specialty": str(expert.get("specialty", "Analyse spécialisée")),
    }


def _estimate_tokens(text: str) -> int:
    """Estimation rapide du nombre de tokens (1 token ≈ 4 caractères)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Détection de stack
# ---------------------------------------------------------------------------

_STACK_KEYWORDS: dict[str, list[str]] = {
    "python/fastapi": ["fastapi", "uvicorn", "pydantic", "async def"],
    "python/flask": ["flask", "blueprint", "app.route", "jsonify"],
    "python/django": ["django", "models.py", "views.py", "urls.py"],
    "python": ["python", ".py", "pip", "pandas", "numpy", "pytest"],
    "typescript/react": ["react", "tsx", "jsx", "next.js", "nextjs", "vite"],
    "typescript/node": ["express", "nestjs", "ts-node", "typescript node"],
    "typescript": ["typescript", ".ts", "tsc", "interface ", "type "],
    "javascript": ["javascript", "node.js", "nodejs", ".js", "npm", "yarn"],
    "go": ["golang", " go ", ".go", "goroutine", "gin"],
    "rust": ["rust", "cargo", "tokio", ".rs"],
    "java": ["java", "spring", "maven", "gradle", ".java"],
    "kotlin": ["kotlin", ".kt", "coroutine"],
    "swift": ["swift", "swiftui", "uikit", ".swift"],
    "sql": ["sql", "postgres", "mysql", "sqlite", "query"],
}


def _detect_stack(query: str, explicit_stack: str | None = None) -> str | None:
    if explicit_stack:
        return explicit_stack
    q = query.lower()
    for stack, keywords in _STACK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return stack
    return None


# ---------------------------------------------------------------------------
# Validation syntaxique du code généré
# ---------------------------------------------------------------------------

def _validate_syntax(code: str, lang: str) -> tuple[bool, list[str]]:
    """
    Tente une validation syntaxique basique.
    Retourne (is_valid, list_of_errors).
    """
    errors: list[str] = []
    if lang == "python":
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as exc:
            errors.append(f"SyntaxError ligne {exc.lineno}: {exc.msg}")
            return False, errors
    # Pour les autres langages : vérifications heuristiques
    opens = code.count("{") + code.count("(") + code.count("[")
    closes = code.count("}") + code.count(")") + code.count("]")
    if abs(opens - closes) > 3:
        errors.append(f"Déséquilibre de parenthèses/accolades : {opens} ouvrantes / {closes} fermantes")
    return len(errors) == 0, errors


def _extract_files(programmer_output: str) -> list[dict]:
    """
    Parse les blocs ### FILE: name.ext + ```lang...``` dans la sortie brute
    et retourne une liste structurée {filename, language, content, line_count, syntax_valid, syntax_errors}.
    """
    files: list[dict] = []
    # Pattern : ### FILE: filename.ext suivi d'un bloc ```lang ... ```
    pattern = re.compile(
        r"###\s*FILE:\s*(\S+)\s*\n```(\w*)\n(.*?)```",
        re.S,
    )
    for match in pattern.finditer(programmer_output):
        filename, lang_hint, content = match.group(1), match.group(2).lower(), match.group(3)
        # Déduire le langage depuis l'extension si non précisé
        if not lang_hint:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            lang_hint = {
                "py": "python", "ts": "typescript", "tsx": "typescript",
                "js": "javascript", "jsx": "javascript", "go": "go",
                "rs": "rust", "java": "java", "kt": "kotlin", "swift": "swift",
                "sql": "sql", "sh": "bash", "html": "html", "css": "css",
            }.get(ext, ext)
        valid, errs = _validate_syntax(content, lang_hint)
        files.append({
            "filename": filename,
            "language": lang_hint,
            "content": content.strip(),
            "line_count": content.count("\n") + 1,
            "syntax_valid": valid,
            "syntax_errors": errs,
        })

    # Si aucun bloc FILE: trouvé mais du code existe, retourner un fichier générique
    if not files:
        code_match = re.search(r"```(\w*)\n(.*?)```", programmer_output, re.S)
        if code_match:
            lang_hint = code_match.group(1).lower() or "text"
            content = code_match.group(2)
            valid, errs = _validate_syntax(content, lang_hint)
            files.append({
                "filename": f"output.{lang_hint or 'txt'}",
                "language": lang_hint,
                "content": content.strip(),
                "line_count": content.count("\n") + 1,
                "syntax_valid": valid,
                "syntax_errors": errs,
            })

    return files


def _clean_programmer_output(raw: str) -> str:
    """Retire le marqueur final et les éventuelles lignes narratives parasites."""
    output = raw.replace(FINAL_CODE_END_MARKER, "").rstrip()
    return output


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

warroom_bp = Blueprint("warroom", __name__)

# ---------------------------------------------------------------------------
# Profils d'experts disponibles
# ---------------------------------------------------------------------------

EXPERT_FAMILIES: list[dict] = [
    {"id": "informatique", "label": "💻 Informatique & Tech", "color": "#3b82f6"},
    {"id": "art", "label": "🎨 Art & Design", "color": "#ec4899"},
    {"id": "business", "label": "💼 Business & Management", "color": "#f97316"},
    {"id": "sante", "label": "🏥 Santé & Bien-être", "color": "#22c55e"},
    {"id": "sciences", "label": "🔬 Sciences", "color": "#06b6d4"},
    {"id": "education", "label": "📚 Éducation & Coaching", "color": "#a855f7"},
    {"id": "societe", "label": "🌍 Société & Environnement", "color": "#f59e0b"},
]

EXPERT_PROFILES: dict[str, dict] = {
    "dev_senior": {
        "role": "Développeur Senior", "emoji": "💻", "family": "informatique",
        "description": "Architecte logiciel, code propre et performance.",
        "specialty": "Architecture logicielle, design patterns, refactoring, bonnes pratiques, performance applicative et revue de code.",
    },
    "frontend_dev": {
        "role": "Développeur Frontend", "emoji": "🖥️", "family": "informatique",
        "description": "Interfaces web modernes et réactives.",
        "specialty": "React, Vue, Angular, CSS avancé, accessibilité WCAG et optimisation front.",
    },
    "backend_dev": {
        "role": "Développeur Backend", "emoji": "🔧", "family": "informatique",
        "description": "APIs robustes, bases de données et microservices.",
        "specialty": "APIs REST/GraphQL, microservices, SQL/NoSQL, caching, scalabilité et sécurité backend.",
    },
    "mobile_dev": {
        "role": "Développeur Mobile", "emoji": "📱", "family": "informatique",
        "description": "Applications iOS, Android et cross-platform.",
        "specialty": "React Native, Flutter, Swift, Kotlin, UX mobile et optimisation des performances.",
    },
    "data_scientist": {
        "role": "Data Scientist", "emoji": "📊", "family": "informatique",
        "description": "Analyse de données, ML et statistiques.",
        "specialty": "Machine learning, statistiques, visualisation et évaluation de modèles.",
    },
    "ml_engineer": {
        "role": "ML Engineer", "emoji": "🧠", "family": "informatique",
        "description": "Pipelines IA, déploiement et fine-tuning de modèles.",
        "specialty": "MLOps, déploiement, pipelines de données et monitoring de modèles.",
    },
    "data_engineer": {
        "role": "Data Engineer", "emoji": "🗄️", "family": "informatique",
        "description": "ETL, data warehouses et pipelines de données.",
        "specialty": "Spark, Kafka, Airflow, dbt, data lakes et gouvernance des données.",
    },
    "security_expert": {
        "role": "Expert Cybersécurité", "emoji": "🔒", "family": "informatique",
        "description": "Protection des systèmes et gestion des vulnérabilités.",
        "specialty": "Pentest, threat modeling, IAM, cryptographie, cloud security et réponse aux incidents.",
    },
    "devops": {
        "role": "DevOps / Cloud", "emoji": "⚙️", "family": "informatique",
        "description": "Infrastructure, CI/CD et scalabilité cloud.",
        "specialty": "Docker, Kubernetes, Terraform, CI/CD, observabilité et SRE.",
    },
    "ux_designer": {
        "role": "UX/UI Designer", "emoji": "🎨", "family": "art",
        "description": "Expérience utilisateur et interfaces intuitives.",
        "specialty": "Research UX, wireframes, prototypage, design systems et accessibilité.",
    },
    "creative_dir": {
        "role": "Directeur Créatif", "emoji": "✨", "family": "art",
        "description": "Innovation visuelle, branding et storytelling.",
        "specialty": "Direction artistique, identité de marque et stratégie créative.",
    },
    "graphic_designer": {
        "role": "Graphiste", "emoji": "🖌️", "family": "art",
        "description": "Identité visuelle, print et illustration.",
        "specialty": "Branding, illustration, mise en page éditoriale et packaging.",
    },
    "motion_designer": {
        "role": "Motion Designer", "emoji": "🎬", "family": "art",
        "description": "Animations, vidéo et effets visuels.",
        "specialty": "Animation 2D/3D, motion graphics, storyboard et post-production.",
    },
    "photographer": {
        "role": "Directeur Photo", "emoji": "📷", "family": "art",
        "description": "Composition, lumière et direction artistique.",
        "specialty": "Direction photo, composition, éclairage et post-traitement.",
    },
    "copywriter": {
        "role": "Copywriter", "emoji": "✍️", "family": "art",
        "description": "Contenu persuasif, storytelling et conversion.",
        "specialty": "Copywriting, content marketing, SEO rédactionnel et UX writing.",
    },
    "architect": {
        "role": "Architecte / Designer Intérieur", "emoji": "🏛️", "family": "art",
        "description": "Conception spatiale, ergonomie et esthétique.",
        "specialty": "Conception architecturale, design d'intérieur et durabilité.",
    },
    "product_manager": {
        "role": "Product Manager", "emoji": "🗺️", "family": "business",
        "description": "Roadmap produit, priorisation et go-to-market.",
        "specialty": "Vision produit, user stories, OKRs, priorisation et métriques produit.",
    },
    "business_strat": {
        "role": "Stratège Business", "emoji": "🎯", "family": "business",
        "description": "Vision stratégique, business model et croissance.",
        "specialty": "SWOT, business model, analyse concurrentielle et stratégie de croissance.",
    },
    "marketing": {
        "role": "Expert Marketing", "emoji": "📣", "family": "business",
        "description": "Acquisition, branding et growth hacking.",
        "specialty": "Stratégie digitale, SEO/SEA, growth et analyse d'audience.",
    },
    "finance": {
        "role": "Analyste Finance", "emoji": "💰", "family": "business",
        "description": "ROI, budgets, rentabilité et levée de fonds.",
        "specialty": "Analyse financière, modélisation, trésorerie et gestion des risques.",
    },
    "legal": {
        "role": "Conseiller Légal", "emoji": "⚖️", "family": "business",
        "description": "Droit des affaires, contrats et propriété intellectuelle.",
        "specialty": "Contrats, conformité, propriété intellectuelle et protection des données.",
    },
    "hr_expert": {
        "role": "Expert RH & Management", "emoji": "👥", "family": "business",
        "description": "Recrutement, culture d'entreprise et leadership.",
        "specialty": "Stratégie RH, talents, leadership et conduite du changement.",
    },
    "entrepreneur": {
        "role": "Entrepreneur / Startuper", "emoji": "🚀", "family": "business",
        "description": "Lean startup, MVP, pivots et traction.",
        "specialty": "Lean startup, validation d'hypothèses, MVP et product-market fit.",
    },
    "project_manager": {
        "role": "Chef de Projet", "emoji": "📋", "family": "business",
        "description": "Planning, ressources et livraison des projets.",
        "specialty": "Agile, gestion des risques, coordination et pilotage des KPIs.",
    },
    "medecin": {
        "role": "Médecin Généraliste", "emoji": "🩺", "family": "sante",
        "description": "Diagnostics, prévention et santé globale.",
        "specialty": "Prévention, orientation clinique et médecine basée sur les preuves.",
    },
    "psychologue": {
        "role": "Psychologue", "emoji": "🧠", "family": "sante",
        "description": "Santé mentale, thérapies et bien-être émotionnel.",
        "specialty": "TCC, stress, burn-out et relation d'aide.",
    },
    "nutritionniste": {
        "role": "Nutritionniste", "emoji": "🥗", "family": "sante",
        "description": "Alimentation équilibrée et santé métabolique.",
        "specialty": "Nutrition, micronutrition, troubles alimentaires et performance.",
    },
    "pharmacien": {
        "role": "Pharmacien", "emoji": "💊", "family": "sante",
        "description": "Médicaments, interactions et protocoles de traitement.",
        "specialty": "Pharmacologie, interactions médicamenteuses et suivi des traitements.",
    },
    "coach_sportif": {
        "role": "Coach Sportif / Kinésithérapeute", "emoji": "🏋️", "family": "sante",
        "description": "Performance physique, entraînement et récupération.",
        "specialty": "Planification d'entraînement, biomécanique et prévention des blessures.",
    },
    "physicien": {
        "role": "Physicien", "emoji": "⚛️", "family": "sciences",
        "description": "Mécanique, énergie, physique quantique et optique.",
        "specialty": "Physique théorique et appliquée, matériaux et modélisation.",
    },
    "biologiste": {
        "role": "Biologiste", "emoji": "🧬", "family": "sciences",
        "description": "Génétique, biologie cellulaire et écosystèmes.",
        "specialty": "Biologie moléculaire, microbiologie, biotechnologies et écologie.",
    },
    "chimiste": {
        "role": "Chimiste", "emoji": "🧪", "family": "sciences",
        "description": "Réactions, matériaux et formulations.",
        "specialty": "Chimie analytique, organique, matériaux et sécurité chimique.",
    },
    "mathematicien": {
        "role": "Mathématicien", "emoji": "📐", "family": "sciences",
        "description": "Modélisation, algorithmes et statistiques avancées.",
        "specialty": "Probabilités, optimisation, algorithmes et cryptographie.",
    },
    "ingenieur": {
        "role": "Ingénieur Généraliste", "emoji": "🔩", "family": "sciences",
        "description": "Conception mécanique, fabrication et optimisation.",
        "specialty": "Conception, simulation, production et contrôle qualité.",
    },
    "formateur": {
        "role": "Formateur / Pédagogue", "emoji": "🎓", "family": "education",
        "description": "Conception pédagogique et transmission des savoirs.",
        "specialty": "Ingénierie pédagogique, e-learning et évaluation des apprentissages.",
    },
    "coach_life": {
        "role": "Coach de Vie / Executive Coach", "emoji": "🌱", "family": "education",
        "description": "Développement personnel, objectifs et mindset.",
        "specialty": "Coaching orienté solution, objectifs SMART et leadership personnel.",
    },
    "philosophe": {
        "role": "Philosophe / Éthicien", "emoji": "🦉", "family": "education",
        "description": "Éthique, logique et analyse critique.",
        "specialty": "Éthique appliquée, logique argumentative et pensée critique.",
    },
    "sociologue": {
        "role": "Sociologue", "emoji": "🌐", "family": "societe",
        "description": "Comportements sociaux, tendances et cultures.",
        "specialty": "Analyse sociologique, dynamiques de groupe et tendances culturelles.",
    },
    "economiste": {
        "role": "Économiste", "emoji": "📈", "family": "societe",
        "description": "Marchés, politiques économiques et analyses macro.",
        "specialty": "Micro, macro, politiques publiques et économétrie.",
    },
    "expert_rse": {
        "role": "Expert RSE / Durabilité", "emoji": "♻️", "family": "societe",
        "description": "Responsabilité sociale et impact environnemental.",
        "specialty": "Stratégie RSE, bilan carbone, CSRD et transition énergétique.",
    },
    "journaliste": {
        "role": "Journaliste / Analyste", "emoji": "📰", "family": "societe",
        "description": "Investigation, fact-checking et communication.",
        "specialty": "Fact-checking, investigation, communication de crise et analyse médiatique.",
    },
}


# ---------------------------------------------------------------------------
# Sélection d'experts
# ---------------------------------------------------------------------------

def _select_experts(
        query: str,
        model: str,
        lang_rule: str,
        selected_experts: list[str],
        num_experts: int = 3,
) -> tuple[list[dict], str]:
    """
    Détermine jusqu'à `num_experts` experts.
    Fast-path local si l'utilisateur a déjà fourni des experts valides.
    """
    num_experts = max(1, min(num_experts, 6))
    forced = [_normalize_expert_payload(EXPERT_PROFILES[eid]) for eid in selected_experts if eid in EXPERT_PROFILES]

    if len(forced) >= num_experts:
        experts = forced[:num_experts]
        return experts, f"Analyse selon les profils sélectionnés : {', '.join(e['role'] for e in experts)}"

    if 0 < len(forced) < num_experts:
        local_fillers = [
            {"role": "Développeur Senior", "emoji": "💻", "specialty": "Architecture, qualité et complétude du code"},
            {"role": "Chef de Projet", "emoji": "📋", "specialty": "Priorisation, risques et exécution rapide"},
            {"role": "Expert QA", "emoji": "✅", "specialty": "Tests, robustesse, cas limites et fiabilité"},
            {"role": "Expert Cybersécurité", "emoji": "🔒", "specialty": "Sécurité, vulnérabilités et conformité"},
            {"role": "DevOps / Cloud", "emoji": "⚙️", "specialty": "Déploiement, CI/CD et observabilité"},
            {"role": "Stratège Business", "emoji": "🎯", "specialty": "Vision stratégique et business model"},
        ]
        picked_roles = {e["role"] for e in forced}
        fillers = [f for f in local_fillers if f["role"] not in picked_roles]
        experts = (forced + fillers)[:num_experts]
        return experts, "Analyse hybride rapide avec experts sélectionnés et compléments locaux"

    # Aucun expert choisi : appel orchestrateur JSON.
    prompt = (
        f"{lang_rule}\n"
        f"Tu es un orchestrateur minimaliste. Choisis EXACTEMENT {num_experts} experts utiles pour traiter la demande.\n"
        "Retourne UNIQUEMENT un JSON valide :\n"
        '{"experts": [{"role": "...", "emoji": "...", "specialty": "..."}], "strategy": "..."}\n\n'
        "Contraintes : réponses courtes, sans texte additionnel.\n"
        f"Demande : {query}"
    )
    try:
        ai_data = _parse_json_llm(
            call_llm_with_retry(
                [{"role": "user", "content": prompt}],
                model=model,
                max_tokens=TOKENS_ORCHESTRATOR,
                step_name="Orchestrateur",
            )
        )
        experts = [_normalize_expert_payload(e) for e in ai_data.get("experts", [])][:num_experts]
        if len(experts) < num_experts:
            raise ValueError(f"Pas assez d'experts ({len(experts)}/{num_experts})")
        return experts, ai_data.get("strategy", "Analyse multi-angle")
    except Exception as exc:
        logger.warning("Orchestrateur fallback : %s", exc)
        fallback = [
            {"role": "Expert Technique", "emoji": "💻", "specialty": "Analyse technique et implémentation"},
            {"role": "Stratège", "emoji": "🎯", "specialty": "Vision stratégique et risques"},
            {"role": "Expert QA", "emoji": "✅", "specialty": "Tests, robustesse et validation"},
            {"role": "Expert Cybersécurité", "emoji": "🔒", "specialty": "Sécurité et conformité"},
            {"role": "DevOps", "emoji": "⚙️", "specialty": "Déploiement et infrastructure"},
            {"role": "Product Manager", "emoji": "🗺️", "specialty": "Vision produit et priorisation"},
        ]
        return fallback[:num_experts], "Analyse multi-angle en fallback"


# ---------------------------------------------------------------------------
# Route catalogue des experts
# ---------------------------------------------------------------------------

@warroom_bp.route("/api/experts", methods=["GET"])
def get_experts():
    return jsonify({
        "families": EXPERT_FAMILIES,
        "experts": {k: {**v} for k, v in EXPERT_PROFILES.items()},
    })


# ---------------------------------------------------------------------------
# Étapes du pipeline
# ---------------------------------------------------------------------------

def _build_pm_data(query: str, model: str, lang_rule: str) -> tuple[dict, float]:
    """Cadrage PM. Retourne (pm_data, duration_ms)."""
    t0 = time.time()
    pm_prompt = (
        f"{lang_rule}\n"
        "Tu es un Chef de Projet Senior. Fournis un cadrage TRÈS CONCIS.\n"
        "Retourne UNIQUEMENT un JSON valide :\n"
        '{"project_title":"...","objective":"...","action_plan":[{"step":1,"action":"...","priority":"haute|moyenne|basse"}],"key_risks":["..."],"success_criteria":"..."}\n\n'
        f"Demande : {query}"
    )
    try:
        result = _parse_json_llm(
            call_llm_with_retry(
                [{"role": "user", "content": pm_prompt}],
                model=model,
                max_tokens=TOKENS_PM,
                step_name="Chef-de-Projet",
            )
        )
        return result, (time.time() - t0) * 1000
    except Exception as exc:
        logger.warning("Chef de projet fallback : %s", exc)
        return {
            "project_title": "Analyse en cours",
            "objective": "Résoudre la demande de manière fiable et rapide",
            "action_plan": [
                {"step": 1, "action": "Cadrer le besoin", "priority": "haute"},
                {"step": 2, "action": "Produire une solution exploitable", "priority": "haute"},
                {"step": 3, "action": "Valider complétude et risques", "priority": "haute"},
            ],
            "key_risks": ["Ambiguïté du besoin", "Sortie incomplète"],
            "success_criteria": "Réponse complète, exploitable et livrée rapidement",
        }, (time.time() - t0) * 1000


def _call_agent(
        query: str, model: str, lang_rule: str, expert: dict,
        idx: int, tokens_agent: int = TOKENS_AGENT,
) -> dict:
    """Appel d'un expert individuel.

    Optimisations vs version précédente :
    - Prompt compressé : query tronquée à 2000 chars max pour réduire la latence LLM.
    - Réponse ciblée : on demande 4 points courts au lieu d'une analyse longue.
    - tokens_agent plafonné à 1500 ici pour éviter des générations inutilement longues
      côté expert (la synthèse consolidera de toute façon).
    """
    # Limiter la query dans le prompt expert : un prompt plus court = réponse plus rapide.
    query_excerpt = _safe_excerpt(query, 2_000)
    # Plafonner les tokens expert pour accélérer (la synthèse aura le budget complet).
    effective_tokens = min(tokens_agent, 1_500)
    agent_prompt = (
        f"{lang_rule}\n"
        f"Tu es {expert['role']} ({expert['specialty']}).\n"
        "Analyse COURTE et très actionnable en 4 points (2-3 phrases max chacun) :\n"
        "1) Diagnostic\n2) Recommandations\n3) Risques\n4) Actions immédiates\n\n"
        f"Demande : {query_excerpt}"
    )
    t0 = time.time()
    try:
        response = call_llm_with_retry(
            [{"role": "user", "content": agent_prompt}],
            model=model,
            max_tokens=effective_tokens,
            step_name=f"Expert-{idx}-{expert['role']}",
        )
        return {"expert": expert, "proposal": response, "idx": idx, "error": None,
                "duration_ms": int((time.time() - t0) * 1000)}
    except Exception as exc:
        logger.warning("Expert %s erreur : %s", expert.get("role"), exc)
        return {"expert": expert, "proposal": "Analyse indisponible.",
                "idx": idx, "error": str(exc), "duration_ms": int((time.time() - t0) * 1000)}


def _run_parallel_experts(
        query: str, model: str, lang_rule: str,
        experts: list[dict], tokens_agent: int = TOKENS_AGENT,
) -> list[dict]:
    """Lance tous les experts en parallèle et attend CHACUN individuellement avant de continuer."""
    proposals: list[dict | None] = [None] * len(experts)
    if not experts:
        return []

    with ThreadPoolExecutor(max_workers=len(experts)) as executor:
        futures = {
            executor.submit(_call_agent, query, model, lang_rule, exp, i, tokens_agent): i
            for i, exp in enumerate(experts)
        }
        for future, idx in futures.items():
            expert = experts[idx]
            try:
                result = future.result(timeout=EXPERTS_TIMEOUT_SECONDS)
                proposals[idx] = result
                logger.info("Expert [%d/%d] '%s' reçu en %dms.",
                            idx + 1, len(experts), expert.get("role"),
                            result.get("duration_ms", 0))
            except TimeoutError:
                logger.warning("Expert [%d/%d] '%s' timeout après %ds.",
                               idx + 1, len(experts), expert.get("role"), EXPERTS_TIMEOUT_SECONDS)
                proposals[idx] = {"expert": expert, "proposal": "Analyse non revenue à temps.",
                                  "idx": idx, "error": "timeout", "duration_ms": EXPERTS_TIMEOUT_SECONDS * 1000}
            except Exception as exc:
                logger.warning("Expert [%d/%d] '%s' erreur : %s.",
                               idx + 1, len(experts), expert.get("role"), exc)
                proposals[idx] = {"expert": expert, "proposal": "Analyse indisponible.",
                                  "idx": idx, "error": str(exc), "duration_ms": 0}

    logger.info("Tous les experts (%d/%d) ont répondu.", len(experts), len(experts))
    return [p for p in proposals if p]


def _run_expert_debate(
        query: str, model: str, lang_rule: str,
        proposals: list[dict], rounds: int = 1,
) -> list[dict]:
    """
    Rounds de débat inter-experts : chaque expert lit les analyses des autres
    et peut réfuter, valider ou enrichir sa position.
    Retourne les propositions mises à jour avec les débats.
    """
    if rounds <= 0 or len(proposals) < 2:
        return proposals

    updated = list(proposals)
    for round_idx in range(1, rounds + 1):
        logger.info("Débat inter-experts round %d/%d...", round_idx, rounds)
        others_text_by_idx: dict[int, str] = {}
        for p in updated:
            others = [
                f"--- {o['expert']['emoji']} {o['expert']['role']} ---\n{_safe_excerpt(o['proposal'], 2500)}"
                for o in updated if o["idx"] != p["idx"]
            ]
            others_text_by_idx[p["idx"]] = "\n\n".join(others)

        def _debate_one(p: dict) -> dict:
            debate_prompt = (
                f"{lang_rule}\n"
                f"Tu es {p['expert']['role']} ({p['expert']['specialty']}).\n"
                "Tu viens de lire les analyses de tes collègues. "
                "Enrichis ou corrige ta position initiale en tenant compte de leurs avis.\n"
                "Structure :\n"
                "1) Points d'accord\n"
                "2) Points de désaccord (avec justification)\n"
                "3) Position enrichie et finale\n\n"
                f"DEMANDE : {query}\n\n"
                f"TA PREMIÈRE ANALYSE :\n{_safe_excerpt(p['proposal'], 2000)}\n\n"
                f"ANALYSES DES AUTRES EXPERTS :\n{others_text_by_idx[p['idx']]}"
            )
            try:
                updated_proposal = call_llm_with_retry(
                    [{"role": "user", "content": debate_prompt}],
                    model=model,
                    max_tokens=TOKENS_DEBATE,
                    step_name=f"Débat-R{round_idx}-{p['expert']['role']}",
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
    """Critique interne avant le programmeur (avocat du diable)."""
    prompt = (
        f"{lang_rule}\n"
        "Tu es l'Avocat du Diable. Analyse la synthèse ci-dessous et identifie les failles.\n"
        "Structure :\n"
        "## Failles critiques\n## Risques majeurs\n## Hypothèses dangereuses\n## Points de vigilance pour le code\n\n"
        f"DEMANDE ORIGINALE : {query}\n\n"
        f"SYNTHÈSE À CRITIQUER :\n{_safe_excerpt(synthesis, 5000)}"
    )
    try:
        return call_llm_with_retry(
            [{"role": "user", "content": prompt}],
            model=model,
            max_tokens=TOKENS_ADVOCATE,
            step_name="Avocat-du-Diable",
        )
    except Exception as exc:
        logger.warning("Avocat du diable erreur : %s", exc)
        return "Critique indisponible."


def _build_synthesis(
        query: str, pm_data: dict, proposals: list[dict],
        model: str, lang_rule: str,
) -> tuple[str, float]:
    """Synthèse consolidée. Retourne (synthesis, duration_ms)."""
    t0 = time.time()
    proposals_text = "\n\n".join(
        f"--- {p['expert']['emoji']} {p['expert']['role']} ---\n{_safe_excerpt(p['proposal'], 7000)}"
        for p in proposals
    )
    synthesis_prompt = (
        f"{lang_rule}\n"
        "Tu es un synthétiseur critique.\n"
        "Objectif : produire un plan final complet et orienté exécution.\n\n"
        f"DEMANDE : {query}\n\n"
        "CADRAGE PM :\n"
        f"- Objectif : {pm_data.get('objective', '')}\n"
        f"- Étapes : {', '.join(s.get('action', '') for s in pm_data.get('action_plan', []))}\n"
        f"- Risques : {', '.join(pm_data.get('key_risks', []))}\n\n"
        f"AVIS EXPERTS :\n{proposals_text}\n\n"
        "Réponds avec les sections suivantes :\n"
        "1. Diagnostic consolidé\n"
        "2. Décisions / arbitrages\n"
        "3. Plan d'action final priorisé\n"
        "4. Risques et mitigations\n"
        "5. Inputs minimum pour coder proprement\n"
        "Sois concret et exhaustif."
    )
    try:
        result = call_llm_with_retry(
            [{"role": "user", "content": synthesis_prompt}],
            model=model,
            max_tokens=TOKENS_SYNTHESIS,
            step_name="Synthèse",
        )
        return result, (time.time() - t0) * 1000
    except Exception as exc:
        logger.warning("Synthèse fallback : %s", exc)
        return (
            "1. Diagnostic consolidé\nBesoin traité avec solution fiable et testable.\n\n"
            "2. Décisions / arbitrages\nSimplifier le pipeline, imposer une sortie finale complète.\n\n"
            "3. Plan d'action final priorisé\n- Générer le code\n- Valider complétude\n- Ajouter tests\n\n"
            "4. Risques et mitigations\n- Troncature => continuation ciblée\n\n"
            "5. Inputs minimum\nStack cible, exemples entrée/sortie, contraintes métier."
        ), (time.time() - t0) * 1000


def _build_programmer_prompt(
        query: str, synthesis: str, lang_rule: str,
        advocate_critique: str | None = None,
        stack: str | None = None,
) -> str:
    stack_hint = f"\nSTACK CIBLE : {stack}" if stack else ""
    advocate_section = (
        f"\nPOINTS DE VIGILANCE (Avocat du Diable — tu DOIS les adresser dans le code) :\n"
        f"{_safe_excerpt(advocate_critique, 2000)}\n"
        if advocate_critique else ""
    )
    return (
        f"{lang_rule}\n"
        "Tu es un Développeur Expert Senior Full-Stack.\n"
        "Ta mission : produire UNIQUEMENT du code, complet et directement intégrable.\n"
        f"{stack_hint}\n\n"
        f"DEMANDE ORIGINALE : {query}\n\n"
        f"PLAN FINAL :\n{synthesis}\n"
        f"{advocate_section}\n"
        "RÈGLES DE SORTIE — ABSOLUMENT OBLIGATOIRES, AUCUNE EXCEPTION :\n"
        "1. INTERDIT : toute phrase d'introduction, d'explication, de conclusion ou de commentaire narratif.\n"
        "2. INTERDIT : tout texte en prose avant, pendant ou après le code.\n"
        "3. Commence DIRECTEMENT par le code ou par `### FILE:` — zéro mot avant.\n"
        "4. Si plusieurs fichiers, utilise ce format strict pour chaque fichier :\n"
        "   ### FILE: nom_du_fichier.ext\n"
        "   ```lang\n   ...\n   ```\n"
        "5. Les commentaires inline dans le code (`#`, `//`, `/* */`) sont autorisés.\n"
        "6. Ne coupe jamais une fonction, une classe, un bloc ou un fichier en cours.\n"
        "7. Termine impérativement par cette ligne seule, rien après :\n"
        f"   {FINAL_CODE_END_MARKER}\n"
        "8. AUCUN texte, AUCUNE phrase après le marqueur final.\n"
        "9. Pas de pseudocode ni de TODO non implémenté.\n"
    )


def _complete_generation_if_needed(
        initial_output: str,
        original_prompt: str,
        model: str,
        max_rounds: int = 5,
) -> str:
    """Continuation ciblée si la génération semble tronquée.

    Corrections vs version précédente :
    - Le `break` sur exception est remplacé par `continue` : une erreur sur un round
      n'interrompt plus la boucle, le round suivant tente quand même de finir.
    - Après tous les rounds de continuation, si le code est encore tronqué, un appel
      "rescue" final régénère la fin depuis le dernier point d'ancrage connu.
    - La fenêtre de contexte passée en continuation est limitée à 30 000 chars
      (au lieu de 40 000) pour réduire la latence et le risque de nouveau timeout.
    """
    output = initial_output
    if not _is_likely_truncated(output):
        return output

    logger.info("Génération tronquée détectée — lancement de %d round(s) de continuation.", max_rounds)

    for round_idx in range(1, max_rounds + 1):
        continuation_prompt = (
            "Tu dois TERMINER une génération de code potentiellement tronquée.\n"
            "Règles impératives :\n"
            "1) Ne répète JAMAIS le début déjà généré.\n"
            "2) Reprends EXACTEMENT à partir de la dernière ligne utile ci-dessous.\n"
            "3) Ferme tous les blocs, fonctions, classes et fichiers encore ouverts.\n"
            f"4) Termine obligatoirement par cette ligne seule : {FINAL_CODE_END_MARKER}\n"
            "5) Aucun texte, aucune phrase après ce marqueur.\n\n"
            "DEMANDE ORIGINALE :\n"
            f"{original_prompt[:3_000]}\n\n"  # résumé court pour ne pas surcharger le contexte
            "SORTIE DÉJÀ GÉNÉRÉE (fin) :\n"
            f"{output[-30_000:]}\n\n"
            "Donne UNIQUEMENT la suite manquante, en commençant là où le code s'est arrêté."
        )
        try:
            suffix = call_llm_with_retry(
                [{"role": "user", "content": continuation_prompt}],
                model=model,
                max_tokens=TOKENS_CONTINUATION,
                step_name=f"Programmeur-continuation-{round_idx}",
            )
            output = output.rstrip() + "\n" + suffix.lstrip()
            if not _is_likely_truncated(output):
                logger.info("Continuation terminée au round %d/%d.", round_idx, max_rounds)
                return output
            logger.info("Round %d/%d — encore tronqué, prochain round.", round_idx, max_rounds)
        except Exception as exc:
            # On log mais on NE s'arrête PAS : le round suivant peut réussir.
            logger.warning("Continuation round %d/%d échouée : %s — on continue.", round_idx, max_rounds, exc)
            continue  # ← correction critique : était `break` avant

    # Appel rescue final : si après tous les rounds le code est encore tronqué,
    # on demande à un nouvel appel indépendant de seulement fermer ce qui est ouvert.
    if _is_likely_truncated(output):
        logger.warning("Encore tronqué après %d rounds — appel rescue final.", max_rounds)
        rescue_prompt = (
            "Le code ci-dessous est incomplet. Ta seule mission : "
            "fermer proprement TOUS les blocs ouverts (fonctions, classes, fichiers) "
            "et ajouter le marqueur de fin.\n"
            "Ne répète rien. Commence directement par la fermeture manquante.\n"
            f"Marqueur de fin obligatoire : {FINAL_CODE_END_MARKER}\n\n"
            "FIN DU CODE TRONQUÉ :\n"
            f"{output[-15_000:]}"
        )
        try:
            rescue_suffix = call_llm_with_retry(
                [{"role": "user", "content": rescue_prompt}],
                model=model,
                max_tokens=16_000,
                step_name="Programmeur-rescue",
            )
            output = output.rstrip() + "\n" + rescue_suffix.lstrip()
            if not _is_likely_truncated(output):
                logger.info("Rescue final réussi.")
            else:
                logger.warning("Rescue final insuffisant — sortie retournée telle quelle.")
        except Exception as exc:
            logger.error("Rescue final échoué : %s", exc)

    return output


def _run_code_review(
        query: str, programmer_output: str, model: str, lang_rule: str,
) -> tuple[str, list[dict]]:
    """
    Expert Reviewer : relit le code généré, identifie les corrections nécessaires.
    Retourne (review_text, corrections_list).
    """
    review_prompt = (
        f"{lang_rule}\n"
        "Tu es un Expert Reviewer de code Senior.\n"
        "Relis le code ci-dessous et liste les corrections nécessaires.\n"
        "Format de réponse OBLIGATOIRE :\n"
        "Si tout est correct : LGTM\n"
        "Sinon, retourne UNIQUEMENT un JSON :\n"
        '{"corrections": [{"file": "...", "line_hint": "...", "issue": "...", "fix": "..."}], "summary": "..."}\n\n'
        f"DEMANDE ORIGINALE : {query}\n\n"
        f"CODE À REVIEWER :\n{_safe_excerpt(programmer_output, 30000)}"
    )
    try:
        raw = call_llm_with_retry(
            [{"role": "user", "content": review_prompt}],
            model=model,
            max_tokens=TOKENS_REVIEWER,
            step_name="Code-Reviewer",
        )
        if "LGTM" in raw.upper() and "{" not in raw:
            return raw, []
        data = _parse_json_llm(raw)
        return data.get("summary", raw), data.get("corrections", [])
    except Exception as exc:
        logger.warning("Code reviewer erreur : %s", exc)
        return "Review indisponible.", []


def _apply_corrections_if_needed(
        programmer_output: str, corrections: list[dict],
        query: str, synthesis: str, model: str, lang_rule: str,
        advocate_critique: str | None, stack: str | None,
) -> str:
    """Si le reviewer a trouvé des corrections, relance le programmeur avec les consignes."""
    if not corrections:
        return programmer_output
    corrections_text = "\n".join(
        f"- [{c.get('file', '?')}] {c.get('issue', '')} => {c.get('fix', '')}"
        for c in corrections
    )
    corrected_prompt = (
            _build_programmer_prompt(query, synthesis, lang_rule, advocate_critique, stack)
            + f"\n\nCORRECTIONS OBLIGATOIRES DU REVIEWER :\n{corrections_text}\n"
              "Applique TOUTES ces corrections dans le code final."
    )
    try:
        corrected = call_llm_with_retry(
            [{"role": "user", "content": corrected_prompt}],
            model=model,
            max_tokens=TOKENS_PROGRAMMER,
            step_name="Programmeur-post-review",
        )
        return _complete_generation_if_needed(corrected, corrected_prompt, model=model)
    except Exception as exc:
        logger.warning("Programmeur post-review erreur : %s", exc)
        return programmer_output


def _generate_tests(
        query: str, files: list[dict], model: str, lang_rule: str,
) -> str:
    """Génère des tests unitaires pour le code produit."""
    if not files:
        return ""
    code_context = "\n\n".join(
        f"### FILE: {f['filename']}\n```{f['language']}\n{_safe_excerpt(f['content'], 6000)}\n```"
        for f in files
    )
    test_prompt = (
        f"{lang_rule}\n"
        "Tu es un Expert QA Senior spécialisé en tests automatisés.\n"
        "Génère des tests unitaires complets pour le code fourni.\n"
        "RÈGLES :\n"
        "- Utilise le framework de test standard du langage (pytest, Jest, JUnit, etc.).\n"
        "- Couvre : cas nominaux, cas limites, cas d'erreur.\n"
        "- Commente chaque test avec son intention.\n"
        "- Commence DIRECTEMENT par `### FILE: test_xxx.ext` — zéro introduction.\n"
        f"- Termine par : {FINAL_CODE_END_MARKER}\n\n"
        f"DEMANDE ORIGINALE : {query}\n\n"
        f"CODE À TESTER :\n{code_context}"
    )
    try:
        return call_llm_with_retry(
            [{"role": "user", "content": test_prompt}],
            model=model,
            max_tokens=TOKENS_TESTS,
            step_name="Génération-Tests",
        )
    except Exception as exc:
        logger.warning("Génération tests erreur : %s", exc)
        return ""


def _build_executive_summary(
        query: str, synthesis: str, programmer_output: str,
        model: str, lang_rule: str,
) -> str:
    """Résumé exécutif court (3 bullets, niveau manager)."""
    prompt = (
        f"{lang_rule}\n"
        "Tu es un assistant de direction. Rédige un résumé exécutif en 3 points maximum.\n"
        "Niveau manager : sans jargon technique, orienté résultat et décision.\n"
        "Format : 3 bullet points courts (max 2 phrases chacun).\n\n"
        f"DEMANDE : {query}\n\n"
        f"SYNTHÈSE : {_safe_excerpt(synthesis, 2000)}"
    )
    try:
        return call_llm_with_retry(
            [{"role": "user", "content": prompt}],
            model=model,
            max_tokens=TOKENS_SUMMARY,
            step_name="Résumé-Exécutif",
        )
    except Exception as exc:
        logger.warning("Résumé exécutif erreur : %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Pipeline principal (synchrone)
# ---------------------------------------------------------------------------

def _run_pipeline(params: dict) -> dict:
    """
    Exécute l'intégralité du pipeline War Room et retourne le résultat complet.
    `params` est le dictionnaire validé issu du payload POST.
    """
    timings: dict[str, float] = {}
    started_at = time.time()

    query = params["query"]
    model = params["model"]
    lang_rule = params["lang_rule"]
    num_experts = params["num_experts"]
    selected_experts = params["selected_experts"]
    debate_rounds = params["debate_rounds"]
    advocate_enabled = params["advocate"]
    reviewer_enabled = params["reviewer"]
    generate_tests = params["generate_tests"]
    exec_summary = params["executive_summary"]
    tokens_agent = params["tokens_agent"]
    tokens_programmer = params["tokens_programmer"]
    stack = params["stack"]
    session_id = params.get("session_id")

    # Historique de session (contexte des warrooms précédentes).
    session_context = ""
    if session_id and session_id in _session_store and _session_store[session_id]:
        prev = _session_store[session_id][-1]
        session_context = (
            f"\nCONTEXTE SESSION PRÉCÉDENTE :\n"
            f"- Demande précédente : {prev.get('query', '')[:300]}\n"
            f"- Synthèse résumée : {_safe_excerpt(prev.get('synthesis', ''), 800)}\n"
        )
        query_with_context = query + session_context
    else:
        query_with_context = query

    # Étape 0 : cadrage PM
    t0 = time.time()
    pm_data, timings["pm_ms"] = _build_pm_data(query_with_context, model, lang_rule)

    # Étape 1 : sélection d'experts
    t0 = time.time()
    experts, strategy = _select_experts(
        query_with_context, model, lang_rule, selected_experts, num_experts
    )
    timings["orchestrator_ms"] = (time.time() - t0) * 1000

    # Étape 2 : avis experts en parallèle
    t0 = time.time()
    proposals = _run_parallel_experts(query_with_context, model, lang_rule, experts, tokens_agent)
    timings["experts_ms"] = (time.time() - t0) * 1000

    # Étape 2b : débat inter-experts (optionnel)
    if debate_rounds > 0:
        t0 = time.time()
        proposals = _run_expert_debate(query_with_context, model, lang_rule, proposals, debate_rounds)
        timings["debate_ms"] = (time.time() - t0) * 1000

    # Étape 3 : synthèse
    synthesis, timings["synthesis_ms"] = _build_synthesis(
        query_with_context, pm_data, proposals, model, lang_rule
    )

    # Étape 3b : avocat du diable (optionnel)
    advocate_critique: str | None = None
    if advocate_enabled:
        t0 = time.time()
        advocate_critique = _run_advocate(query, synthesis, model, lang_rule)
        timings["advocate_ms"] = (time.time() - t0) * 1000

    # Étape 4 : génération de code
    programmer_prompt = _build_programmer_prompt(
        query, synthesis, lang_rule, advocate_critique, stack
    )
    t0 = time.time()
    try:
        programmer_output = call_llm_with_retry(
            [{"role": "user", "content": programmer_prompt}],
            model=model,
            max_tokens=tokens_programmer,
            step_name="Programmeur",
        )
        programmer_output = _complete_generation_if_needed(
            programmer_output, programmer_prompt, model=model
        )
    except Exception as exc:
        programmer_output = f"Erreur lors de la génération du code : {exc}"
    timings["programmer_ms"] = (time.time() - t0) * 1000

    # Étape 4b : reviewer de code (optionnel)
    review_text = ""
    corrections = []
    if reviewer_enabled:
        t0 = time.time()
        review_text, corrections = _run_code_review(query, programmer_output, model, lang_rule)
        timings["reviewer_ms"] = (time.time() - t0) * 1000
        if corrections:
            t0 = time.time()
            programmer_output = _apply_corrections_if_needed(
                programmer_output, corrections, query, synthesis,
                model, lang_rule, advocate_critique, stack,
            )
            timings["programmer_post_review_ms"] = (time.time() - t0) * 1000

    # Post-traitement : extraction structurée des fichiers + nettoyage
    files = _extract_files(programmer_output)
    clean_output = _clean_programmer_output(programmer_output)

    # Étape 5 : génération de tests (optionnel)
    tests_output = ""
    test_files: list[dict] = []
    if generate_tests and files:
        t0 = time.time()
        tests_output = _generate_tests(query, files, model, lang_rule)
        test_files = _extract_files(tests_output)
        timings["tests_ms"] = (time.time() - t0) * 1000

    # Étape 6 : résumé exécutif (optionnel)
    executive_summary_text = ""
    if exec_summary:
        t0 = time.time()
        executive_summary_text = _build_executive_summary(
            query, synthesis, programmer_output, model, lang_rule
        )
        timings["executive_summary_ms"] = (time.time() - t0) * 1000

    # Métriques de qualité
    complete = FINAL_CODE_END_MARKER in programmer_output
    total_chars = sum(len(f["content"]) for f in files)
    experts_ok = sum(1 for p in proposals if not p.get("error"))
    quality_score = round(
        (0.4 * (1 if complete else 0))
        + (0.3 * experts_ok / max(len(proposals), 1))
        + (0.3 * (1 if files else 0)),
        2,
    )
    token_estimate = {
        "pm": _estimate_tokens(json.dumps(pm_data)),
        "experts": sum(_estimate_tokens(p["proposal"]) for p in proposals),
        "synthesis": _estimate_tokens(synthesis),
        "programmer": _estimate_tokens(programmer_output),
    }

    timings["total_ms"] = int((time.time() - started_at) * 1000)

    result = {
        "pm": pm_data,
        "strategy": strategy,
        "experts": experts,
        "proposals": [
            {
                "expert": p["expert"],
                "text": p["proposal"],
                "error": p.get("error"),
                "debated": p.get("debated", False),
                "duration_ms": p.get("duration_ms", 0),
            }
            for p in proposals
        ],
        "synthesis": synthesis,
        "advocate_critique": advocate_critique,
        "programmer_output": clean_output,
        "files": files,
        "review_comments": review_text,
        "review_corrections": corrections,
        "tests_output": _clean_programmer_output(tests_output) if tests_output else "",
        "test_files": test_files,
        "executive_summary": executive_summary_text,
        "meta": {
            "query": query,
            "model": model,
            "stack_detected": stack,
            "num_experts": len(experts),
            "debate_rounds": debate_rounds,
            "advocate_enabled": advocate_enabled,
            "reviewer_enabled": reviewer_enabled,
            "tests_generated": bool(test_files),
            "programmer_complete": complete,
            "total_files": len(files),
            "total_code_chars": total_chars,
            "quality_score": quality_score,
            "experts_timeout_seconds": EXPERTS_TIMEOUT_SECONDS,
            "token_estimate": token_estimate,
            "token_budgets": {
                "pm": TOKENS_PM,
                "orchestrator": TOKENS_ORCHESTRATOR,
                "agent": tokens_agent,
                "synthesis": TOKENS_SYNTHESIS,
                "programmer": tokens_programmer,
                "continuation": TOKENS_CONTINUATION,
            },
            "timings_ms": {k: int(v) for k, v in timings.items()},
        },
    }

    # Stocker dans la session si demandé
    if session_id:
        if session_id not in _session_store:
            _session_store[session_id] = []
        _session_store[session_id].append({
            "query": query,
            "synthesis": synthesis,
            "timestamp": time.time(),
        })
        # Limiter à 10 entrées par session
        _session_store[session_id] = _session_store[session_id][-10:]

    return result


# ---------------------------------------------------------------------------
# Validation du payload entrant
# ---------------------------------------------------------------------------

def _parse_warroom_payload(data: dict) -> tuple[dict | None, str | None]:
    """
    Valide et normalise le payload POST.
    Retourne (params, None) si OK, (None, error_message) sinon.
    """
    query = data.get("query", "").strip()
    if not query:
        return None, "Requête vide"
    if len(query) > MAX_QUERY_LENGTH:
        return None, f"Requête trop longue (max {MAX_QUERY_LENGTH} caractères)"

    model = data.get("model", DEFAULT_MODEL)
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL

    lang = data.get("lang", "fr")
    lang_rule = LANG_RULES_SHORT.get(lang, LANG_RULES_SHORT["fr"])

    selected_experts_raw = data.get("selected_experts", [])
    if not isinstance(selected_experts_raw, list):
        selected_experts_raw = []
    selected_experts = [e for e in selected_experts_raw if e in EXPERT_PROFILES]

    # Appliquer le preset de mode si fourni
    mode = data.get("mode", "balanced")
    preset = MODE_PRESETS.get(mode, MODE_PRESETS["balanced"])

    params: dict = {
        "query": query,
        "model": model,
        "lang_rule": lang_rule,
        "selected_experts": selected_experts,
        "num_experts": int(data.get("num_experts", preset["num_experts"])),
        "debate_rounds": int(data.get("debate_rounds", preset["debate_rounds"])),
        "advocate": bool(data.get("advocate", preset["advocate"])),
        "reviewer": bool(data.get("reviewer", preset["reviewer"])),
        "generate_tests": bool(data.get("generate_tests", preset["generate_tests"])),
        "executive_summary": bool(data.get("executive_summary", preset["executive_summary"])),
        "tokens_agent": int(data.get("tokens_agent", preset["tokens_agent"])),
        "tokens_programmer": int(data.get("tokens_programmer", preset["tokens_programmer"])),
        "stack": _detect_stack(query, data.get("stack")),
        "session_id": data.get("session_id"),
        "webhook_url": data.get("webhook_url"),
    }

    params["num_experts"] = max(1, min(params["num_experts"], 6))
    return params, None


# ---------------------------------------------------------------------------
# Route principale : POST /api/warroom
# ---------------------------------------------------------------------------

@warroom_bp.route("/api/warroom", methods=["POST"])
def warroom():
    """Pipeline War Room complet."""
    ip = request.remote_addr or "unknown"
    if not _rate_check(ip):
        return jsonify({"error": "Rate limit dépassé. Réessayez dans 60s."}), 429

    data = request.get_json(force=True, silent=True) or {}
    params, err = _parse_warroom_payload(data)
    if err:
        return jsonify({"error": err}), 400

    # Cache hit ?
    cache_key = _cache_key(params["query"], params["model"], params["selected_experts"])
    cached = _cache_get(cache_key)
    if cached and not data.get("no_cache"):
        cached["meta"]["cache_hit"] = True
        return jsonify(cached)

    # Mode async avec webhook
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
                    http_requests.post(
                        params["webhook_url"],
                        json={"job_id": job_id, "result": result},
                        timeout=15,
                    )
                except Exception as exc:
                    logger.warning("Webhook delivery failed : %s", exc)
            except Exception as exc:
                _job_store[job_id]["status"] = "error"
                _job_store[job_id]["error"] = str(exc)

        threading.Thread(target=_async_run, daemon=True).start()
        return jsonify({"job_id": job_id, "status": "pending"}), 202

    # Mode synchrone
    result = _run_pipeline(params)
    result["meta"]["cache_hit"] = False
    _cache_set(cache_key, result)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Route streaming SSE : POST /api/warroom/stream
# ---------------------------------------------------------------------------

@warroom_bp.route("/api/warroom/stream", methods=["POST"])
def warroom_stream():
    """
    Pipeline War Room en Server-Sent Events.
    Le client reçoit chaque étape dès qu'elle est disponible.
    Events : pm_ready | experts_ready | debate_ready | synthesis_ready |
             advocate_ready | code_ready | review_ready | tests_ready | done | error
    """
    ip = request.remote_addr or "unknown"
    if not _rate_check(ip):
        def _err():
            yield "data: " + json.dumps({"event": "error", "message": "Rate limit dépassé."}) + "\n\n"

        return Response(stream_with_context(_err()), mimetype="text/event-stream")

    data = request.get_json(force=True, silent=True) or {}
    params, err = _parse_warroom_payload(data)
    if err:
        def _err():
            yield "data: " + json.dumps({"event": "error", "message": err}) + "\n\n"

        return Response(stream_with_context(_err()), mimetype="text/event-stream")

    def _sse(event: str, payload: dict) -> str:
        return "data: " + json.dumps({"event": event, **payload}) + "\n\n"

    @stream_with_context
    def _generate() -> Generator[str, None, None]:
        query = params["query"]
        model = params["model"]
        lang_rule = params["lang_rule"]
        stack = params["stack"]

        # PM
        pm_data, _ = _build_pm_data(query, model, lang_rule)
        yield _sse("pm_ready", {"pm": pm_data})

        # Experts
        experts, strategy = _select_experts(
            query, model, lang_rule, params["selected_experts"], params["num_experts"]
        )
        proposals = _run_parallel_experts(query, model, lang_rule, experts, params["tokens_agent"])
        yield _sse("experts_ready", {
            "experts": experts,
            "strategy": strategy,
            "proposals": [{"expert": p["expert"], "text": p["proposal"], "error": p.get("error")} for p in proposals],
        })

        # Débat
        if params["debate_rounds"] > 0:
            proposals = _run_expert_debate(query, model, lang_rule, proposals, params["debate_rounds"])
            yield _sse("debate_ready", {
                "proposals": [{"expert": p["expert"], "text": p["proposal"]} for p in proposals],
            })

        # Synthèse
        synthesis, _ = _build_synthesis(query, pm_data, proposals, model, lang_rule)
        yield _sse("synthesis_ready", {"synthesis": synthesis})

        # Avocat du Diable
        advocate_critique = None
        if params["advocate"]:
            advocate_critique = _run_advocate(query, synthesis, model, lang_rule)
            yield _sse("advocate_ready", {"advocate_critique": advocate_critique})

        # Code
        programmer_prompt = _build_programmer_prompt(
            query, synthesis, lang_rule, advocate_critique, stack
        )
        try:
            programmer_output = call_llm_with_retry(
                [{"role": "user", "content": programmer_prompt}],
                model=model,
                max_tokens=params["tokens_programmer"],
                step_name="Programmeur-stream",
            )
            programmer_output = _complete_generation_if_needed(
                programmer_output, programmer_prompt, model=model
            )
        except Exception as exc:
            programmer_output = f"Erreur : {exc}"

        files = _extract_files(programmer_output)
        yield _sse("code_ready", {
            "programmer_output": _clean_programmer_output(programmer_output),
            "files": files,
            "complete": FINAL_CODE_END_MARKER in programmer_output,
        })

        # Reviewer
        review_text = ""
        if params["reviewer"]:
            review_text, corrections = _run_code_review(query, programmer_output, model, lang_rule)
            if corrections:
                programmer_output = _apply_corrections_if_needed(
                    programmer_output, corrections, query, synthesis,
                    model, lang_rule, advocate_critique, stack,
                )
                files = _extract_files(programmer_output)
            yield _sse("review_ready", {
                "review_comments": review_text,
                "programmer_output": _clean_programmer_output(programmer_output),
                "files": files,
            })

        # Tests
        tests_output = ""
        if params["generate_tests"] and files:
            tests_output = _generate_tests(query, files, model, lang_rule)
            yield _sse("tests_ready", {
                "tests_output": _clean_programmer_output(tests_output),
                "test_files": _extract_files(tests_output),
            })

        # Résumé exécutif
        exec_sum = ""
        if params["executive_summary"]:
            exec_sum = _build_executive_summary(query, synthesis, programmer_output, model, lang_rule)

        yield _sse("done", {
            "executive_summary": exec_sum,
            "stack_detected": stack,
            "complete": FINAL_CODE_END_MARKER in programmer_output,
        })

    return Response(_generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Route itérative : POST /api/warroom/refine
# ---------------------------------------------------------------------------

@warroom_bp.route("/api/warroom/refine", methods=["POST"])
def warroom_refine():
    """
    Relance uniquement le programmeur sur une warroom existante avec des consignes supplémentaires.
    Économise 80% du coût en tokens par rapport à un pipeline complet.
    """
    ip = request.remote_addr or "unknown"
    if not _rate_check(ip):
        return jsonify({"error": "Rate limit dépassé."}), 429

    data = request.get_json(force=True, silent=True) or {}
    refinement = data.get("refinement", "").strip()
    synthesis = data.get("synthesis", "").strip()
    query = data.get("query", "").strip()
    model = data.get("model", DEFAULT_MODEL)
    lang = data.get("lang", "fr")

    if not refinement or not synthesis or not query:
        return jsonify({"error": "Champs requis : query, synthesis, refinement"}), 400
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL

    lang_rule = LANG_RULES_SHORT.get(lang, LANG_RULES_SHORT["fr"])
    stack = _detect_stack(query, data.get("stack"))

    refined_synthesis = synthesis + f"\n\nCONSIGNE DE RAFFINAGE :\n{refinement}"
    programmer_prompt = _build_programmer_prompt(query, refined_synthesis, lang_rule, stack=stack)

    t0 = time.time()
    try:
        programmer_output = call_llm_with_retry(
            [{"role": "user", "content": programmer_prompt}],
            model=model,
            max_tokens=int(data.get("tokens_programmer", TOKENS_PROGRAMMER)),
            step_name="Programmeur-refine",
        )
        programmer_output = _complete_generation_if_needed(
            programmer_output, programmer_prompt, model=model
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    files = _extract_files(programmer_output)
    return jsonify({
        "programmer_output": _clean_programmer_output(programmer_output),
        "files": files,
        "complete": FINAL_CODE_END_MARKER in programmer_output,
        "meta": {
            "refinement": refinement,
            "duration_ms": int((time.time() - t0) * 1000),
            "total_files": len(files),
        },
    })


# ---------------------------------------------------------------------------
# Route polling jobs : GET /api/warroom/jobs/<job_id>
# ---------------------------------------------------------------------------

@warroom_bp.route("/api/warroom/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    """Polling d'un job lancé en mode async (webhook)."""
    job = _job_store.get(job_id)
    if not job:
        return jsonify({"error": "Job introuvable"}), 404
    response = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
    }
    if job["status"] == "done":
        response["result"] = job["result"]
    elif job["status"] == "error":
        response["error"] = job.get("error", "Erreur inconnue")
    return jsonify(response)


# ---------------------------------------------------------------------------
# Route session : GET /api/warroom/sessions/<session_id>
# ---------------------------------------------------------------------------

@warroom_bp.route("/api/warroom/sessions/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Historique des warrooms d'une session."""
    history = _session_store.get(session_id, [])
    return jsonify({"session_id": session_id, "count": len(history), "history": history})


# ---------------------------------------------------------------------------
# Route cache : DELETE /api/warroom/cache
# ---------------------------------------------------------------------------

@warroom_bp.route("/api/warroom/cache", methods=["DELETE"])
def clear_cache():
    """Vide le cache des résultats."""
    count = len(_cache_store)
    _cache_store.clear()
    return jsonify({"cleared": count})


# ---------------------------------------------------------------------------
# Route catalogue des experts
# ---------------------------------------------------------------------------

@warroom_bp.route("/api/experts", methods=["GET"])
def get_experts_route():
    return jsonify({
        "families": EXPERT_FAMILIES,
        "experts": {k: {**v} for k, v in EXPERT_PROFILES.items()},
        "modes": list(MODE_PRESETS.keys()),
    })


# ---------------------------------------------------------------------------
# Route Avocat du Diable (standalone)
# ---------------------------------------------------------------------------

@warroom_bp.route("/api/advocate", methods=["POST"])
def advocate():
    ip = request.remote_addr or "unknown"
    if not _rate_check(ip):
        return jsonify({"error": "Rate limit dépassé."}), 429

    data = request.get_json(force=True, silent=True) or {}
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
        "Structure :\n"
        "## Failles critiques\n## Risques majeurs\n## Points faibles\n"
        "## Hypothèses dangereuses\n## Scénarios catastrophe\n## Mitigations\n\n"
        f"Sujet : {topic}"
    )
    try:
        result = call_llm_with_retry(
            [{"role": "user", "content": prompt}],
            model=model,
            max_tokens=3_000,
            step_name="Avocat-du-Diable",
        )
        return jsonify({"critique": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


# ---------------------------------------------------------------------------
# Route feedback : POST /api/warroom/feedback
# ---------------------------------------------------------------------------

_feedback_store: list[dict] = []


@warroom_bp.route("/api/warroom/feedback", methods=["POST"])
def submit_feedback():
    """Collecte la notation du code généré."""
    data = request.get_json(force=True, silent=True) or {}
    score = data.get("score")
    comment = data.get("comment", "")
    query = data.get("query", "")

    if score is None or not isinstance(score, (int, float)) or not (1 <= score <= 5):
        return jsonify({"error": "score requis entre 1 et 5"}), 400

    entry = {
        "score": float(score),
        "comment": str(comment)[:1000],
        "query": str(query)[:500],
        "timestamp": time.time(),
        "ip": request.remote_addr,
    }
    _feedback_store.append(entry)

    avg = sum(f["score"] for f in _feedback_store) / len(_feedback_store)
    return jsonify({
        "recorded": True,
        "avg_score": round(avg, 2),
        "total": len(_feedback_store),
    })


@warroom_bp.route("/api/warroom/feedback", methods=["GET"])
def get_feedback():
    """Stats de feedback agrégées."""
    if not _feedback_store:
        return jsonify({"avg_score": None, "total": 0, "recent": []})
    avg = sum(f["score"] for f in _feedback_store) / len(_feedback_store)
    return jsonify({
        "avg_score": round(avg, 2),
        "total": len(_feedback_store),
        "recent": _feedback_store[-10:],
    })
