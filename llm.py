from __future__ import annotations

import re
import urllib.parse
from typing import List, Dict

import requests

from config import API_KEY, OPENROUTER_URL, DEFAULT_MODEL, logger


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def _get_headers() -> dict:
    """Construit les headers d'autorisation pour LLM7."""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Appel LLM
# ---------------------------------------------------------------------------

def call_llm(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 500000,
) -> str:
    """Envoie une requête à LLM7 et retourne le texte de la réponse."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        OPENROUTER_URL,
        headers=_get_headers(),
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        err = result["error"]
        raise RuntimeError(
            f"IA indisponible ({err.get('code', '?')}): {err.get('message', '')}"
        )
    return result["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Recherche DuckDuckGo
# ---------------------------------------------------------------------------

def perform_search(query: str, max_results: int = 5) -> str:
    """
    Effectue une recherche via l'API DuckDuckGo Instant Answers.
    Retourne une chaîne formatée avec les résultats.

    Note : urllib.parse.quote() encode déjà la requête ; ne pas passer
    la valeur encodée dans le dict params (requests l'encodera une 2e fois).
    """
    try:
        # BUG FIX : on passe `query` brut — requests se charge de l'encodage.
        params: Dict[str, str] = {
            "q":              query,
            "format":         "json",
            "no_html":        "1",
            "skip_disambig":  "1",
        }
        response = requests.get(
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
