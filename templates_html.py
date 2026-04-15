"""
Charge les templates HTML depuis les fichiers dédiés.
En production, Flask pourrait utiliser le dossier templates/ à la place,
mais on conserve render_template_string pour rester compatible avec le code existant.
"""
from __future__ import annotations

import os

_BASE = os.path.dirname(__file__)


def _load(filename: str) -> str:
    path = os.path.join(_BASE, filename)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


LOGIN_TEMPLATE: str = _load("templates/login.html")
HTML_TEMPLATE:  str = _load("templates/chat.html")
