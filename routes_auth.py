from __future__ import annotations

import time
from contextlib import closing

from flask import Blueprint, jsonify, redirect, render_template_string, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from config import logger
from database import get_db

auth_bp = Blueprint("auth", __name__)

# Template HTML importé depuis templates/login.html (voir fichier dédié)
from templates_html import LOGIN_TEMPLATE


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@auth_bp.route("/login")
def login_page():
    if session.get("username"):
        return redirect("/")
    return render_template_string(LOGIN_TEMPLATE)


# ---------------------------------------------------------------------------
# API Auth
# ---------------------------------------------------------------------------

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

    session["username"] = username
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
        row = db.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()

    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Identifiants incorrects"}), 401

    session["username"] = username
    logger.info("User logged in: %s", username)
    return jsonify({"ok": True, "username": username})


@auth_bp.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@auth_bp.route("/api/auth/me", methods=["GET"])
def me():
    username = session.get("username")
    if not username:
        return jsonify({"logged_in": False})

    with closing(get_db()) as db:
        row = db.execute(
            "SELECT username, email, created_at FROM users WHERE username = ?", (username,)
        ).fetchone()

    if not row:
        session.clear()
        return jsonify({"logged_in": False})

    return jsonify({
        "logged_in":  True,
        "username":   row["username"],
        "email":      row["email"]      or "",
        "created_at": row["created_at"] or "",
    })
