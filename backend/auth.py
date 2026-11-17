import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

USERS_FILE = "users.json"


def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username, password):
    users = _load_users()
    if username in users:
        return False, "Username already exists"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    users[username] = generate_password_hash(password)
    _save_users(users)
    return True, "Registered successfully"


def verify_user(username, password):
    users = _load_users()
    if username not in users:
        return False
    return check_password_hash(users[username], password)