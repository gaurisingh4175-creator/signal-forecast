from flask import Flask, jsonify, request, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from functools import wraps
from forecast_engine import run_forecast
from validation_engine import run_validation
from location_engine import get_locations, find_nearest_good_signal
from auth import register_user, verify_user
import os
import threading
import secrets

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
CORS(app, supports_credentials=True, origins=["http://localhost:5173"])

app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True,
)

UPLOAD_DIR = "uploads"
DEFAULT_FILE = os.path.join("data", "File A- signal_rsrp_India_Today_(2026-07-05)_detailed.xlsx")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_cache = {}
_lock = threading.Lock()


def clamp_days(value, default=7, lo=1, hi=10):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return wrapper


# ---------- Auth routes ----------
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = (data.get('username') or "").strip()
    password = data.get('password') or ""
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    ok, msg = register_user(username, password)
    if not ok:
        return jsonify({"error": msg}), 400

    session['username'] = username
    return jsonify({"message": msg, "username": username})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = (data.get('username') or "").strip()
    password = data.get('password') or ""

    if not verify_user(username, password):
        return jsonify({"error": "Invalid username or password"}), 401

    session['username'] = username
    return jsonify({"message": "Logged in successfully", "username": username})


@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({"message": "Logged out"})


@app.route('/api/me', methods=['GET'])
def me():
    if "username" in session:
        return jsonify({"username": session["username"]})
    return jsonify({"username": None}), 200


# ---------- Health check (public, no login needed) ----------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Flask backend is running"})


# ---------- Protected routes ----------
@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.endswith('.xlsx'):
        return jsonify({"error": "Only .xlsx files are supported"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)
    return jsonify({"message": "File uploaded successfully", "filename": filename})


@app.route('/api/forecast', methods=['GET'])
@login_required
def get_forecast():
    filename = request.args.get('filename')
    model = request.args.get('model', 'random_forest')
    train_days = clamp_days(request.args.get('train_days'), default=7)
    predict_days = clamp_days(request.args.get('predict_days'), default=7)

    file_path = os.path.join(UPLOAD_DIR, filename) if filename else DEFAULT_FILE
    cache_key = (filename or "default", model, train_days, predict_days)

    try:
        with _lock:
            if cache_key not in _cache:
                if model == 'prophet':
                    from prophet_engine import run_forecast_prophet
                    result = run_forecast_prophet(file_path, train_days=train_days, predict_days=predict_days)
                else:
                    result = run_forecast(file_path, train_days=train_days, predict_days=predict_days)
                _cache[cache_key] = result
        return jsonify(_cache[cache_key])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/forecast/refresh', methods=['POST'])
@login_required
def refresh_forecast():
    filename = request.args.get('filename')
    model = request.args.get('model', 'random_forest')
    train_days = clamp_days(request.args.get('train_days'), default=7)
    predict_days = clamp_days(request.args.get('predict_days'), default=7)

    file_path = os.path.join(UPLOAD_DIR, filename) if filename else DEFAULT_FILE
    cache_key = (filename or "default", model, train_days, predict_days)

    try:
        with _lock:
            if model == 'prophet':
                from prophet_engine import run_forecast_prophet
                result = run_forecast_prophet(file_path, train_days=train_days, predict_days=predict_days)
            else:
                result = run_forecast(file_path, train_days=train_days, predict_days=predict_days)
            _cache[cache_key] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/validate', methods=['GET'])
@login_required
def get_validation():
    filename = request.args.get('filename')
    file_path = os.path.join(UPLOAD_DIR, filename) if filename else DEFAULT_FILE
    try:
        result = run_validation(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/locations', methods=['GET'])
@login_required
def get_locations_route():
    filename = request.args.get('filename')
    file_path = os.path.join(UPLOAD_DIR, filename) if filename else DEFAULT_FILE
    try:
        locations = get_locations(file_path)
        return jsonify({"locations": locations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/nearest-good-signal', methods=['GET'])
@login_required
def nearest_good_signal_route():
    filename = request.args.get('filename')
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if lat is None or lon is None:
        return jsonify({"error": "lat and lon query parameters are required"}), 400

    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return jsonify({"error": "lat and lon must be valid numbers"}), 400

    file_path = os.path.join(UPLOAD_DIR, filename) if filename else DEFAULT_FILE
    try:
        result = find_nearest_good_signal(file_path, lat, lon)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/uploaded-files', methods=['GET'])
@login_required
def list_uploaded_files():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.xlsx')]
    return jsonify({"files": files})

# ---------- Signal Quality Classification ----------
@app.route('/api/classification', methods=['GET'])
@login_required
def get_classification():

    try:
        from classification_service import get_latest_predictions

        result = get_latest_predictions()

        return jsonify(result), 200

    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404

    except Exception as e:
        app.logger.exception("Classification API failed")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)