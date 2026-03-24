"""Lightweight Flask web interface for running analyses."""
from __future__ import annotations

import argparse
import os
from functools import wraps
from pathlib import Path
from typing import Callable

from flask import (
    Flask,
    Response,
    abort,
    render_template_string,
    request,
    send_file,
)
from werkzeug.utils import secure_filename

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _check_auth(username: str, password: str) -> bool:
    """Validate provided credentials against environment variables."""
    expected_user = os.environ.get("SM_USER")
    expected_pass = os.environ.get("SM_PASS")
    if expected_user is None or expected_pass is None:
        return False
    return username == expected_user and password == expected_pass


def _auth_failed() -> Response:  # pragma: no cover - simple response
    """Return a 401 response requesting basic authentication."""
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="Login"'},
    )


def requires_auth(f: Callable) -> Callable:
    """Enforce basic authentication on endpoints."""

    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not _check_auth(auth.username, auth.password):
            return _auth_failed()
        return f(*args, **kwargs)

    return decorated


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB uploads


@app.route("/", methods=["GET"])
@requires_auth
def index():  # pragma: no cover - simple template
    """Render a minimal upload form."""
    return render_template_string(
        "<h1>Sensor Modeling</h1>"
        "<form action='{{ url_for(\"run\") }}' method='post' "
        "enctype='multipart/form-data'>"
        "<input type='file' name='data'>"
        "<input type='text' name='param'>"
        "<input type='submit'>"
        "</form>"
    )


@app.route("/run", methods=["POST"])
@requires_auth
def run():
    """Persist uploaded data and return a placeholder result."""
    file = request.files["data"]
    param = request.form.get("param", "")
    filename = secure_filename(file.filename)
    if not filename:
        abort(400, "invalid filename")
    path = (UPLOAD_DIR / filename).resolve()
    if path.parent != UPLOAD_DIR.resolve():  # path traversal guard
        abort(400, "invalid path")
    file.save(path)
    # Placeholder analysis step
    result_path = path.with_suffix(".txt")
    with open(result_path, "w", encoding="utf-8") as fh:
        fh.write(f"processed with param={param}\n")
    return send_file(result_path, as_attachment=True)


def create_app() -> Flask:
    """Create the Flask application."""
    return app


def main() -> None:
    """Run the Flask development server for the web app."""
    parser = argparse.ArgumentParser(description="Run the sensor visualization app")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the Flask development server in debug mode",
    )
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
