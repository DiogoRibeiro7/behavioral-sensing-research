"""Tests for upload security in the web app."""
import io
import os

import pytest

from sensor_modeling.visualization import web_app


@pytest.fixture()
def app():
    """Provide a Flask test app with authentication configured."""
    os.environ["SM_USER"] = "user"
    os.environ["SM_PASS"] = "pass"
    return web_app.create_app()


def _auth():
    return {"Authorization": "Basic dXNlcjpwYXNz"}


def test_filename_sanitized(app, tmp_path):
    """Uploads are stored with sanitized names inside UPLOAD_DIR."""
    client = app.test_client()
    data = {"data": (io.BytesIO(b"hi"), "../../evil.txt"), "param": "x"}
    resp = client.post(
        "/run",
        data=data,
        headers=_auth(),
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    saved = web_app.UPLOAD_DIR / "evil.txt"
    assert saved.exists()


def test_size_limit(app):
    """Large uploads are rejected with HTTP 413."""
    client = app.test_client()
    big = io.BytesIO(b"a" * (2 * 1024 * 1024 + 1))
    data = {"data": (big, "big.txt"), "param": "x"}
    resp = client.post(
        "/run",
        data=data,
        headers=_auth(),
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
