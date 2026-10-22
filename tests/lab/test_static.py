"""The server serves the frontend shell and its static assets."""

import pytest
from fastapi.testclient import TestClient

from lab.server.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_index_has_app_div(client):
    r = client.get("/")
    assert r.status_code == 200
    assert '<div id="app">' in r.text


@pytest.mark.parametrize(
    "path",
    [
        "app.js",
        "charts.js",
        "viewers.js",
        "style.css",
        "vendor/uplot.iife.min.js",
        "vendor/uplot.min.css",
    ],
)
def test_static_assets(client, path):
    assert client.get(f"/static/{path}").status_code == 200


def test_index_references_scripts_in_order(client):
    html = client.get("/").text
    order = [html.index(f"/static/{s}") for s in ("vendor/uplot.iife.min.js", "charts.js", "viewers.js", "app.js")]
    assert order == sorted(order)
