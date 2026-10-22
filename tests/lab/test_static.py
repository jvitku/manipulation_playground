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
    scripts = ("vendor/uplot.iife.min.js", "charts.js", "viewers.js", "app.js")
    order = [html.index(f"/static/{s}") for s in scripts]
    assert order == sorted(order)


def test_task1_lesson_served_with_schematic(client):
    r = client.get("/lessons/bc-mse.html")
    assert r.status_code == 200
    assert "<svg" in r.text
    assert "<html" not in r.text.lower()


def test_task1_spec_points_at_lesson(client):
    spec = client.get("/api/tasks/bc-mse").json()["spec"]
    assert spec["lesson_url"] == "/lessons/bc-mse.html"


def test_static_and_lessons_revalidate(client):
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
    assert client.get("/lessons/flow-bc.html").headers["cache-control"] == "no-cache"


@pytest.mark.parametrize(
    "path",
    [
        "vendor/katex/katex.min.js",
        "vendor/katex/katex.min.css",
        "vendor/katex/auto-render.min.js",
        "vendor/katex/fonts/KaTeX_Main-Regular.woff2",
    ],
)
def test_katex_assets(client, path):
    assert client.get(f"/static/{path}").status_code == 200


def test_index_loads_katex(client):
    html = client.get("/").text
    for s in ("vendor/katex/katex.min.css", "vendor/katex/katex.min.js", "vendor/katex/auto-render.min.js"):
        assert f"/static/{s}" in html


@pytest.mark.parametrize("lesson", ["bc-mse", "flow", "flow-bc"])
def test_lessons_have_display_equations(client, lesson):
    assert "$$" in client.get(f"/lessons/{lesson}.html").text
