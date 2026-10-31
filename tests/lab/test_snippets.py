"""Lesson code snippets are excerpts of the real source: they may not drift from it."""

import html
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LESSONS = sorted((ROOT / "lab/web/lessons").glob("*.html"))
BLOCK = re.compile(r'<code\b[^>]*\bdata-src="([^"]+)"[^>]*>(.*?)</code>', re.S)
NEW_LESSONS = ("attention", "vit-mnist", "cross-attn", "find-digit", "tokenization")


def blocks(lesson: Path):
    return [(m.group(1), html.unescape(m.group(2))) for m in BLOCK.finditer(lesson.read_text())]


def test_lessons_found():
    assert len(LESSONS) >= 13


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda p: p.stem)
def test_snippets_are_real_source_lines(lesson):
    for src, text in blocks(lesson):
        path = ROOT / src
        assert path.is_file(), f"{lesson.name}: data-src {src} does not exist"
        have = {ln.strip() for ln in path.read_text().splitlines()}
        for line in text.splitlines():
            s = line.strip()
            if s and s != "# ...":
                assert s in have, f"{lesson.name}: line not in {src}: {s!r}"


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda p: p.stem)
def test_snippets_are_katex_safe(lesson):
    for _, text in blocks(lesson):
        assert "$$" not in text and "\\(" not in text


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda p: p.stem)
def test_details_code_wrap_a_sourced_block(lesson):
    txt = lesson.read_text()
    assert txt.count('<details class="code">') == len(blocks(lesson))
    assert txt.count("<details") == txt.count("</details>")


@pytest.mark.parametrize("name", NEW_LESSONS)
def test_new_lessons_have_at_least_three_snippets(name):
    assert len(blocks(ROOT / f"lab/web/lessons/{name}.html")) >= 3
