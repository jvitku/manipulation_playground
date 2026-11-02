"""Sync the 'Tech development' collapsibles in the force-VLA lessons from the master log.

Source of truth: docs/force_vla_school_techdev.md. Each lesson gets one
<details class="techdev"> block (inserted before <h3>Knobs</h3>, replaced in place on rerun)
holding the entries of its section(s). Run: python3 tools/techdev_sync.py [lesson.html ...]
"""

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "docs/force_vla_school_techdev.md"
LESSONS = ROOT / "lab/web/lessons"
# lesson file -> section-heading prefixes it shows (the plan-stage entries live with lesson 15)
MAP = {
    "force-sim.html": ("## Before any code", "## Lesson 15"),
    "nano-vla.html": ("## Lesson 16",),
    "nano-force-vla.html": ("## Lesson 17",),
    "nano-force-vla2.html": ("## Lesson 18",),
    "force-rl.html": ("## Lesson 19",),
    "lora-vla.html": ("## Lesson 20",),
}
SUMMARY = "Tech development &mdash; what broke, how it was found, what changed"
BLOCK = re.compile(r'<details class="techdev">.*?</details>\n?', re.S)


def inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*([^*\s][^*]*?)\*", r"<em>\1</em>", out)
    return out


def sections(md: str) -> dict[str, list[tuple[str, str]]]:
    secs: dict[str, list[tuple[str, str]]] = {}
    cur = None
    for para in re.split(r"\n\s*\n", md):
        para = para.strip()
        if para.startswith("## "):
            cur = para.splitlines()[0]
            secs[cur] = []
            continue
        m = re.match(r"\*\*(.+?)\*\*\s*(.*)", para, re.S)
        if cur and m:
            secs[cur].append((m.group(1), " ".join(m.group(2).split())))
    return secs


def render(entries: list[tuple[str, str]]) -> str:
    body = "\n".join(
        f'<p><b>{inline(title)}</b> {inline(text)}</p>' for title, text in entries
    )
    intro = (
        '<p class="ref">Every change that was needed to make this lesson\'s experiments and '
        "claims work, in the order it was found: symptom, how it was found, cause, change, "
        "take-away. Master copy: <code>docs/force_vla_school_techdev.md</code>.</p>"
    )
    return f'<details class="techdev"><summary>{SUMMARY}</summary>\n{intro}\n{body}\n</details>\n'


def main() -> None:
    import sys

    only = set(sys.argv[1:])  # optional lesson file names; default: all
    secs = sections(LOG.read_text())
    for name, prefixes in MAP.items():
        if only and name not in only:
            continue
        path = LESSONS / name
        if not path.exists():
            continue
        entries = [e for h, es in secs.items() if h.startswith(prefixes) for e in es]
        if not entries:
            continue
        s = path.read_text()
        block = render(entries)
        if BLOCK.search(s):
            s = BLOCK.sub(lambda _m: block, s, count=1)
        else:
            anchor = "<h3>Knobs</h3>"
            assert s.count(anchor) == 1, f"{name}: Knobs anchor"
            s = s.replace(anchor, block + anchor)
        path.write_text(s)
        print(f"{name}: {len(entries)} entries")


if __name__ == "__main__":
    main()
