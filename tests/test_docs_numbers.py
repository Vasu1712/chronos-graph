"""No number is typed into a document: the templates may contain none outside code, links and model names."""

import re
from pathlib import Path

import pytest

TEMPLATES = sorted((Path(__file__).resolve().parents[1] / "docs" / "templates").glob("*.tmpl.md"))
# a standalone number: not part of a word/identifier (Qwen2.5, 120b, v1.5, s01) and not a list marker
NUMBER = re.compile(r"(?<![\w.\-/#])\d[\d,]*(?:\.\d+)?%?(?![\w])")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s")


def prose(text: str) -> list[tuple[int, str]]:
    lines, fenced = [], False
    for i, ln in enumerate(text.splitlines(), start=1):
        if ln.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or ln.lstrip().startswith(">") and "Generated" in ln:
            continue
        ln = re.sub(r"`[^`]*`", " ", ln)  # inline code
        ln = re.sub(r"\{\{[^}]*\}\}", " ", ln)  # placeholders
        ln = re.sub(r"\]\([^)]*\)", "]", ln)  # link targets
        ln = re.sub(r"https?://\S+", " ", ln)
        if LIST_MARKER.match(ln):
            ln = re.sub(r"^\s*\d+[.)]\s", "", ln)
        lines.append((i, ln))
    return lines


def test_there_are_three_templates():
    assert [t.name for t in TEMPLATES] == ["DESIGN.tmpl.md", "EVAL.tmpl.md", "README.tmpl.md"]


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_templates_contain_no_hand_typed_numbers(path):
    bad = [
        (i, m.group(0), ln.strip()[:90])
        for i, ln in prose(path.read_text())
        for m in NUMBER.finditer(ln)
    ]
    assert not bad, "numbers must come from placeholders:\n" + "\n".join(
        f"  line {i}: {n!r} in {ln!r}" for i, n, ln in bad
    )


def test_the_lint_itself_catches_numbers_and_spares_names():
    assert NUMBER.findall("accuracy is 71.4% on 50 questions, k=60") == ["71.4%", "50", "60"]
    assert NUMBER.findall("Qwen2.5-7B, gpt-oss-120b, bge-small-en-v1.5, s01, M1, DeBERTa-v3") == []
    assert [m.group(0) for m in NUMBER.finditer("budget of {{budget}} tokens, 3 runs")] == ["3"]
