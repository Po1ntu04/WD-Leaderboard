from __future__ import annotations

import json
from pathlib import Path


SEGMENT_DELIMITER = "/"
SEGMENT_DELIMITER_PADDED = f" {SEGMENT_DELIMITER} "


def _escape_token(token: str) -> str:
    out: list[str] = []
    for ch in token:
        if ch in {"\\", "/", "／"}:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _unescape_token(token: str) -> str:
    out: list[str] = []
    escaped = False
    for ch in token:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        out.append(ch)
    if escaped:
        out.append("\\")
    return "".join(out)


def parse_segmented_line(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []

    whitespace_tokens = line.split()
    if whitespace_tokens:
        if (
            len(whitespace_tokens) >= 3
            and len(whitespace_tokens) % 2 == 1
            and all(whitespace_tokens[i] in {"/", "／"} for i in range(1, len(whitespace_tokens), 2))
        ):
            return [_unescape_token(whitespace_tokens[i]) for i in range(0, len(whitespace_tokens), 2)]
        if "/" not in line and "／" not in line:
            return [_unescape_token(tok) for tok in whitespace_tokens]

    out: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in line:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in {"/", "／"}:
            token = "".join(buf).strip()
            if token:
                out.append(_unescape_token(token))
            buf = []
            continue
        buf.append(ch)
    if escaped:
        buf.append("\\")
    token = "".join(buf).strip()
    if token:
        out.append(_unescape_token(token))
    return out


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_lines(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8-sig")
    return [line.rstrip("\n\r") for line in text.splitlines()]


def write_lines(path: str | Path, lines: list[str]) -> None:
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_raw_file(path: str | Path) -> list[str]:
    return [line.strip() for line in read_lines(path) if line.strip()]


def read_segmented_file(path: str | Path) -> list[list[str]]:
    return [parse_segmented_line(line) for line in read_lines(path)]


def write_segmented_file(path: str | Path, rows: list[list[str]]) -> None:
    write_lines(path, [SEGMENT_DELIMITER_PADDED.join(_escape_token(tok) for tok in row) for row in rows])


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
