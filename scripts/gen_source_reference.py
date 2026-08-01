"""Generate SOURCE_CODE.md — a single markdown file containing all tracked source."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Extensions considered "source" to include
SOURCE_EXTS = {
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".cfg",
    ".ini",
    ".lock",
    ".txt",
    ".xml",
    ".sh",
}

# Directories to skip entirely
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".eggs",
    "build",
    "dist",
    ".pytest_cache",
    "htmlcov",
    "node_modules",
    "models",
    "checkpoints",
    ".mypy_cache",
    ".ruff_cache",
    "data",
}

# Individual files to skip (generated/large)
SKIP_FILES = {"SOURCE_CODE.md"}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def is_source(p: Path) -> bool:
    return (p.suffix in SOURCE_EXTS) and p.name not in SKIP_FILES


def main() -> None:
    lines = []
    lines.append("# Repository Source Code Reference")
    lines.append("")
    lines.append(
        "> Auto-generated reference of all source files in the repository. "
        "This file is **git-ignored** and never committed."
    )
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("")

    entries = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        if not is_source(p):
            continue
        entries.append(p)

    for p in entries:
        lines.append(f"- [{rel(p)}](#{rel(p).replace('/', '-').replace('.', '_')})")

    lines.append("")
    lines.append("---")
    lines.append("")

    for p in entries:
        anchor = rel(p).replace("/", "-").replace(".", "_")
        lines.append(f"<a id='{anchor}'></a>")
        lines.append(f"## {rel(p)}")
        lines.append("")
        lines.append("```" + (p.suffix.lstrip(".") if p.suffix else ""))
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            text = f"[unreadable: {e}]"
        lines.append(text.rstrip("\n"))
        lines.append("```")
        lines.append("")

    out = Path(ROOT / "SOURCE_CODE.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(entries)} files, {out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
