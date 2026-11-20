#!/usr/bin/env python3
"""Create a source-only PHSM archive on the AutoDL machine.

This intentionally collects *code and text configuration only*. Review its
output file list before sharing. Run from /root/macon2.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import tarfile

ROOT = Path("/root/macon2")
OUTPUT = Path("/root") / f"PHSM_sources_{date.today():%Y%m%d}.tar.gz"
SOURCE_EXT = {".py", ".sh", ".md", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
SOURCE_NAMES = {"requirements.txt", "Dockerfile", ".gitignore"}
SOURCE_DIRS = {"phsm_pca", "phsm_coldstart", "macon_reproduction"}
SKIP_PARTS = {
    ".git", ".venv", "venv", "__pycache__", "data", "datasets",
    "InstagramImage", "checkpoints", "outputs", "runs", "logs",
    "wandb", ".ipynb_checkpoints",
}
SKIP_NAME_PARTS = ("backup", "secret", "credential", "password", "token")
MAX_SOURCE_SIZE = 2 * 1024 * 1024

if not ROOT.is_dir():
    raise SystemExit(f"Directory does not exist: {ROOT}")

files: list[Path] = []
for entry in ROOT.rglob("*"):
    if not entry.is_file():
        continue
    rel = entry.relative_to(ROOT)
    if any(part in SKIP_PARTS or part.startswith(".") for part in rel.parts[:-1]):
        continue
    if entry.name.startswith(".") and entry.name != ".gitignore":
        continue
    if any(s in entry.name.lower() for s in SKIP_NAME_PARTS):
        continue
    if entry.suffix.lower() not in SOURCE_EXT and entry.name not in SOURCE_NAMES:
        continue
    # Only source files under the known projects or root-level source files
    if len(rel.parts) > 1 and rel.parts[0] not in SOURCE_DIRS:
        continue
    if entry.stat().st_size > MAX_SOURCE_SIZE:
        continue
    files.append(entry)

if not files:
    raise SystemExit("No source files found; check ROOT / source directories.")

print(f"Sources found: {len(files)}")
for entry in sorted(files):
    print(entry.relative_to(ROOT))

with tarfile.open(OUTPUT, "w:gz") as tar:
    for entry in sorted(files):
        tar.add(entry, arcname=str(entry.relative_to(ROOT)))

print(f"\nCreated: {OUTPUT}")
print("IMPORTANT: Inspect the archive for confidential text and legacy scripts before uploading.")
