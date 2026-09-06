"""
Feed documents into AISHA's knowledge base — text files, notes,
past decisions, anything you want AISHA to actually know about your
business. This is the "data research" layer: what you put in here
is what makes this AI yours, not the base model weights.

Usage:
    python ingest.py path/to/file.txt
    python ingest.py path/to/folder/       (loads every .txt/.md file inside)
"""

import sys
from pathlib import Path

from aisha import init_db, add_knowledge


def ingest_path(conn, path):
    p = Path(path)
    files = (list(p.glob("*.txt")) + list(p.glob("*.md"))) if p.is_dir() else [p]

    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        # Crude chunking — split on blank lines. Good enough to start;
        # swap for smarter chunking once this matters at real scale.
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        for chunk in chunks:
            add_knowledge(conn, source=f.name, content=chunk)
        print(f"Ingested {len(chunks)} chunks from {f.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <file_or_folder>")
        sys.exit(1)

    connection = init_db()
    ingest_path(connection, sys.argv[1])
    print("Done. AISHA can now reference this in conversation.")
