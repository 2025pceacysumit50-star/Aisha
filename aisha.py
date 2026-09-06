"""
AISHA — local AI business + threat-analysis assistant, now with a
real knowledge base (RAG). Runs entirely on your own hardware via
Ollama. No API key, no cloud.
"""

import json
import sqlite3
import requests
import numpy as np
from datetime import datetime, timezone

from config import (
    MODEL, OLLAMA_URL, EMBED_MODEL, OLLAMA_EMBED_URL,
    DB_PATH, COMPANY_NAME, AISHA_SYSTEM_PROMPT,
)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        status TEXT DEFAULT 'open',
        created_at TEXT NOT NULL
    )""")
    # The device model will eventually write here directly.
    conn.execute("""CREATE TABLE IF NOT EXISTS threat_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT,
        threat_type TEXT,
        confidence REAL,
        details TEXT,
        created_at TEXT NOT NULL,
        resolved INTEGER DEFAULT 0
    )""")
    # Your knowledge base — this is what makes AISHA actually "yours".
    conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        content TEXT NOT NULL,
        embedding TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.commit()
    return conn


def now():
    return datetime.now(timezone.utc).isoformat()


def load_recent_history(conn, limit=20):
    rows = conn.execute(
        "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def save_message(conn, role, content):
    conn.execute(
        "INSERT INTO conversations (role, content, created_at) VALUES (?, ?, ?)",
        (role, content, now()),
    )
    conn.commit()


def add_task(conn, title):
    conn.execute(
        "INSERT INTO tasks (title, created_at) VALUES (?, ?)", (title, now())
    )
    conn.commit()


def list_open_tasks(conn):
    rows = conn.execute(
        "SELECT id, title FROM tasks WHERE status='open' ORDER BY id"
    ).fetchall()
    if not rows:
        return "No open tasks."
    return "\n".join(f"[{i}] {t}" for i, t in rows)


def list_unresolved_threats(conn, limit=10):
    rows = conn.execute(
        """SELECT threat_type, confidence, details, created_at
           FROM threat_log WHERE resolved=0
           ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    if not rows:
        return "No unresolved threats logged."
    return "\n".join(
        f"- {t} (confidence {c:.2f}, {ts}): {d or 'no details'}"
        for t, c, d, ts in rows
    )


def embed_text(text):
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def add_knowledge(conn, source, content):
    vector = embed_text(content)
    conn.execute(
        "INSERT INTO knowledge (source, content, embedding, created_at) VALUES (?, ?, ?, ?)",
        (source, content, json.dumps(vector), now()),
    )
    conn.commit()


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def search_knowledge(conn, query, top_k=3):
    try:
        query_vec = embed_text(query)
    except requests.exceptions.RequestException:
        return "(Knowledge base unavailable — run: ollama pull nomic-embed-text)"

    rows = conn.execute("SELECT source, content, embedding FROM knowledge").fetchall()
    if not rows:
        return "No knowledge base entries yet — use /learn or ingest.py to add some."

    scored = []
    for source, content, emb_json in rows:
        vec = json.loads(emb_json)
        score = cosine_similarity(query_vec, vec)
        scored.append((score, source, content))
    scored.sort(reverse=True, key=lambda x: x[0])

    return "\n\n".join(f"[{s:.2f}] ({src}): {c}" for s, src, c in scored[:top_k])


def ask_aisha(conn, user_input):
    history = load_recent_history(conn)
    system = (
        AISHA_SYSTEM_PROMPT.format(company=COMPANY_NAME)
        + "\n\nRelevant knowledge from your own database:\n"
        + search_knowledge(conn, user_input, top_k=3)
        + "\n\nUnresolved threats right now:\n"
        + list_unresolved_threats(conn, limit=5)
    )
    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": user_input}
    ]

    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=60,
    )
    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama doesn't have '{MODEL}' downloaded yet.\n"
            f"    Run this in a terminal and wait for it to fully finish:\n"
            f"    ollama pull {MODEL}"
        )
    resp.raise_for_status()
    reply = resp.json()["message"]["content"]

    save_message(conn, "user", user_input)
    save_message(conn, "assistant", reply)
    return reply


def main():
    conn = init_db()
    print(f"AISHA is running locally (model: {MODEL}).")
    print("Commands: /task <text>   /tasks   /threats   /learn <text>   /exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/tasks":
            print("\n" + list_open_tasks(conn) + "\n")
            continue
        if user_input == "/threats":
            print("\n" + list_unresolved_threats(conn) + "\n")
            continue
        if user_input.startswith("/task "):
            add_task(conn, user_input[len("/task "):].strip())
            print("Task added.\n")
            continue
        if user_input.startswith("/learn "):
            try:
                add_knowledge(conn, source="chat", content=user_input[len("/learn "):].strip())
                print("Learned.\n")
            except requests.exceptions.RequestException:
                print("Couldn't reach the embedding model. Run: ollama pull nomic-embed-text\n")
            continue

        try:
            reply = ask_aisha(conn, user_input)
            print(f"\nAISHA: {reply}\n")
        except requests.exceptions.ConnectionError:
            print(
                "\n[Error] Can't reach Ollama on localhost:11434. "
                "Is it running? Try `ollama serve` in another terminal.\n"
            )
            break
        except RuntimeError as e:
            print(f"\n[Error] {e}\n")
            continue


if __name__ == "__main__":
    main()
