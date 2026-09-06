# AISHA — Local AI System (v0.2)

Your own AI, running entirely on your own hardware. No API key, no
per-token cost, no data leaving your machine. Three layers now exist:

1. **Chat brain** (Ollama + Qwen3) — the base intelligence, downloaded
   once, runs fully offline.
2. **Knowledge base (RAG)** — a database of *your* business content
   that AISHA actually searches before answering. This is what makes
   it know your business, with no training run at all.
3. **Fine-tuning** (`training/`) — for when RAG isn't enough and you
   want to change how the model behaves, not just what it can look
   up. Needs a real GPU — you've already got one; see
   `training/README.md` before starting.

Layer 1 was v0.1. This version adds layer 2, and scaffolds layer 3.

## What's actually happening here

- **The model** (Qwen3, open-weight) runs locally via **Ollama** —
  a program that runs it on *your* CPU/GPU, exposed on `localhost`
  only. Nothing leaves your laptop.
- **The database** is one SQLite file (`aisha.db`), created
  automatically on first run. Conversations, tasks, threats, and now
  your knowledge base all live there, on your disk.
- **The knowledge base** is what makes AISHA genuinely *yours*: feed
  it business documents, past decisions, fraud patterns you've seen —
  it embeds them locally (also via Ollama, no API key) and searches
  them before every answer.
- **The classifier** (`malware_classifier/`) is a separate, much
  smaller model trained from scratch on labeled file data — not an
  LLM at all. See that folder's README for why that split matters.

## 1. Check your hardware first

Same as before — this decides model size, and now also whether local
fine-tuning is realistic.

**Windows**: `Ctrl+Shift+Esc` → Performance tab → look for a GPU
other than "Intel(R) UHD/Iris Graphics", note total RAM.

| You have | Chat model | Fine-tuning locally? |
|---|---|---|
| **Dedicated GPU, 12GB+ VRAM (you)** | `qwen3:14b` ← default | **Yes** — see `training/README.md` |
| Modest GPU, or 16GB+ RAM CPU-only | `qwen3:8b` | Not realistically — use cloud/Colab |
| Integrated graphics only | `qwen3:4b` | No — use cloud/Colab |

## 2. Install Ollama and pull two models

```
ollama pull qwen3:14b
ollama pull nomic-embed-text
```

The first is for chat — set for your confirmed hardware (12GB+
VRAM). The second is a small embedding model that powers the
knowledge base — tiny and CPU-friendly regardless of hardware tier.

## 3. Install Python dependencies

```
pip install -r requirements.txt
```

## 4. Edit config.py

Set `COMPANY_NAME`, confirm `MODEL` matches what you pulled.

## 5. Run AISHA

```
python aisha.py
```

Commands:
- `/task <text>` — log a business task
- `/tasks` — list open tasks
- `/threats` — list unresolved threats
- `/learn <text>` — teach AISHA a fact directly from chat
- `/exit` — quit

## 6. Feed it your actual business (the important part)

```
python ingest.py path/to/some_notes.txt
python ingest.py path/to/a_folder_of_docs/
```

Point this at anything real: past decisions, pricing logic, known
scam patterns, competitor notes, your actual business plan. AISHA
pulls relevant chunks into context automatically when you ask about
related things. This — not the base model — is the part a
competitor can't copy.

## 7. When you outgrow RAG: fine-tuning

RAG makes AISHA know facts. Fine-tuning changes how it *behaves* —
its judgment, tone, the specific calls it makes on ambiguous cases.
Most projects don't need this yet. When you do, `training/README.md`
walks through it with Unsloth, the current standard tool for LoRA
fine-tuning on a single consumer GPU.

## If it's too slow

Switch `MODEL` in `config.py` to `qwen3:8b` (or `qwen3:4b` if still
slow), pull that model, restart. Shouldn't be needed on your GPU,
but worth knowing.

## What's deliberately not in this version

- **Threat ingestion from the device model** — the `threat_log`
  table is ready; the Android app writing into it is the next build.
- **A web UI instead of the terminal** — straightforward to add later.
- **Automatic dataset curation** — `training/export_dataset.py`
  exports raw conversation history; reviewing it by hand before
  fine-tuning is still on you, deliberately.
