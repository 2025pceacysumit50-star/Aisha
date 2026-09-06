"""
AISHA configuration.

Edit MODEL to match what you pulled with `ollama pull <name>`.
Edit COMPANY_NAME to your actual company. See README.md step 1
if you're not sure which model size fits your hardware.
"""

# Pick ONE — must match a model you've already pulled via Ollama.
# You confirmed a dedicated GPU with 12GB+ VRAM, so this is set to
# the stronger model — your hardware handles it comfortably.
MODEL = "qwen3:4b"       # default: fast + stable on this laptop (2.5 GB) — fixes the 'Read timed out' you saw on 14b
# MODEL = "llama3.2:3b"  # even faster fallback (2.0 GB)
# MODEL = "qwen3:14b"    # heavy — only if you have a strong GPU; 14b will be slow + may timeout on most laptops

OLLAMA_URL = "http://localhost:11434/api/chat"

# Powers the knowledge base (RAG) — small, CPU-friendly regardless
# of your hardware tier. Pull it with: ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

DB_PATH = "aisha.db"

COMPANY_NAME = "Black Hole"  # <-- change thi

AISHA_SYSTEM_PROMPT = """You are AISHA, the internal AI system for {company}.

Your job has two parts:
1. Business copilot — help with planning, research, drafting, and
   daily operational tasks. Be direct and concrete, not generic.
2. Threat triage — you will be shown a summary of unresolved
   security threats logged by the device-scanning layer. Help
   decide what's worth acting on, explain risk in plain terms, and
   never invent details about a threat you weren't given.

You run entirely offline on the founder's own hardware. Be honest
about your own limitations — you are a smaller local model, not a
frontier one, so for very complex reasoning it's fine to say so
plainly rather than bluff.
"""
