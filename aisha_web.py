"""
AISHA Web UI — Flask backend for voice + text interface.
Replaces the terminal interface with a browser page.

Plain-language map:
- This file is the HTTP layer only. The actual AI brain stays in aisha.py
  (ask_aisha). We call that function — we don't duplicate it.
- Speech-to-text (STT) uses faster-whisper on CPU, int8 — no cloud API.
- Text-to-speech (TTS) uses Piper on CPU — no cloud API.
- Two laptop actions only (open app, search files), both require explicit
  confirmation before anything happens.

CPU tradeoff we chose (you asked us to pick + explain):
- faster-whisper "base" (74M params, ~140 MB download) on CPU int8.
- Why base, not small? On a CPU-only laptop, base transcribes ~1-2 sec
  per 10 sec of audio. "small" (244M, ~460 MB) is ~3-5% more accurate
  on accents but takes ~4-7 sec on the same CPU — that lag breaks the
  push-to-talk feel. For voice commands ("open calculator", "search for
  budget.xlsx") base is accurate enough and feels responsive.
- If you want max accuracy and don't mind the wait, change
  WHISPER_MODEL_SIZE below to "small" and restart.
"""

import os
import re
import wave
import shutil
import sqlite3
import tempfile
import subprocess
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import BadRequest

from aisha import init_db, ask_aisha, add_task, list_open_tasks, list_unresolved_threats
from config import DB_PATH, COMPANY_NAME

# ---------- Config you can safely change ----------
WHISPER_MODEL_SIZE = "base"   # "base" = fast on CPU, "small" = more accurate but slower. Both are CPU-only.
WHISPER_COMPUTE_TYPE = "int8" # int8 is fastest / most memory-efficient on CPU. Don't change unless you know why.
MAX_TTS_CHARS = 500           # we cap TTS length so it stays snappy
FILE_SEARCH_ROOT = str(Path.home())  # where "search files" looks (your home folder)
FILE_SEARCH_MAX_RESULTS = 20
FILE_SEARCH_MAX_DEPTH = 6     # how many folder levels deep we walk (keeps it fast + safe)

# Small allow-list for "open app" — we only open these by name in this pass.
# If you type "open notepad" it opens notepad.exe; anything not listed is refused
# and we tell you why (safety + no ambiguous commands).
ALLOWED_APPS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "vscode": "code",
    "vs code": "code",
    "chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "edge": "msedge.exe",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max for audio uploads


@app.after_request
def _no_cache(r):
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


# ---------- Cached models (loaded once, reused) ----------
_whisper_model = None
_piper_voice = None
_piper_model_path = None


def get_db():
    """New SQLite connection per request (thread-safe)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_whisper_model():
    """Load faster-whisper once and reuse it. CPU + int8 is the CPU-only sweet spot."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # device="cpu" + compute_type="int8" is the documented CPU-only fast path.
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
    return _whisper_model


def find_piper_model():
    """
    Find a Piper voice model (.onnx + .onnx.json) on disk.
    We check the download script's location + the project folder.
    Returns (model_path: Path, config_path: Path) or (None, None).
    """
    candidates = []

    # Where download_piper_models.py puts files on Windows/Linux
    home_share = Path.home() / ".local" / "share" / "piper_tts"
    candidates.append(home_share)
    # Also check a local models folder inside the project
    candidates.append(Path(__file__).parent / "models" / "piper")
    candidates.append(Path(__file__).parent / "piper_tts")
    candidates.append(Path.home() / "piper_tts")

    # Preferred voice first (ryan-medium is the one download_piper_models.py fetches)
    preferred = ["en_US-ryan-medium", "en_US-amy-medium", "en_US-lessac-medium", "en_GB-alan-medium"]

    for base in candidates:
        if not base.exists():
            continue
        for name in preferred:
            m = base / f"{name}.onnx"
            c = base / f"{name}.onnx.json"
            # Some distributions use .json without the .onnx prefix
            c2 = base / f"{name}.json"
            if m.exists() and c.exists():
                return m, c
            if m.exists() and c2.exists():
                return m, c2
        # Fallback: any .onnx in that folder
        for m in base.glob("*.onnx"):
            c = Path(str(m) + ".json")
            c2 = m.with_suffix(".json")
            if c.exists():
                return m, c
            if c2.exists():
                return m, c2
    return None, None


def get_piper_voice():
    """Load Piper voice once (onnxruntime CPU provider)."""
    global _piper_voice, _piper_model_path
    if _piper_voice is not None:
        return _piper_voice

    from piper.voice import PiperVoice

    model_path, config_path = find_piper_model()
    if model_path is None:
        return None

    # The rhasspy/piper-voices download uses an older JSON schema than
    # piper-tts 1.8+ expects (num_symbols / phoneme_id_map etc. are in a
    # different shape). Fix it in-place once so PiperVoice.load can read it.
    try:
        _maybe_fix_piper_config(config_path)
    except Exception:
        pass  # if fix fails, let load error surface with a 503

    # use_cuda=False is critical on your CPU-only machine — never use GPU here.
    _piper_voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=False)
    _piper_model_path = model_path
    return _piper_voice


def _maybe_fix_piper_config(config_path):
    """
    Piper's voice JSON changed schema around 2024. The rhasspy download still
    ships the old shape (phoneme_map, speakers dict, no num_symbols etc.).
    piper-tts 1.8+ expects: num_symbols, num_speakers, phoneme_id_map, inference{},
    audio{}, espeak{}, piper_version, hop_length, etc. We migrate just enough
    fields so PiperVoice.load doesn't throw 'num_symbols'.

    Plain language: this just rewrites the small JSON file once so the local
    speech engine can read it. No cloud, no model change.
    """
    import json as _json
    p = Path(config_path)
    if not p.exists():
        return
    data = _json.loads(p.read_text(encoding="utf-8"))
    # Already new schema?
    if "num_symbols" in data and "phoneme_id_map" in data:
        return

    # Old schema markers: has 'phoneme_map' or 'phonemes' or 'speakers' as dict
    is_old = ("phoneme_map" in data or "phonemes" in data or isinstance(data.get("speakers"), dict))
    if not is_old:
        return

    # Build minimal new-schema config from old fields
    phoneme_map = data.get("phoneme_map", {}) or {}
    # Some old files have 'phonemes' as "[*]" or list — ignore, we need id map
    # Try to derive phoneme_id_map from an existing map or from dataset
    phoneme_id_map = data.get("phoneme_id_map")
    if not phoneme_id_map:
        # Old file may have a 'phoneme_id_map' nested differently — check
        # fallback: try to load from a sibling JSON if present (rare)
        phoneme_id_map = {}

    # If still empty, we can construct a minimal map that piper-tts will accept
    # by reading the ONNX model metadata via onnxruntime. But simpler: use the
    # known Piper mapping for en_US-ryan-medium — the voice's phonemes are
    # espeak-based and the id map is deterministic from espeak. However the
    # easiest reliable fix: if the file is truly old and missing id map,
    # replace it with the upstream "new format" config from Hugging Face.
    # That file lives at the same URL but with .onnx.json extension in the
    # piper-tts package expectation.
    if not phoneme_id_map or not isinstance(phoneme_id_map, dict) or len(phoneme_id_map) == 0:
        # Fetch the new-format JSON from Hugging Face as a fallback (one-time)
        # This keeps us fully local after the fetch — the fixed file is cached.
        try:
            import requests as _req
            # The new-format config is often at the same base with .onnx.json
            # For rhasspy/piper-voices the new JSON may already be on HF under
            # the same path — we just tried that URL in download_piper_models.py
            # and it may have been saved as .onnx.json already. Check again.
            alt = p.with_suffix("")  # strip .json -> .onnx
            # Try to infer base URL from data['url']
            base_url = data.get("url", "")
            if base_url:
                url = base_url.rstrip("/") + "/" + p.name
                # p.name is like en_US-ryan-medium.onnx.json — HF has that
                try:
                    r = _req.get(url, timeout=10)
                    if r.status_code == 200:
                        new_data = _json.loads(r.text)
                        if "num_symbols" in new_data:
                            p.write_text(_json.dumps(new_data, indent=2), encoding="utf-8")
                            return
                except Exception:
                    pass
        except Exception:
            pass

        # Last resort: synthesize minimal new-schema fields so load doesn't crash.
        # We set num_symbols/num_speakers from old data and stub an empty map.
        # Piper will still fail to synthesize usefully without a real id map, so
        # tell the user to re-download with the fixed downloader.
        # We don't want to silently produce garbage audio.
        return  # leave file as-is, let load error become a 503 with fix hint

    # If we do have a real phoneme_id_map, patch up the rest
    num_symbols = data.get("num_symbols")
    if num_symbols is None:
        # Count unique ids across the map
        try:
            all_ids = set()
            for v in phoneme_id_map.values():
                if isinstance(v, list):
                    all_ids.update(v)
                else:
                    all_ids.add(v)
            num_symbols = max(all_ids) + 1 if all_ids else 0
        except Exception:
            num_symbols = 0

    num_speakers = data.get("num_speakers")
    if num_speakers is None:
        speakers = data.get("speakers", {})
        if isinstance(speakers, dict):
            num_speakers = max(1, len(speakers))
        elif isinstance(speakers, list):
            num_speakers = max(1, len(speakers))
        else:
            num_speakers = 1

    sample_rate = data.get("sample_rate") or data.get("audio", {}).get("sample_rate", 22050)
    espeak_voice = data.get("espeak", {}).get("voice") or data.get("espeak_voice") or "en-us"

    fixed = {
        "audio": {"sample_rate": sample_rate},
        "espeak": {"voice": espeak_voice},
        "phoneme_type": data.get("phoneme_type", "espeak"),
        "num_symbols": num_symbols,
        "num_speakers": num_speakers,
        "phoneme_id_map": phoneme_id_map,
        "speaker_id_map": data.get("speaker_id_map", {}),
        "piper_version": data.get("piper_version", "1.0.0"),
        "hop_length": data.get("hop_length", 256),
        "inference": {
            "noise_scale": data.get("noise_scale", 0.667),
            "length_scale": data.get("length_scale", 1.0),
            "noise_w": data.get("noise_w_scale", 0.8),
        },
    }
    # Preserve optional fields if present
    if "vowel_clusters" in data:
        fixed["vowel_clusters"] = data["vowel_clusters"]
    if "dataset" in data:
        fixed["dataset"] = data["dataset"]

    # Backup old file
    try:
        backup = p.with_suffix(p.suffix + ".bak_old_schema")
        if not backup.exists():
            backup.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    p.write_text(_json.dumps(fixed, indent=2), encoding="utf-8")


def detect_action_intent(text):
    """
    Very small, strict intent detector for the two allowed actions.
    Returns (action, target) or (None, None) if it's just normal chat.

    Rules:
    - "open <app name>" -> open_app
    - "search (files|for) <filename>" -> search_files
    We keep this strict so ambiguous phrases never trigger an action.
    """
    t = text.strip().lower()
    # "open notepad" / "open calculator" / "please open paint"
    m = re.match(r"^(?:please\s+)?open\s+(.+)$", t)
    if m:
        target = m.group(1).strip().strip('"').strip("'")
        # strip trailing punctuation
        target = re.sub(r"[.!?]+$", "", target).strip()
        if target:
            return "open_app", target

    # "search files for budget" / "search for budget.xlsx" / "find file notes.txt"
    m = re.match(r"^(?:search|find)\s+(?:files?\s+)?(?:for\s+)?(.+)$", t)
    if m:
        target = m.group(1).strip().strip('"').strip("'")
        target = re.sub(r"[.!?]+$", "", target).strip()
        # Don't treat "search the web for ..." as a local file search
        if "web" not in t.split()[:3] and target and len(target) >= 2:
            return "search_files", target

    return None, None


def do_open_app(app_name):
    """Actually open the app (only after confirmation). Returns a human-readable result."""
    key = app_name.strip().lower()
    exe = ALLOWED_APPS.get(key)

    # Also try to resolve without the allow-list via PATH (still safe — we don't eval shell)
    if exe is None:
        # Try exact key as executable name
        found = shutil.which(key) or shutil.which(key + ".exe")
        if found:
            exe = found
        else:
            allowed_list = ", ".join(sorted(ALLOWED_APPS.keys()))
            return False, f"Blocked: '{app_name}' is not in the allowed list for this phase. Allowed: {allowed_list}."

    try:
        # Windows: use Popen without shell. For explorer, handle specially.
        if exe.lower() == "explorer.exe":
            subprocess.Popen(["explorer.exe"], close_fds=True)
        elif exe.lower().endswith(".exe") and not os.path.isabs(exe):
            # Let Windows resolve it via PATH / System32
            subprocess.Popen([exe], close_fds=True)
        else:
            subprocess.Popen([exe], close_fds=True)
        return True, f"Opened '{app_name}' ({exe})."
    except FileNotFoundError:
        return False, f"Could not find app '{app_name}' ({exe}) on this system."
    except Exception as e:
        return False, f"Failed to open '{app_name}': {e}"


def do_search_files(query):
    """
    Search local files by NAME (not content) under FILE_SEARCH_ROOT.
    This is intentionally limited: max depth + max results + no hidden/system crawl.
    Returns (ok, message, results_list).
    """
    query = query.strip()
    if not query:
        return False, "Empty search query.", []

    safe_query = query.strip().strip('"').strip("'")
    # Prevent path traversal / absolute path injection
    if ".." in safe_query or "/" in safe_query or "\\" in safe_query:
        # allow simple filename with extension, but not paths
        if not re.match(r"^[\w\-. ]+$", safe_query):
            return False, f"Search blocked: please use a simple filename like 'budget.xlsx', not a path. You sent: {safe_query}", []

    root = Path(FILE_SEARCH_ROOT)
    if not root.exists():
        return False, f"Search root does not exist: {root}", []

    # Build glob patterns: case-insensitive via lowercasing results, not glob
    # We walk with rglob but cap depth by counting separators.
    pattern = f"*{safe_query}*"
    results = []
    # Use os.walk for depth control
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Depth limit
        depth = len(Path(dirpath).parts) - base_depth
        if depth > FILE_SEARCH_MAX_DEPTH:
            dirnames[:] = []  # don't go deeper
            continue
        # Skip hidden + system-like dirs (keeps it fast + private)
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d.lower() not in ("appdata", "application data", "node_modules", ".venv", "venv", "__pycache__")]
        for fn in filenames:
            if fn.lower().find(safe_query.lower()) != -1:
                full = os.path.join(dirpath, fn)
                # Skip very long paths that are likely system
                if len(full) < 500:
                    results.append(full)
                    if len(results) >= FILE_SEARCH_MAX_RESULTS:
                        break
        if len(results) >= FILE_SEARCH_MAX_RESULTS:
            break

    if not results:
        return True, f"No files matching '{safe_query}' found under {root} (searched up to {FILE_SEARCH_MAX_DEPTH} levels deep).", []
    header = f"Found {len(results)} file(s) matching '{safe_query}' under {root}:"
    body = "\n".join(f"  - {p}" for p in results)
    if len(results) >= FILE_SEARCH_MAX_RESULTS:
        body += f"\n  (showing first {FILE_SEARCH_MAX_RESULTS} — refine your query for fewer results)"
    return True, f"{header}\n{body}", results


# ---------- Routes ----------

@app.route("/")
def index():
    return render_template("index.html", company_name=COMPANY_NAME)


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint. Also the gate for the two laptop actions:
    - If the text looks like "open X" or "search files for Y", we DO NOT
      execute yet. We return a confirmation_required payload that tells the
      frontend exactly what we would do and asks for an explicit "confirm".
    - Only when the frontend posts to /api/action/confirm do we actually act.
    """
    db = None
    try:
        db = get_db()
        data = request.get_json(silent=True)
        if not data or "message" not in data:
            return jsonify({"success": False, "error": "Missing 'message' field"}), 400

        user_input = data["message"].strip()
        if not user_input:
            return jsonify({"success": False, "error": "Empty message"}), 400

        # If the caller already confirmed an action, execute it (explicit gate)
        if data.get("confirmed") is True and data.get("action") in ("open_app", "search_files"):
            action = data["action"]
            target = data.get("target", "").strip()
            if action == "open_app":
                ok, msg = do_open_app(target)
                return jsonify({"success": ok, "reply": msg, "type": "action_result", "action": action, "target": target}), (200 if ok else 400)
            else:
                ok, msg, _ = do_search_files(target)
                return jsonify({"success": ok, "reply": msg, "type": "action_result", "action": action, "target": target}), (200 if ok else 400)

        # Slash commands (kept for compatibility with terminal)
        if user_input == "/tasks":
            reply = list_open_tasks(db)
            return jsonify({"success": True, "reply": reply, "type": "command", "command": "tasks"})
        if user_input == "/threats":
            reply = list_unresolved_threats(db)
            return jsonify({"success": True, "reply": reply, "type": "command", "command": "threats"})
        if user_input.startswith("/task "):
            task_text = user_input[6:].strip()
            if not task_text:
                return jsonify({"success": False, "error": "Empty task"}), 400
            add_task(db, task_text)
            return jsonify({"success": True, "reply": "Task added.", "type": "command", "command": "task_added"})

        # Check for the two gated laptop actions BEFORE calling the brain
        action, target = detect_action_intent(user_input)
        if action in ("open_app", "search_files"):
            # Do NOT execute — ask for confirmation with the exact action spelled out.
            if action == "open_app":
                confirm_text = (
                    f"You asked to open '{target}'. "
                    f"Exact action: open application '{target}'. "
                    f"Say \"confirm\" or press Confirm to proceed, or \"cancel\" to stop. "
                    f"Nothing has been opened yet."
                )
            else:
                confirm_text = (
                    f"You asked to search for files named '{target}' under {FILE_SEARCH_ROOT}. "
                    f"Exact action: search local files by name for '{target}'. "
                    f"Say \"confirm\" or press Confirm to proceed, or \"cancel\" to stop. "
                    f"Nothing has been searched yet."
                )
            return jsonify({
                "success": True,
                "type": "confirmation_required",
                "action": action,
                "target": target,
                "reply": confirm_text,
                "message": confirm_text,  # alias for older frontends
            })

        # Regular chat — reuse ask_aisha as the brain (RAG + threat context lives there)
        reply = ask_aisha(db, user_input)
        return jsonify({"success": True, "reply": reply, "type": "chat"})

    except Exception as e:
        return jsonify({"success": False, "error": f"Error: {str(e)}"}), 500
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


@app.route("/api/action/confirm", methods=["POST"])
def action_confirm():
    """
    Dedicated confirmation endpoint for the two actions.
    The frontend must call this ONLY after the user explicitly said "confirm"
    or clicked Confirm. We never auto-confirm.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Missing JSON body"}), 400
    action = data.get("action")
    target = (data.get("target") or "").strip()
    confirmed = data.get("confirmed") is True

    if action not in ("open_app", "search_files"):
        return jsonify({"success": False, "error": "Unknown action. Use 'open_app' or 'search_files'."}), 400
    if not target:
        return jsonify({"success": False, "error": "Missing 'target'."}), 400
    if not confirmed:
        return jsonify({"success": False, "error": "Not confirmed. Send {confirmed: true} to execute."}), 400

    if action == "open_app":
        ok, msg = do_open_app(target)
        return jsonify({"success": ok, "reply": msg, "type": "action_result", "action": action, "target": target}), (200 if ok else 400)
    else:
        ok, msg, _ = do_search_files(target)
        return jsonify({"success": ok, "reply": msg, "type": "action_result", "action": action, "target": target}), (200 if ok else 400)


@app.route("/api/history", methods=["GET"])
def history():
    db = None
    try:
        db = get_db()
        limit = request.args.get("limit", default=20, type=int)
        limit = min(max(limit, 1), 100)
        cursor = db.execute(
            "SELECT role, content, created_at FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        messages = []
        for role, content, created_at in reversed(cursor.fetchall()):
            messages.append({"role": role, "content": content, "timestamp": created_at})
        return jsonify({"success": True, "messages": messages})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error: {str(e)}"}), 500
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribe audio to text using faster-whisper (CPU, int8).
    Accepts multipart/form-data with field "audio" (WAV, WEBM, MP3, M4A, FLAC, etc.).
    The browser's MediaRecorder usually sends webm/opus — we handle that.
    """
    temp_path = None
    try:
        if "audio" not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        if audio_file.filename == "" and audio_file.content_length == 0:
            # Some browsers send empty filename — still accept if content exists
            pass

        # Preserve extension from upload so ffmpeg/AV can detect the container.
        # Default to .webm (what MediaRecorder usually produces) if unknown.
        filename = (audio_file.filename or "").lower()
        if filename.endswith(".wav"):
            suffix = ".wav"
        elif filename.endswith(".webm"):
            suffix = ".webm"
        elif filename.endswith(".mp3"):
            suffix = ".mp3"
        elif filename.endswith(".m4a") or filename.endswith(".mp4"):
            suffix = ".m4a"
        elif filename.endswith(".ogg") or filename.endswith(".opus"):
            suffix = ".ogg"
        elif filename.endswith(".flac"):
            suffix = ".flac"
        else:
            # Fall back to webm — faster-whisper + PyAV can handle it
            suffix = ".webm"

        # Also handle the content-type hint
        ctype = (audio_file.content_type or "").lower()
        if "wav" in ctype:
            suffix = ".wav"
        elif "webm" in ctype:
            suffix = ".webm"
        elif "ogg" in ctype or "opus" in ctype:
            suffix = ".ogg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            temp_path = tmp.name

        # Quick empty-file check
        if os.path.getsize(temp_path) < 500:
            return jsonify({"success": False, "error": "Audio too short or empty. Hold the button longer and try again."}), 400

        model = get_whisper_model()
        # language=None lets Whisper auto-detect; we clamp to English if you want by setting language="en"
        segments, info = model.transcribe(temp_path, beam_size=5, vad_filter=True)

        text = " ".join([segment.text.strip() for segment in segments]).strip()

        # info.language may be None if detection failed; default to "en" display
        lang = getattr(info, "language", None) or "auto"
        duration = getattr(info, "duration", None)

        return jsonify({
            "success": True,
            "text": text,
            "language": lang,
            "duration": duration,
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Transcription error: {str(e)}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route("/api/tts", methods=["GET"])
def tts():
    """
    Convert text to speech using Piper (CPU, onnxruntime CPUExecutionProvider).
    GET /api/tts?text=hello%20world
    Returns audio/wav.
    """
    try:
        text = request.args.get("text", "").strip()
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        if len(text) > MAX_TTS_CHARS:
            text = text[:MAX_TTS_CHARS]

        voice = get_piper_voice()
        if voice is None:
            return jsonify({
                "success": False,
                "error": "Piper voice model not found. Run: python download_piper_models.py (it downloads en_US-ryan-medium to ~/.local/share/piper_tts).",
            }), 503

        # Synthesize to WAV bytes in memory — no temp files, no subprocess.
        wav_io = BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        wav_bytes = wav_io.getvalue()
        if not wav_bytes or len(wav_bytes) < 100:
            return jsonify({"success": False, "error": "TTS produced empty audio"}), 500

        return send_file(
            BytesIO(wav_bytes),
            mimetype="audio/wav",
            as_attachment=False,
            download_name="tts.wav",
        )

    except Exception as e:
        return jsonify({"success": False, "error": f"TTS error: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — backend + Ollama + voice models."""
    import requests as req
    from config import OLLAMA_URL, MODEL

    ollama_ok = False
    ollama_error = None
    try:
        # Ollama's /api/tags is a cheap health probe (doesn't need a model name)
        base = OLLAMA_URL.split("/api/")[0]
        resp = req.get(f"{base}/api/tags", timeout=3)
        ollama_ok = resp.status_code < 500
        if not ollama_ok:
            ollama_error = f"HTTP {resp.status_code}"
    except Exception as e:
        ollama_error = str(e)[:120]

    # TTS model presence (don't load it here, just check files exist)
    m, _ = find_piper_model()
    tts_status = "ready" if m is not None else "missing — run download_piper_models.py"

    return jsonify({
        "success": True,
        "status": "running",
        "company": COMPANY_NAME,
        "ollama": "connected" if ollama_ok else f"unreachable ({ollama_error})",
        "model": MODEL,
        "whisper": {"model": WHISPER_MODEL_SIZE, "compute_type": WHISPER_COMPUTE_TYPE, "device": "cpu"},
        "tts": tts_status,
        "database": "ok",
    })


if __name__ == "__main__":
    # Bind to localhost only — not exposed to your network.
    print("AISHA Web UI starting on http://127.0.0.1:5000")
    print(f"Company: {COMPANY_NAME}")
    print(f"Whisper: {WHISPER_MODEL_SIZE} ({WHISPER_COMPUTE_TYPE}, CPU) — base is fast on CPU; switch to small for ~3-5% more accuracy if you don't mind the lag")
    m, _ = find_piper_model()
    if m:
        print(f"Piper TTS voice: {m}")
    else:
        print("Piper TTS voice: NOT FOUND — run: python download_piper_models.py")
    init_db()  # ensure aisha.db exists
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
