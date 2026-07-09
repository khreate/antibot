# khronic — a self-directed Discord bot (local LLM)

A Discord bot backed by a locally-hosted uncensored 24B model (Dolphin-Mistral-24B-Venice-Edition via Ollama by default). Built to feel like a real chat participant instead of a request/response assistant:

- **Persona-first prompt** (`persona.py`) — the model plays a specific character, not "a helpful AI."
- **Per-user long-term memory** — SQLite-backed profiles built by periodic LLM summarization. Nothing leaves your machine.
- **Per-channel short-term transcript + mood** — the last N turns are replayed verbatim; mood tints the tone.
- **Self-directed** — three independent tick loops decide, on their own timers, whether to speak up, chime in on an active channel, or react to something with an emoji. Nothing here is gated on you sending a new message.
- **Consent-gated web search** (DuckDuckGo, no API key) — the bot asks before searching, never silently.
- **Image understanding** (Qwen-VL via Ollama) — when a message carries image attachments, a separate vision model describes them and the description is folded into context, so the persona can respond to pictures without ever talking to the vision model directly.
- **Prompt-injection defenses** — layered persona-level, sandwiched reminder, regex detection, optional LLM classifier, and send-time mention safety.
- **Threads** — the model can open a Discord thread when a topic clearly deserves its own space.
- **Message coalescing** — a burst of quick messages gets one thoughtful reply, not one-per-message.
- **Follow-up detection** — the LLM decides whether an unaddressed message is actually continuing a conversation with the bot, so no @mention is needed every turn.

For a deeper walkthrough of the architecture, see [docs.md](docs.md).

---

## 1. Run the model locally

You need [Ollama](https://ollama.com/download) running on the same machine.

```bash
ollama pull ikiru/Dolphin-Mistral-24B-Venice-Edition   # ~13GB, Q4-ish, 32K context
ollama run  ikiru/Dolphin-Mistral-24B-Venice-Edition   # optional: try it in the terminal first
```

Ollama exposes an API at `http://localhost:11434` — that's what the bot talks to.

**Higher quality (Q5_K_M / Q6_K)** — build your own Ollama model from a GGUF:

```bash
huggingface-cli download bartowski/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-GGUF \
  --include "*Q5_K_M.gguf" --local-dir ./models

cat > Modelfile <<'EOF'
FROM ./models/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q5_K_M.gguf
PARAMETER num_ctx 8192
EOF

ollama create khronic -f Modelfile
```

Then set `MODEL_NAME=khronic` in your `.env`.

**Optional: image understanding (vision).** Enabled by default (`VISION_ENABLED = True`). It uses a *separate* vision-capable model because Dolphin-Mistral is text-only. Pull one:

```bash
ollama pull qwen2.5vl:7b
```

The vision model is only ever invoked when an incoming message actually has image attachments, so text-only messages cost nothing extra. Its description is folded into the message text as plain context and the normal khronic persona handles the reply — the vision model never speaks to users. Override the model with `VISION_MODEL_NAME` in `.env`, or set `VISION_ENABLED = False` in `config.py` to turn the feature off entirely.

---

## 2. Create the Discord bot application

1. https://discord.com/developers/applications → **New Application**
2. **Bot** tab → **Reset Token**, copy it — this is your `DISCORD_TOKEN`
3. **Privileged Gateway Intents**: enable **Message Content Intent** (required — the bot can't see message text without it) and **Server Members Intent** (needed to resolve `@name` mentions the model writes)
4. **OAuth2 → URL Generator** → scopes: `bot` → permissions: `Send Messages`, `Read Message History`, `Read Messages/View Channels`, `Add Reactions`, `Create Public Threads`, `Send Messages in Threads` → open the generated URL to invite the bot

---

## 3. Install and run

**Linux / macOS:**

```bash
git clone <this-repo>
cd <this-repo>
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
cp .env.example .env             # paste your DISCORD_TOKEN into .env
cp config.example.py config.py   # tweak knobs / add channel restrictions here
python3 bot.py
```

**Windows (PowerShell):**

```powershell
git clone <this-repo>
cd <this-repo>
py -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env             # paste your DISCORD_TOKEN into .env
Copy-Item config.example.py config.py   # tweak knobs / add channel restrictions here
py bot.py
```

Optional: `cp examples.txt.example examples.txt` and paste in real conversation excerpts to give the model a few-shot voice reference. It's re-read on every reply, so tuning voice doesn't require a restart.

---

## 4. What it does once it's running

**When addressed** (@mentioned, replied to, DMed, or your name shows up as a whole word):
- Waits `COALESCE_WINDOW_SECONDS` to see if more messages land in the same burst.
- Generates one reply from the accumulated context. Prompt-injection safeguards run in parallel.
- The model can respond with `[SKIP]` to say nothing (rare on addressed turns), or lead with `[[thread: title]]` to spin up a Discord thread for the reply.

**Not addressed, but the bot recently spoke:**
- A follow-up check (LLM-judged by default) decides whether the new message is actually continuing the conversation. If yes, it responds normally.

**Not addressed at all:**
- The message is logged to memory for context, but no reply is generated from the arrival itself. Instead, three background loops decide independently whether to act:
  - **`attention_loop`** (every ~90s) — looks at recent messages in active channels and may add a single emoji reaction.
  - **`autonomy_loop` active mode** (every 5 min) — may chime in with a full unprompted message in an active channel.
  - **`autonomy_loop` idle mode** (every 5 min) — may say something after a channel's been quiet for a couple of hours.

**Current-info questions** (news, prices, weather, sports scores, etc.):
- The bot proposes a web search first ("want me to look that up?"). It only actually searches on the next message if the user agrees.

**Messages with images:**
- If vision is enabled and the message has image attachments, a vision model describes them first and the description is folded into the message text before anything else runs. By default this also happens for images posted in channels the bot is only watching (`VISION_ANALYZE_PASSIVE`), so it can reference them later; set that to `False` to only analyze images in messages that directly address the bot.

**In DMs:** it replies to every message. That's standard DM behavior.

All timings, probabilities, cooldowns, and channel restrictions live in `config.py`. Start conservative and turn the knobs up — an overeager bot that reacts to everything gets old fast.

---

## 5. Tuning the "sentience"

- **`persona.py` → `PERSONA_DESCRIPTION`**: the single highest-leverage lever. Specificity (speech tics, opinions, dislikes, things it wouldn't say) beats length every time.
- **`examples.txt`**: real conversation excerpts the model reads as a voice reference. Re-loaded per reply.
- **`config.py` → `AUTONOMY_*` / `ATTENTION_*`**: how often the self-directed loops fire, how likely each is to act, per-channel cooldowns.
- **`config.py` → memory depth**: `SHORT_TERM_TURNS` (verbatim context window), `SUMMARIZE_EVERY` (how often profiles are rewritten). Continuity comes from profiles carrying real specifics forward; inspect/edit `sentient_bot.db` in any SQLite browser if a summary drifts weird.
- **`config.py` → `GUILD_CHANNEL_RESTRICTIONS`**: whitelist channels per guild so the bot only responds in specific rooms.
- **Mood**: a short free-text phrase updated alongside summarization and injected into the system prompt. Not sticky/decaying yet; that's an easy upgrade if you want it.

---

## Known limitations worth knowing about

- **Profiles are per-user, not per-server.** If the bot is in multiple servers, a user's profile follows them everywhere. Add `guild_id` to the `profiles` table if you want it siloed (see [docs.md](docs.md#79-silo-memory-per-server)).
- **No LLM rate limiting.** A very active channel will serialize on Ollama; on one GPU that can lag under heavy load.
- **Uncensored model.** Venice Edition will genuinely follow whatever persona you write, including edgy ones. You're responsible for what it says in servers you invite it to.
- **Reactions have ~90s latency by design.** Because the reaction path is tick-driven, not message-triggered, the bot decides when to react on its own schedule. Lower `ATTENTION_CHECK_INTERVAL_SECONDS` if that feels too slow.
