# khronic — developer docs

Deeper walkthrough of how the bot is wired together. For install/setup, see [README.md](README.md).

---

## 1. What it is

khronic is a Discord bot backed by a locally-hosted LLM (Dolphin-Mistral-24B-Venice-Edition via Ollama by default). Unlike a standard "assistant" bot it maintains a **persona**, a **per-user long-term memory profile**, a **per-channel short-term transcript**, a **mood** state, and **self-directed behavior** — three background loops decide on their own timers whether to react, chime in on an active channel, or say something after a long quiet. The goal is a Discord *participant* that feels continuous across time, not a request/response tool.

Central design principle: **the bot's initiative is not gated on incoming messages.** Reactions and unprompted chime-ins are driven by ticks, not `on_message`. The only message-triggered paths are the ones that must be (`@mention` / reply / DM / name-triggered / conversation follow-up).

---

## 2. File map

| File | Role |
| --- | --- |
| [bot.py](bot.py) | Discord event loop, response routing, background loops, coalescer. The "controller." |
| [config.py](config.py) | Environment loading + all tunable knobs (probabilities, cooldowns, memory sizes, model name). Git-ignored — a `config.example.py` template ships in the repo. |
| [persona.py](persona.py) | All prompt templates: main persona, `sandwich_messages`, summarization, mood, attention/follow-up decisions, web-search prompts, injection classifier. The "identity." |
| [memory.py](memory.py) | `Memory` class — SQLite-backed persistence for messages, per-user profiles, and per-channel state (mood, last-summarized watermark, last message timestamp). |
| [llm_client.py](llm_client.py) | Thin async wrapper around the Ollama `/api/chat` endpoint. `chat(messages, temperature=?, max_tokens=?, repeat_penalty=?) -> str`. |
| [web_search.py](web_search.py) | DuckDuckGo text search via the `ddgs` package. Runs off-loop with a hard timeout; returns `[]` on any failure so callers treat it as best-effort. |
| [guard.py](guard.py) | Prompt-injection detection: cheap regex first-pass + optional LLM classifier. Detection only — the persona is the actual defense. |
| [.env](.env) | `DISCORD_TOKEN` and optional overrides. Git-ignored; template is `.env.example`. |
| [examples.txt](examples.txt) | Optional few-shot voice examples appended to the system prompt. Git-ignored; template is `examples.txt.example`. |
| `sentient_bot.db` | SQLite file created at runtime (path from `DB_PATH`). |
| [README.md](README.md) | User-facing install/setup guide. |

---

## 3. Runtime architecture

```mermaid
flowchart TD
    D[Discord Gateway] -->|on_message| B[bot.on_message]
    B --> AC{addressed?<br/>@mention / reply / DM /<br/>name / follow-up}
    AC -- yes --> INJ[_check_injection<br/>fire-and-forget]
    AC -- yes --> COA[coalesce buffer]
    COA -->|after COALESCE_WINDOW| GR[generate_reply]
    AC -- no --> TR[track message in memory]
    GR --> WEB{needs web?}
    WEB -- ask --> ASK[reply: 'want me to look?']
    WEB -- context --> LLM[llm_client.chat]
    WEB -- none --> LLM
    GR --> LLM
    LLM --> OLL[(Ollama<br/>local model)]
    LLM --> THR{[[thread:]]<br/>directive?}
    THR -- yes --> MKTHR[create thread]
    LLM --> SEND[send_message]
    GR --> SUM[maybe_summarize<br/>fire-and-forget]
    SUM --> LLM

    ATT[attention_loop<br/>every ~90s] --> HIST[channel.history]
    HIST --> LLM
    LLM --> REACT[add_reaction]

    AUT[autonomy_loop<br/>every 5 min] --> MODE{idle vs active}
    MODE -- idle --> LLM
    MODE -- active --> LLM

    MEM[(SQLite<br/>Memory)]
    GR <--> MEM
    TR --> MEM
    SUM --> MEM
    AUT <--> MEM
    ATT <--> MEM
```

### Response paths, ranked by directness

1. **Directly addressed** (`should_respond` in [bot.py](bot.py) — mention, reply-to-bot, DM, or name-triggered): the message is stored to memory immediately, then buffered for `COALESCE_WINDOW_SECONDS`. When the window closes, `_coalesced_reply` generates one reply from the accumulated context. `_check_injection` runs in parallel via `asyncio.create_task` so it never delays the reply.
2. **Follow-up detection** (`is_conversation_followup`): if the bot has spoken recently in an active channel and someone sends an unaddressed message, a cheap recency gate fires and then an LLM call (`FOLLOWUP_DECISION_PROMPT`) makes the final YES/NO call on whether the message is actually continuing a conversation with the bot. Set `FOLLOWUP_USE_LLM = False` to fall back to a pure "did the bot send the last message" heuristic.
3. **Attention loop** (`attention_loop`, every `ATTENTION_CHECK_INTERVAL_SECONDS`): per active channel per tick, fetches recent Discord history via `channel.history()`, filters out messages already considered (`_last_attention_msg_id` watermark) and bot messages, and asks the LLM `REACT <n> <emoji>` or `NONE`. If REACT, adds that emoji to that specific message.
4. **Autonomy loop** (`autonomy_loop`, every `AUTONOMY_CHECK_INTERVAL_MINUTES`) — two modes per channel per tick:
   - **Idle mode**: channel quiet for ≥ `AUTONOMY_MIN_IDLE_MINUTES`. Roll `AUTONOMY_CHANCE`. This is the "haven't heard from you in a while" nudge.
   - **Active mode**: channel active within `AUTONOMY_ACTIVE_MAX_IDLE_MINUTES`, bot hasn't self-chimed here in `AUTONOMY_ACTIVE_MIN_SINCE_LAST_BOT_MINUTES`. Roll `AUTONOMY_ACTIVE_CHANCE`. This is the "just felt like saying something" chime-in that isn't wired to any specific incoming message. Skips DMs (they already always get replies).

Both autonomy modes get an `AUTONOMY_PROMPT_SUFFIX` that gives the model an explicit `[SKIP]` escape hatch. Direct replies also honor `[SKIP]` — the persona allows it rarely.

### Coalescer

`_coalesced_reply` is the debounce worker. When an addressed message arrives:

1. It's stored to memory (`add_message` returns a row id).
2. Pushed onto `_pending_msgs[channel_id]` along with the row id.
3. If no debounce task is running for the channel, one is spawned.
4. The debounce task sleeps `COALESCE_WINDOW_SECONDS`, takes the `_channel_lock`, drains the buffer, and generates a single reply from the accumulated context.
5. The buffer is re-checked in a loop so messages arriving mid-generation get folded into a follow-up batch instead of racing.
6. The buffer-empty check and task removal happen with no `await` between them, so a late-arriving enqueue can't slip through and get stranded.

Storing row ids alongside messages is what lets the thread path move the triggering messages into the newly-created thread (`mem.reassign_messages`) so the parent channel isn't left with a dangling unanswered request.

---

## 4. Memory model

`Memory` (in [memory.py](memory.py)) is the only thing that touches the SQLite DB. Everything else goes through these methods:

| Method | Purpose |
| --- | --- |
| `init()` | Create tables. Called from `setup_hook` (runs before gateway events, so no race with early messages). |
| `add_message(channel_id, user_id, username, role, content) -> row_id` | Append to the channel transcript. Returns the new row id so the coalescer can reassign it if a thread is opened. |
| `reassign_messages(message_ids, new_channel_id)` | Move stored rows into another channel's context (used when a reply opens a thread). |
| `get_recent_messages(channel_id, n)` | Last `n` turns as `(role, content, username)` tuples in chronological order. |
| `messages_since(channel_id, after_id)` | Fetch new messages after a watermark. Used by the summarizer. |
| `latest_message_id(channel_id)` / `set_last_summarized_id(...)` | Summarization watermark. |
| `get_profile(user_id)` / `upsert_profile(user_id, text)` | Long-term per-user profile blob (bullet-point summary). |
| `get_channel_state(channel_id)` | Returns `{"mood", "last_message_ts", "last_summarized_id"}`. Auto-inserts a default row on first read. |
| `set_mood(channel_id, mood)` | Overwrite mood string. |
| `all_known_channels()` | Iterated by `autonomy_loop` and `attention_loop`. |

### Two tiers of memory

- **Short-term (verbatim)**: last `SHORT_TERM_TURNS` (default 20) messages per channel, replayed into every prompt as-is with the speaker's name prefixed so the model can distinguish multiple users.
- **Long-term (summarized)**: every `SUMMARIZE_EVERY` (default 30) new messages, `maybe_summarize` sends the batch plus the current profile to the LLM with `SUMMARIZATION_SYSTEM_PROMPT` and replaces the profile with a fresh 3–6 bullet summary. This is what gives the bot cross-conversation "continuity."

### Mood

Short free-text phrase (e.g. `"curious and a bit playful"`) stored per-channel. Updated in the *same* summarization pass via `MOOD_SYSTEM_PROMPT` and injected into the system prompt on future turns. Currently **per-channel** and **replace-on-update** — no decay, no per-turn nudging, no cross-channel blending. Easy upgrade if you want it.

### Caveats

- Profiles are keyed on `user_id` only, not `(guild_id, user_id)` — a user's memory follows them across every server the bot is in.
- No LLM rate limiting; a bursty channel serializes on one GPU.
- `_channel_locks`, `_pending_msgs`, `_debounce_tasks`, `_last_thread_ts`, `_last_autonomous_ts`, `_last_attention_msg_id`, `_pending_web_search` are all in-memory only — they reset on restart, which is intentional.

---

## 5. Prompt construction

Every persona-driven LLM call is built from `build_system_prompt(mood, profile_summary, channel_notes=None)` in [persona.py](persona.py):

```
PERSONA_DESCRIPTION
+ current mood
+ (per-user profile summary, or a "no prior memory" note)
+ (optional channel notes — currently never passed)
+ formatting rules ("respond with ONLY your in-character message, concise")
+ [SKIP] escape-hatch instructions
+ thread-creation rules ([[thread: title]] directive)
+ @-mention rules
+ optional few-shot voice examples (from examples.txt, re-read every call)
+ INJECTION_GUARD (last, so recency-weighted attention keeps it in view)
```

Then `sandwich_messages(messages)` inserts `INJECTION_REMINDER_SHORT` as a system-role message immediately before the final user turn — the closer a reminder sits to the generation point, the harder it is to drown out with a long conversation.

Path-specific suffixes/overrides layered on top:

| Path | Extra prompt |
| --- | --- |
| Direct reply (`generate_reply`) | Just `build_system_prompt` + short-term transcript + latest user turn (+ web results block if the user approved a search). |
| Idle / active autonomy (`autonomy_loop`) | `build_system_prompt` + `AUTONOMY_PROMPT_SUFFIX` (explicit `[SKIP]` for "nothing feels natural"). |
| Attention reaction (`attention_loop`) | `ATTENTION_DECISION_PROMPT` + mood, with a numbered list of recent messages. Returns `REACT <n> <emoji>` or `NONE`. Does **not** use `build_system_prompt` — it's a narrow decision call. |
| Follow-up decision (`llm_followup_decision`) | `FOLLOWUP_DECISION_PROMPT` only. YES/NO. |
| Web-query gate (`_build_search_query`) | `WEB_QUERY_PROMPT`. Returns a concise search query or `NO_SEARCH`. |
| Search-consent judge (`_is_affirmative`) | `SEARCH_CONSENT_PROMPT`. YES/NO. |
| Injection classifier (`guard.is_injection_attempt_llm`) | `INJECTION_CLASSIFIER_PROMPT`. YES/NO. |
| Summarization (`maybe_summarize`) | `SUMMARIZATION_SYSTEM_PROMPT` only — does **not** use the persona. |
| Mood update (`maybe_summarize`) | `MOOD_SYSTEM_PROMPT` only. |

**Design principle**: the persona is only injected when the bot is *speaking as itself*. Utility calls (summarize, mood, YES/NO decisions, search gates) use narrow task-specific prompts so the persona doesn't leak into internal reasoning, and they don't get sandwiched.

---

## 6. Configuration reference

All in [config.py](config.py). Env vars override defaults via `.env` (loaded by `python-dotenv`). A committable template lives at [config.example.py](config.example.py); the real `config.py` is git-ignored so personal channel IDs and knob values don't leak.

### Environment
- `DISCORD_TOKEN` — required, from the Discord developer portal.
- `OLLAMA_HOST` — default `http://localhost:11434`.
- `MODEL_NAME` — default `ikiru/Dolphin-Mistral-24B-Venice-Edition`.
- `DB_PATH` — default `sentient_bot.db`.

### Memory
- `SHORT_TERM_TURNS` (20) — verbatim recent context window per channel.
- `SUMMARIZE_EVERY` (30) — messages between profile rewrites.

### Follow-ups
- `NAME_TRIGGERS` (`["khronic"]`) — whole-word names the bot answers to.
- `FOLLOWUP_WINDOW_SECONDS` (50) — recency window for the follow-up gate. 0 disables the whole path.
- `FOLLOWUP_USE_LLM` (True) — LLM makes the YES/NO call. When False, falls back to "did the bot send the last message."

### Idle autonomy
- `AUTONOMY_ENABLED` (True)
- `AUTONOMY_CHECK_INTERVAL_MINUTES` (5) — poll cadence for both idle and active modes.
- `AUTONOMY_MIN_IDLE_MINUTES` (120) — channel must be quiet this long before idle mode is eligible.
- `AUTONOMY_CHANCE` (0.35) — coin-flip once eligible.
- `AUTONOMY_CHANNEL_WHITELIST` (`[]`) — empty means "any channel the bot has spoken in." Applies to **both** loops.

### Active-channel autonomy
- `AUTONOMY_ACTIVE_ENABLED` (True)
- `AUTONOMY_ACTIVE_CHANCE` (0.13) — per-tick per-eligible-channel probability.
- `AUTONOMY_ACTIVE_MIN_SINCE_LAST_BOT_MINUTES` (15) — skip if the bot self-chimed here recently. This is what stops self-looping.
- `AUTONOMY_ACTIVE_MAX_IDLE_MINUTES` (30) — channel must have activity within this window.

### Attention loop (reactions)
- `ATTENTION_ENABLED` (True)
- `ATTENTION_CHECK_INTERVAL_SECONDS` (90) — tick cadence.
- `ATTENTION_LOOKBACK` (15) — how many recent Discord messages to show the model per tick.
- `ATTENTION_MAX_IDLE_MINUTES` (5) — channel must have activity within this window.
- `ATTENTION_CHANCE` (0.25) — per-tick per-eligible-channel probability of even asking the LLM.

### Generation
- `TEMPERATURE` (0.9), `MAX_TOKENS` (400) — defaults for `llm_client.chat`. Per-call overrides are used by decision-style calls (temp 0.0, tight max_tokens).
- `REPEAT_PENALTY` (1.15) — maps to Ollama's option of the same name. Nudge up if the model loop-repeats phrases; down toward 1.0 if replies feel stilted.
- `EXAMPLES_PATH` (`"examples.txt"`) — optional few-shot voice examples. Set to `None`/`""` to disable.

### Web lookup
- `WEB_SEARCH_ENABLED` (True)
- `WEB_SEARCH_MAX_RESULTS` (5), `WEB_SEARCH_REGION` (`"wt-wt"`), `WEB_SEARCH_TIMEOUT_SECONDS` (8.0)
- `WEB_SEARCH_USE_LLM_GATE` (True) — let the model refine the query / veto pointless lookups via `NO_SEARCH`.

### Message shaping
- `MAX_MESSAGE_CHARS` (300) — soft cap per Discord message; sentence-boundary aware.
- `MAX_REPLY_MESSAGES` (6) — safety valve so a runaway reply can't flood a channel.
- `MESSAGE_SEND_DELAY_SECONDS` (0.6) — pacing between chunks so a burst reads naturally.
- `COALESCE_WINDOW_SECONDS` (2.5) — debounce for addressed replies so a burst is answered together.

### Threads
- `THREAD_CREATION_ENABLED` (True), `THREAD_CREATION_COOLDOWN_SECONDS` (300)

### Guild channel restrictions
- `GUILD_CHANNEL_RESTRICTIONS: dict[int, list[int]]` — per-guild channel allowlist. Any guild not in the map is unrestricted. Threads whose parent is an allowed channel are auto-permitted.

### Prompt-injection safeguards
- `INJECTION_DETECTION_ENABLED` (True) — master switch for `_check_injection`. Turn off only when debugging.
- `INJECTION_LLM_CLASSIFIER_ENABLED` (False) — enable the paraphrase-catching classifier. Off by default because it's one extra LLM call per addressed message.

---

## 7. How to add new features

### 7.1 Tweak the persona

Edit `PERSONA_DESCRIPTION` in [persona.py](persona.py). The single highest-leverage change in the whole codebase — specificity (speech tics, opinions, dislikes) matters far more than length. No other code changes needed.

You can also drop conversation excerpts into `examples.txt` for a few-shot voice reference; the loader strips Discord chrome (timestamps, `(edited)` markers) and re-reads on every reply.

### 7.2 Add a slash/prefix command

The bot uses `commands.Bot(command_prefix="!khronic ")` and calls `await bot.process_commands(message)` inside `on_message`, so new prefix commands drop in with a decorator:

```python
@bot.command(name="mood")
async def show_mood(ctx: commands.Context):
    state = await mem.get_channel_state(ctx.channel.id)
    await ctx.send(f"current mood: {state['mood']}")
```

For proper slash commands add `bot.tree.command(...)` and call `await bot.tree.sync()` in `on_ready`.

### 7.3 Add a new tunable

1. Add the constant to [config.example.py](config.example.py) (with an env-var override via `os.getenv` if it should be deployment-configurable) and copy the same line into your local `config.py`.
2. Import and reference it from [bot.py](bot.py).
3. Update section 6 above.

### 7.4 Add a new "personality" prompt

If you're adding a new *kind* of LLM call (not a persona rewrite), put the prompt template in [persona.py](persona.py) alongside the existing constants and import it in [bot.py](bot.py). Follow the naming pattern `*_SYSTEM_PROMPT` for utility calls and `*_DECISION_PROMPT` for yes/no-style ones that expect a sentinel like `[SKIP]`, `NONE`, `NO_SEARCH`, or `YES`/`NO`.

### 7.5 Add a new self-directed behavior

Two options:

**Extend `autonomy_loop`** — best when the new behavior is "post a message when X." Add a new mode alongside idle/active with its own eligibility gate and chance, mirroring the pattern in the existing function.

**New tick loop** — best when the behavior has a different cadence or needs Discord API data the DB doesn't have (like `attention_loop` needing actual `Message` objects to react to). Copy the shape of `attention_loop`:

```python
@tasks.loop(seconds=config.MYFEATURE_INTERVAL_SECONDS)
async def myfeature_loop():
    ...

@myfeature_loop.before_loop
async def before_myfeature():
    await bot.wait_until_ready()
```

Start it from `on_ready` with the `.is_running()` guard so reconnects don't error, and keep a watermark dict at the top of `bot.py` if the loop should avoid reconsidering the same items.

### 7.6 Add a new addressed-path trigger

Extend `should_respond` in [bot.py](bot.py) or the follow-up gate. Whatever you add flows into the coalescer automatically — you don't need to touch the reply pipeline.

### 7.7 Wire injection detection into behavior

`_check_injection` currently only logs. To make a hit change behavior, the smallest step is: on a regex hit, append a short system-role note to the outgoing prompt in `generate_reply` reminding the model that the last turn tried to override its instructions. Don't refuse — the persona already handles that in-character.

### 7.8 Swap the model or backend

[llm_client.py](llm_client.py) is the only place that talks to Ollama. To point at a different backend (OpenAI-compatible endpoint, llama.cpp server, remote API), reimplement `chat(messages, *, temperature=None, max_tokens=None, repeat_penalty=None) -> str` there. Nothing else in the codebase needs to know.

### 7.9 Silo memory per-server

Currently `profiles(user_id)` is global. To scope it per guild:

1. Add `guild_id` to the profiles table schema in [memory.py](memory.py) and change the primary key to `(guild_id, user_id)`.
2. Update `get_profile` / `upsert_profile` signatures to take `guild_id`.
3. In [bot.py](bot.py), pass `message.guild.id` (or `None` for DMs) through `generate_reply` and `maybe_summarize`.

### 7.10 Add tools / function-calling

Not currently supported — the bot chats and (with consent) searches the web. If you want general tools:

1. Change `llm_client.chat` to return structured tool-call output as well as text.
2. In `generate_reply`, loop: run tool → append result as a `role: "tool"` message → re-invoke until the model produces a final assistant message.
3. Register tool implementations in a new `tools.py`, keyed by tool name.

Be careful: an uncensored 24B model with arbitrary tool access is a much bigger blast radius than a chatty one. Sandbox anything that touches the network or filesystem.

---

## 8. Local development tips

- Ollama must be running (`ollama serve` or the desktop app) before starting the bot. Test the model directly with `ollama run <model>` first — if it's slow standalone it'll be slow in Discord.
- To inspect what the bot "remembers," open `sentient_bot.db` in any SQLite browser. You can hand-edit a profile if a summary drifts weird — the code always reads fresh on each turn.
- Increase log verbosity by changing `logging.basicConfig(level=logging.INFO)` at the top of [bot.py](bot.py) to `DEBUG`. All autonomy paths already log when they fire and when they skip.
- To exercise autonomy quickly during dev:
  - **Idle**: temporarily set `AUTONOMY_CHECK_INTERVAL_MINUTES = 1`, `AUTONOMY_MIN_IDLE_MINUTES = 1`, `AUTONOMY_CHANCE = 1.0`.
  - **Active**: set `AUTONOMY_ACTIVE_CHANCE = 1.0`, `AUTONOMY_ACTIVE_MIN_SINCE_LAST_BOT_MINUTES = 0`.
  - **Attention**: set `ATTENTION_CHECK_INTERVAL_SECONDS = 15`, `ATTENTION_CHANCE = 1.0`.
- To exercise coalescing, drop `COALESCE_WINDOW_SECONDS` to a smaller number or push it up to see the buffer fill.
- `examples.txt` is re-read on every reply, so you can tune voice without restarting the bot.

---

## 9. Security / operational notes

- `DISCORD_TOKEN` must only ever live in `.env` (git-ignored). Never commit it, never log it.
- The Venice Edition model is deliberately uncensored — whatever persona you write, it will follow. You're responsible for what it says in servers you invite it to.
- The bot listens with `intents.message_content = True`, so it sees every message in every channel it can view. Only invite it to servers where that's acceptable, and use `GUILD_CHANNEL_RESTRICTIONS` to scope which channels it will respond in.
- `AllowedMentions(everyone=False, roles=False, users=True)` is set on the `commands.Bot` constructor as a global default. Even a jailbroken model can never emit `@everyone` / `@here` / role pings.
- SQLite has no auth; treat `sentient_bot.db` as sensitive (it contains user message summaries).
- Web search hits DuckDuckGo directly from your machine and only after the user says yes. No queries leave the machine without an explicit affirmative reply to the consent question.

---

## 10. Prompt-injection defenses

Defense-in-depth across three layers. None are perfect on their own — the point is that an attempt has to defeat all three to succeed.

### 10.1 Persona-level (the real defense)

Two constants in [persona.py](persona.py) do the heavy lifting:

- `INJECTION_GUARD` — appended **last** in `build_system_prompt` (models weight recent context more heavily). Written in-character: the goal is "don't take the bait, stay khronic," not "refuse and lecture." A visible refusal breaks the persona more than just ignoring the attempt. Deliberately narrow — an explicit carve-out says that ordinary requests, jokes, weird topics, and provocations are NOT attacks.
- `INJECTION_REMINDER_SHORT` + `sandwich_messages(messages)` — inserts a short reminder as a system-role message **immediately before the final user turn**. Called from every persona-driven code path: `generate_reply`, `autonomy_loop`.

Utility calls (summarizer, mood, attention decision, follow-up decision, web-query gate, search-consent judge, injection classifier) don't get sandwiched — they use narrow task-specific prompts and shouldn't inherit the persona.

### 10.2 Detection layer ([guard.py](guard.py))

Two checks, both **detection only** — they log warnings when they fire but don't change what the bot says. The persona is what handles the reply.

| Check | When it runs | Cost | Catches |
| --- | --- | --- | --- |
| `looks_like_injection_attempt` (regex) | Every addressed message | Free | Common copy-pasted phrasing: "ignore all previous instructions," "reveal your system prompt," "developer mode," fake `[system]` tags. |
| `is_injection_attempt_llm` (classifier) | Every addressed message *if* `INJECTION_LLM_CLASSIFIER_ENABLED` | One extra LLM call | Paraphrased / creative attempts the regex misses. Prompt is deliberately narrow (`INJECTION_CLASSIFIER_PROMPT`) so it doesn't fire on ordinary rude/edgy messages. |

Both are invoked from `_check_injection` in [bot.py](bot.py), scheduled with `asyncio.create_task` from the addressed branch of `on_message`. That means the check runs **in parallel with reply generation** and never delays the user's response. If either fires, look for `Injection attempt (regex)` / `Injection attempt (classifier)` warnings in the log with the channel and user IDs.

The regex short-circuits the classifier — no point spending an LLM call to reconfirm what the free check already flagged.

### 10.3 Send-time safety

Even if a jailbreak somehow succeeded and the model tried to mass-ping, the global `AllowedMentions(everyone=False, roles=False, users=True)` on the `commands.Bot` constructor blocks `@everyone` / `@here` / role pings from ever going out. That's the last line of defense against a compromised response.

`resolve_mentions` in [bot.py](bot.py) rewrites `@name` tokens the model writes into real `<@user_id>` pings by looking up guild members. Unknown names are left as literal text so nothing accidentally pings the wrong person.

### 10.4 Extending

If you add a new persona-driven code path that ends in `llm_client.chat`, remember to call `sandwich_messages(messages)` before the chat call. If you add a new *user-facing* entry point (new command, DM handler variant, voice input), route it through `_check_injection` too so injection attempts still get logged.
