"""Template config -- copy to `config.py` and edit for your deployment.

`config.py` is git-ignored so your personal channel restrictions, tuning knobs,
and any locally-hardcoded IDs never leak into version control. Everything here
is safe defaults you can commit; edit the copy freely.
"""

import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Ollama settings. Pull the model first with:
#   ollama pull ikiru/Dolphin-Mistral-24B-Venice-Edition
# On a 24GB card you have headroom for a better quant than the default 13GB (Q4-ish) pull.
# If you want higher quality, grab a Q5_K_M / Q6_K GGUF from bartowski's repo instead and
# `ollama create` a Modelfile pointing at it (see README).
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "ikiru/Dolphin-Mistral-24B-Venice-Edition")

DB_PATH = os.getenv("DB_PATH", "sentient_bot.db")

# --- Memory ---
SHORT_TERM_TURNS = 20          # verbatim recent turns kept in context per channel
SUMMARIZE_EVERY = 30           # new messages before folding old ones into long-term profile

# --- Conversation follow-ups ---
# Names the bot answers to when spoken aloud in chat (case-insensitive, whole-word match).
# Include the display name plus any nicknames people commonly use for it. Set to [] to
# require an actual @mention / reply / DM to trigger a response.
NAME_TRIGGERS: list[str] = ["khronic"]

# If the bot sent the most recent message in a channel and a user replies within this window
# (without @mentioning or using Discord's reply feature), treat it as a continuation of the
# conversation and respond in full. Keeps natural back-and-forth flowing without needing a
# ping every single turn. Set to 0 to disable and require an explicit mention/reply.
FOLLOWUP_WINDOW_SECONDS = 50

# When True, once the cheap recency gate passes (bot spoke recently in an active channel), the
# LLM makes the final call on whether an unaddressed message is actually directed at the bot --
# smarter than the pure "bot spoke last" heuristic, at the cost of one small extra LLM call.
# When False, falls back to the heuristic: only treat it as a follow-up if the bot sent the
# single most recent message.
FOLLOWUP_USE_LLM = True

# --- Autonomy: idle-triggered (speaks up after a quiet channel) ---
AUTONOMY_ENABLED = True
AUTONOMY_CHECK_INTERVAL_MINUTES = 5      # tick cadence; lower = feels more "alive" but polls more
AUTONOMY_MIN_IDLE_MINUTES = 120
AUTONOMY_CHANCE = 0.35          # probability of speaking up once idle threshold is met
AUTONOMY_CHANNEL_WHITELIST: list[int] = []   # empty = any channel the bot has spoken in

# --- Autonomy: active-channel chime-ins (self-initiated, not tied to any specific message) ---
# Lets the bot decide to speak up in channels it's watching WHILE they're active, so it isn't
# purely message-triggered. Runs on the same tick as the idle loop. The cooldown below stops
# it from looping on its own output; without it the loop could keep chiming in on top of
# itself.
AUTONOMY_ACTIVE_ENABLED = True
AUTONOMY_ACTIVE_CHANCE = 0.13            # per-tick per-eligible-channel probability
AUTONOMY_ACTIVE_MIN_SINCE_LAST_BOT_MINUTES = 15   # skip if the bot self-chimed here recently
AUTONOMY_ACTIVE_MAX_IDLE_MINUTES = 30             # channel must have activity in this window

# --- Autonomy: attention loop (tick-driven reactions to recent chatter) ---
# Periodically samples recent messages the bot has NOT yet reacted to and lets the model
# choose whether to react to one. Decouples reactions from message arrival: the bot decides
# WHEN it feels like reacting, not just what. Ambient/unprompted full messages are handled by
# AUTONOMY_ACTIVE above, so this loop is reactions-only. Runs on its own faster tick so
# reactions still feel reasonably timely.
ATTENTION_ENABLED = True
ATTENTION_CHECK_INTERVAL_SECONDS = 90    # tick cadence; lower = more responsive but more polls
ATTENTION_LOOKBACK = 15                  # how many recent Discord messages to show the model
ATTENTION_MAX_IDLE_MINUTES = 5           # channel must have activity within this window to be considered
ATTENTION_CHANCE = 0.25                  # per-tick per-eligible-channel probability of even asking the LLM

# --- Generation ---
TEMPERATURE = 0.9
MAX_TOKENS = 400
# Ollama's `repeat_penalty` option: >1 discourages the model from repeating the same tokens.
# Default upstream is 1.1; nudge up (e.g. 1.15-1.25) if khronic starts loop-repeating phrases
# ("that's just how it is. that's just how it is."), or down toward 1.0 if replies feel
# stilted / avoid natural word repetition.
REPEAT_PENALTY = 1.15

# Optional few-shot voice examples appended to the system prompt (see persona.build_system_prompt).
# Edit the file freely -- it's re-read on every generation, so no restart needed to tune voice.
# Set to None or "" to disable.
EXAMPLES_PATH = "examples.txt"

# --- Web lookup (DuckDuckGo, free / no API key) ---
# Consent-gated: the bot never searches silently. When a message looks like it needs current
# info, it asks first ("want me to look that up?") and only actually searches on the next message
# if the user approves. A cheap keyword heuristic gates the proposal (so most messages never
# prompt one); WEB_SEARCH_USE_LLM_GATE then refines the query and vetoes pointless/private ones.
# Uses the `ddgs` package -- no key, no cost.
WEB_SEARCH_ENABLED = True
WEB_SEARCH_MAX_RESULTS = 5          # how many results to fold into context
WEB_SEARCH_REGION = "wt-wt"         # DuckDuckGo region; "wt-wt" = no region
WEB_SEARCH_TIMEOUT_SECONDS = 8.0    # hard cap so a slow search never stalls a reply
WEB_SEARCH_USE_LLM_GATE = True      # let the model refine the query / veto needless lookups

# --- Message shaping ---
# Soft cap for each Discord message. Replies are split on SENTENCE boundaries first, so a
# message always ends on a complete thought rather than mid-sentence; only a single sentence
# longer than this cap gets broken on word boundaries. Links / very long single words are kept
# intact even if they exceed the cap. MAX_REPLY_MESSAGES is a safety valve so a runaway reply
# can't flood a channel; SEND_DELAY paces the burst so it reads naturally.
MAX_MESSAGE_CHARS = 300
MAX_REPLY_MESSAGES = 6
MESSAGE_SEND_DELAY_SECONDS = 0.6

# When addressed, wait this long before generating so any messages that land in the same burst
# (e.g. several people talking at once, or one person sending three quick lines) are gathered
# and answered together in a single reply, instead of firing a separate reply per message.
# Higher = calmer/less spammy but slightly laggier; lower = snappier but chattier. 0 disables.
COALESCE_WINDOW_SECONDS = 2.5

# --- Threads ---
# The model can open a thread by leading a reply with [[thread: title]]. These guardrails stop
# it fixating on threads: it can't open one from inside an existing thread (Discord has no
# threads-in-threads anyway), and it can't create them faster than this per-channel cooldown.
# Set THREAD_CREATION_ENABLED = False to turn the capability off entirely.
THREAD_CREATION_ENABLED = True
THREAD_CREATION_COOLDOWN_SECONDS = 300

# --- Guild channel restrictions ---
# Map of guild_id -> list of channel IDs the bot is allowed to respond in / send to.
# Any guild not listed here has no restriction. Use this to confine the bot to specific
# channels in a server (e.g. a #bot-dump channel) without affecting other servers.
# Threads whose parent is an allowed channel are automatically permitted.
#
# Example:
#   GUILD_CHANNEL_RESTRICTIONS = {
#       123456789012345678: [234567890123456789],  # #bot-dump in "My Server"
#   }
GUILD_CHANNEL_RESTRICTIONS: dict[int, list[int]] = {}

# --- Prompt-injection safeguards ---
# Layered defense-in-depth on top of persona.INJECTION_GUARD and persona.sandwich_messages.
# The regex pass in guard.py runs on every addressed message and only *logs* when it fires --
# the persona is what actually handles the reply. The optional LLM classifier catches
# paraphrased attempts the regex misses; it runs in parallel with reply generation (via
# asyncio.create_task) so it never adds latency to the user's response.
INJECTION_DETECTION_ENABLED = True
INJECTION_LLM_CLASSIFIER_ENABLED = False  # one extra LLM call per addressed message; off by default

# --- Vision (image understanding via Qwen-VL, also through Ollama) ---
# Pull a vision-capable Qwen model first, e.g.:
#   ollama pull qwen2.5vl:7b
# A *separate* model from MODEL_NAME on purpose -- Dolphin-Mistral is text-only. Vision is only
# invoked when an incoming Discord message actually has image attachments (checked before any
# network/model call happens), so it costs nothing on ordinary text-only messages. The
# description it produces gets folded into the message text as plain context, then the normal
# khronic persona/model handles the actual reply -- Qwen never talks to the user directly.
VISION_ENABLED = True
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "qwen3-vl:8b")
VISION_MAX_TOKENS = 300
VISION_MAX_IMAGES_PER_MESSAGE = 3     # cap so someone dumping a pile of images doesn't stall the reply
VISION_TIMEOUT_SECONDS = 30.0        # per-message cap on download + model time combined
# Whether to run vision analysis on messages that AREN'T addressed to the bot (i.e. images
# posted in normal channel chatter it's just watching). Keep True so the bot can later reference
# or react to an image via the attention/autonomy loops or a follow-up; set False to only ever
# analyze images in messages that directly address the bot, if you want to keep it cheaper.
VISION_ANALYZE_PASSIVE = True
