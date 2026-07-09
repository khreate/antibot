import asyncio
import random
import re
import time
import logging
from datetime import datetime, timezone

import discord
from discord.ext import commands, tasks

import config
import llm_client
import web_search
import guard
import vision
from memory import Memory
from persona import (
    build_system_prompt,
    sandwich_messages,
    AUTONOMY_PROMPT_SUFFIX,
    SUMMARIZATION_SYSTEM_PROMPT,
    MOOD_SYSTEM_PROMPT,
    ATTENTION_DECISION_PROMPT,
    FOLLOWUP_DECISION_PROMPT,
    WEB_QUERY_PROMPT,
    SEARCH_CONSENT_PROMPT,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sentient_bot")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(
    command_prefix="!khronic ",
    intents=intents,
    # Safety default for every send: allow user pings but block @everyone / @here / role pings
    # so the model can't ever mass-notify a server, no matter what it types.
    allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=True),
)
mem = Memory()

# in-memory cooldown trackers (ephemeral -- reset on restart, that's fine for this purpose)
_last_thread_ts: dict[int, float] = {}
# tracks the last time the autonomy loop *itself* posted in a channel, so the active-channel
# tick can throttle self-initiated chime-ins independently of normal-reply cadence.
_last_autonomous_ts: dict[int, float] = {}
# attention_loop watermark: highest Discord message id already "seen" per channel, so the
# LLM only ever considers new messages for reactions.
_last_attention_msg_id: dict[int, int] = {}

# per-channel locks so two people addressing the bot at once are handled one at a time,
# instead of racing on memory reads/writes and interleaving replies
_channel_locks: dict[int, asyncio.Lock] = {}

# coalescing state: buffered addressed messages awaiting a single batched reply, plus the
# in-flight debounce worker per channel (see _coalesced_reply). Each buffered entry is a
# (message, memory_row_id) pair so, if the reply spins up a thread, those exact stored rows
# can be moved into the thread's context instead of dangling in the parent channel.
_pending_msgs: dict[int, list[tuple[discord.Message, int]]] = {}
_debounce_tasks: dict[int, asyncio.Task] = {}


def _channel_lock(channel_id: int) -> asyncio.Lock:
    lock = _channel_locks.get(channel_id)
    if lock is None:
        lock = asyncio.Lock()
        _channel_locks[channel_id] = lock
    return lock


async def _check_injection(text: str, message: discord.Message) -> None:
    """Log-only injection safeguard that layers on top of persona.INJECTION_GUARD.

    Fire-and-forget from the addressed path so it never blocks the reply. The regex is free
    and runs first; if `INJECTION_LLM_CLASSIFIER_ENABLED` is set the LLM classifier also runs
    (in parallel with generation, since it's just detection -- the persona is the actual
    defense).
    """
    if not config.INJECTION_DETECTION_ENABLED or not text:
        return
    where = f"channel {getattr(message.channel, 'id', '?')} user {message.author.id}"
    if guard.looks_like_injection_attempt(text):
        log.warning(f"Injection attempt (regex) from {where}: {text[:200]!r}")
        return  # already flagged; no need to spend an LLM call to reconfirm
    if config.INJECTION_LLM_CLASSIFIER_ENABLED:
        try:
            if await guard.is_injection_attempt_llm(text):
                log.warning(f"Injection attempt (classifier) from {where}: {text[:200]!r}")
        except Exception as e:
            log.warning(f"Injection classifier failed: {e}")


def should_respond(message: discord.Message) -> bool:
    if message.author.bot:
        return False
    if bot.user in message.mentions:
        return True
    if isinstance(message.channel, discord.DMChannel):
        return True
    if message.reference and message.reference.resolved:
        ref = message.reference.resolved
        if getattr(ref, "author", None) == bot.user:
            return True
    if mentions_by_name(message.content):
        return True
    return False


def mentions_by_name(content: str) -> bool:
    """True if the message says one of the bot's names as a whole word (case-insensitive).

    Lets people address the bot by name without needing a proper @mention -- e.g. typing
    'khronic what do you think' in chat. Whole-word match so it doesn't false-fire on things
    like 'chronically'.
    """
    if not content or not config.NAME_TRIGGERS:
        return False
    for name in config.NAME_TRIGGERS:
        if re.search(rf"\b{re.escape(name)}\b", content, re.IGNORECASE):
            return True
    return False


async def is_conversation_followup(message: discord.Message) -> bool:
    """True when the message looks like a continuation of a back-and-forth the bot is in.

    Catches the natural case where the bot just spoke and the same person keeps talking
    without re-@mentioning it (e.g. bot asks "what's the topic?" and they answer "hell").

    A cheap recency gate runs first so we never spend an LLM call on channels the bot isn't
    actively part of: the channel must have been active within the window AND the bot must
    have spoken in the recent turns. Once that passes, the LLM makes the real judgment (unless
    FOLLOWUP_USE_LLM is off, in which case we fall back to "did the bot send the last message").
    """
    if config.FOLLOWUP_WINDOW_SECONDS <= 0:
        return False
    if isinstance(message.channel, discord.DMChannel):
        return False  # DMs already always respond
    # if they're pinging other people (and not the bot), they're talking to someone else
    if message.mentions and bot.user not in message.mentions:
        return False

    state = await mem.get_channel_state(message.channel.id)
    last_ts = state["last_message_ts"]
    if last_ts is None or (time.time() - last_ts) > config.FOLLOWUP_WINDOW_SECONDS:
        return False  # channel has gone quiet -- not an active exchange

    recent = await mem.get_recent_messages(message.channel.id, config.SHORT_TERM_TURNS)
    if not recent or not any(role == "assistant" for role, _c, _u in recent):
        return False  # the bot isn't part of this recent conversation at all

    if not config.FOLLOWUP_USE_LLM:
        return recent[-1][0] == "assistant"  # heuristic: bot sent the most recent message

    return await llm_followup_decision(recent, message)


async def llm_followup_decision(recent, message: discord.Message) -> bool:
    """Ask the model whether an unaddressed message is actually directed at the bot."""
    name = bot.user.display_name
    lines = []
    for role, content, uname in recent:
        speaker = name if role == "assistant" else (uname or "someone")
        lines.append(f"{speaker}: {content}")
    transcript = "\n".join(lines)

    messages = [
        {"role": "system", "content": FOLLOWUP_DECISION_PROMPT.format(name=name)},
        {"role": "user", "content": (
            f"Recent conversation:\n{transcript}\n\n"
            f"New message:\n{message.author.display_name}: {strip_mention(message.content)}\n\n"
            "Is this new message directed at you? Answer YES or NO."
        )},
    ]
    try:
        result = await llm_client.chat(messages, temperature=0.0, max_tokens=3)
    except Exception as e:
        log.warning(f"Follow-up decision failed: {e}")
        return recent[-1][0] == "assistant"  # fall back to the heuristic
    return result.strip().upper().startswith("YES")


def strip_mention(content: str) -> str:
    return content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()


# Matches an action directive the model can emit to spin up a Discord thread, e.g.
#   [[thread: hell]]
_THREAD_RE = re.compile(r"\[\[\s*thread\s*:\s*(.+?)\s*\]\]", re.IGNORECASE)

# Matches natural-looking name mentions like `@khreate` the model can write inline in a reply.
# Resolved to real Discord <@id> pings via guild member lookup at send-time. The lookbehind
# skips tokens already inside a proper <@id> mention or part of another word/email.
_MENTION_RE = re.compile(r"(?<![<\w])@([A-Za-z0-9_\-.]{2,32})")


def extract_thread_directive(reply: str) -> tuple[str | None, str]:
    """Pull a [[thread: title]] directive out of a reply.

    Returns (thread_name_or_None, reply_without_the_directive). Discord caps thread names at
    100 chars. Only the direct-reply path acts on the name; everywhere else the directive is
    simply stripped so it never leaks into chat as literal text.
    """
    m = _THREAD_RE.search(reply)
    if not m:
        return None, reply
    name = m.group(1).strip()[:100]
    cleaned = _THREAD_RE.sub("", reply).strip()
    return (name or None), cleaned


def _can_create_thread(channel: discord.abc.Messageable) -> bool:
    """Guardrails so the bot can't fixate on threads.

    Blocks creation when the feature is off, when we're already inside a thread (Discord has
    no threads-in-threads, so attempts just raise and spam the log), or when the per-channel
    cooldown hasn't elapsed. This is what stops the "only makes and talks in threads" loop.
    """
    if not config.THREAD_CREATION_ENABLED:
        return False
    if isinstance(channel, discord.Thread):
        return False
    if isinstance(channel, discord.DMChannel):
        return False  # safety: don't open threads in DMs (Discord doesn't support them)
    if isinstance(channel, discord.TextChannel) and not channel.guild:
        return False  # safety: don't open threads in non-guild channels (e.g. DMs)
    if not _channel_allowed(channel):
        return False  # respect guild channel restrictions
    last = _last_thread_ts.get(getattr(channel, "id", 0), 0)
    return (time.time() - last) >= config.THREAD_CREATION_COOLDOWN_SECONDS


def _pack_words(text: str, limit: int) -> list[str]:
    """Greedily pack words into chunks of at most `limit` chars, keeping whole words intact
    (a single word longer than the limit -- e.g. a URL -- is kept whole rather than chopped)."""
    out: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current += " " + word
        else:
            out.append(current)
            current = word
    if current:
        out.append(current)
    return out


def _looks_like_list(text: str) -> bool:
    """Return True when text contains markdown-ish list items that should stay grouped."""
    return bool(re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text or ""))


def chunk_text(text: str, limit: int) -> list[str]:
    """Split text into messages of at most `limit` chars, breaking on sentence boundaries.

    Whole sentences are packed together up to the limit so each message ends on a complete
    thought instead of being cut off mid-sentence. Only a single sentence that's itself longer
    than the limit falls back to word-level packing. Existing newlines are preserved inside a
    chunk when the combined text still fits, so short lists can stay in one Discord message.
    A final pass enforces Discord's hard 2000-char ceiling.
    """
    chunks: list[str] = []
    current_chunk = ""
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        # split after . ! ? (incl. repeats like "?!") when followed by whitespace
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", block) if s]
        block_chunks: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > limit:
                if current:
                    block_chunks.append(current)
                    current = ""
                block_chunks.extend(_pack_words(sentence, limit))
            elif not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= limit:
                current += " " + sentence
            else:
                block_chunks.append(current)
                current = sentence
        if current:
            block_chunks.append(current)

        for block_chunk in block_chunks:
            if not current_chunk:
                current_chunk = block_chunk
            elif len(current_chunk) + 1 + len(block_chunk) <= limit:
                current_chunk += "\n" + block_chunk
            else:
                chunks.append(current_chunk)
                current_chunk = block_chunk

    if current_chunk:
        chunks.append(current_chunk)

    safe: list[str] = []
    for c in chunks:
        if len(c) <= 2000:
            safe.append(c)
        else:
            safe.extend(c[i:i + 2000] for i in range(0, len(c), 2000))
    return safe


def resolve_mentions(text: str, guild: discord.Guild | None) -> str:
    """Rewrite `@name` tokens in `text` as real Discord `<@user_id>` mentions.

    Matches on display name first, then username, case-insensitive. Unknown names are left
    as literal text so the message still reads sensibly (rather than silently vanishing).
    Only runs when we have a guild -- DMs have nobody to ping. Already-formatted `<@id>`
    mentions are untouched by the regex.
    """
    if not guild or not text or "@" not in text:
        return text

    # Build a lowercase lookup once per call: name -> user_id (display name wins over username)
    lookup: dict[str, int] = {}
    for member in guild.members:
        lookup.setdefault(member.name.lower(), member.id)
        lookup[member.display_name.lower()] = member.id

    def _sub(m: re.Match) -> str:
        uid = lookup.get(m.group(1).lower())
        return f"<@{uid}>" if uid else m.group(0)

    return _MENTION_RE.sub(_sub, text)


def _channel_allowed(channel: discord.abc.Messageable) -> bool:
    """Return False if the guild this channel belongs to has a channel restriction and this
    channel (or its thread parent) is not in the allowed list."""
    guild_id = getattr(getattr(channel, "guild", None), "id", None)
    if guild_id is None:
        return True  # DMs are always allowed
    allowed = config.GUILD_CHANNEL_RESTRICTIONS.get(guild_id)
    if not allowed:
        return True  # no restriction for this guild
    channel_id = getattr(channel, "id", None)
    if channel_id in allowed:
        return True
    # also allow threads whose parent channel is in the allowed list
    parent_id = getattr(channel, "parent_id", None)
    if parent_id is not None and parent_id in allowed:
        return True
    return False


async def send_message(channel: discord.abc.Messageable, text: str,
                        guild: discord.Guild | None = None) -> bool:
    """Send a reply safely: skip empties, cap each message at MAX_MESSAGE_CHARS, log failures.

    Returns True if at least one chunk was sent. The raw send() call raising here (empty
    content, over-length message, or missing permissions) is the usual cause of the bot
    generating a reply but never posting it.

    When `guild` is passed, any `@name` tokens the model wrote are resolved into real Discord
    `<@id>` mentions so it can actually ping people (mass pings stay blocked via the bot's
    global allowed_mentions default).
    """
    if not _channel_allowed(channel):
        log.info(f"Suppressing send: channel {getattr(channel, 'id', '?')} not in guild restriction list")
        return False
    # strip any stray thread directive so it never leaks as literal text on non-direct paths
    _name, text = extract_thread_directive((text or "").strip())
    if not text:
        log.warning("Skipping send: model returned an empty message")
        return False

    text = resolve_mentions(text, guild)

    chunk_limit = 2000 if _looks_like_list(text) and len(text) <= 2000 else config.MAX_MESSAGE_CHARS
    chunks = chunk_text(text, chunk_limit)
    if len(chunks) > config.MAX_REPLY_MESSAGES:
        log.info(f"Reply exceeded {config.MAX_REPLY_MESSAGES} messages; truncating the tail")
        chunks = chunks[:config.MAX_REPLY_MESSAGES]

    try:
        for i, chunk in enumerate(chunks):
            await channel.send(chunk)
            if i < len(chunks) - 1 and config.MESSAGE_SEND_DELAY_SECONDS > 0:
                await asyncio.sleep(config.MESSAGE_SEND_DELAY_SECONDS)
        return True
    except discord.Forbidden:
        log.warning(f"Missing permission to send in channel {getattr(channel, 'id', '?')}")
    except discord.HTTPException as e:
        log.warning(f"Send failed: {e}")
    return False


# --- Consent-gated web lookup ------------------------------------------------------------
# The bot never searches silently. When a message looks like it needs current info, it proposes a
# lookup ("want me to look that up?"), stashes the query, and only actually hits the network on
# the next message if the user approves. Pending proposals are per-channel and in-memory only.
_pending_web_search: dict[int, str] = {}

# Cheap keyword gate for "does this smell like a current-event / post-cutoff question?" -- runs
# before any LLM/web call so ordinary chatter never triggers a lookup proposal.
_CURRENT_EVENT_RE = re.compile(
    r"\b(today|todays|tonight|tomorrow|yesterday|now|currently|current|latest|"
    r"recent|recently|this (?:week|month|year|morning|evening)|"
    r"news|headline|headlines|score|scores|weather|forecast|price|prices|stock|"
    r"who won|what happened|release[ds]?|released|launch(?:ed|ing)?|update[ds]?|"
    r"in theaters|in theatres|showtimes|20(?:2[4-9]|[3-9]\d)|version)\b",
    re.IGNORECASE,
)

# Affirmations that count as "yes, go search" when a lookup proposal is pending.
_AFFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|ya|yea|sure|ok|okay|k|please|pls|plz|"
    r"go ahead|go for it|do it|sounds good|why not|search|look it up|"
    r"look that up|find out)\b",
    re.IGNORECASE,
)

# A few low-key, in-voice ways to ask permission before searching.
_SEARCH_ASK_LINES = [
    "want me to look that up?",
    "i can look it up if you want",
    "should i actually go check?",
    "i'd have to look that up -- want me to?",
]


def _looks_current(text: str) -> bool:
    return bool(text and _CURRENT_EVENT_RE.search(text))


async def _is_affirmative(message: str) -> bool:
    """Judge whether the user's reply grants permission to search.

    Asks the LLM for a YES/NO verdict; falls back to the keyword regex if the call fails.
    """
    try:
        verdict = await llm_client.chat(
            [
                {"role": "system", "content": SEARCH_CONSENT_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=3,
        )
        return verdict.strip().upper().startswith("YES")
    except Exception as e:
        log.warning(f"Consent judge failed, using keyword fallback: {e}")
        return bool(_AFFIRM_RE.search(message))


async def _build_search_query(message: str) -> str | None:
    """Turn a user message into a concise search query, or None if a lookup isn't warranted.

    Uses the LLM query gate when enabled (it also vetoes pointless/private lookups via NO_SEARCH);
    otherwise falls back to the raw message.
    """
    if not config.WEB_SEARCH_USE_LLM_GATE:
        return message
    try:
        decision = await llm_client.chat(
            [
                {"role": "system", "content": WEB_QUERY_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=30,
        )
        decision = decision.strip()
        if not decision or decision.upper().startswith("NO_SEARCH"):
            return None
        return decision
    except Exception as e:
        log.warning(f"Web-search gate failed, using raw query: {e}")
        return message


async def resolve_web_lookup(channel_id: int, latest_user_msg: str | None) -> tuple[str, str | None]:
    """Consent-gated web-lookup state machine. Returns (action, value):

      ("none", None)      -- no web lookup involved; reply normally
      ("ask", None)       -- propose a lookup; caller should ask the user for permission
      ("context", block)  -- user approved; `block` is the results to fold into the prompt
    """
    if not config.WEB_SEARCH_ENABLED or not latest_user_msg:
        return ("none", None)

    pending = _pending_web_search.get(channel_id)
    if pending is not None:
        # we previously proposed a lookup and are waiting on a yes/no
        _pending_web_search.pop(channel_id, None)
        if await _is_affirmative(latest_user_msg):
            results = await web_search.search(pending)
            if not results:
                return ("none", None)
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            log.info(f"Web lookup (approved) for {pending!r} -> {len(results)} results")
            block = (f'Live web results (DuckDuckGo, retrieved {stamp}) for "{pending}":\n'
                     + web_search.format_results(results))
            return ("context", block)
        # not an approval -- fall through to check if THIS message needs its own lookup

    if not _looks_current(latest_user_msg):
        return ("none", None)
    query = await _build_search_query(latest_user_msg)
    if not query:
        return ("none", None)
    _pending_web_search[channel_id] = query
    return ("ask", None)


async def generate_reply(channel_id: int, profile_user_id: int | None) -> str:
    """Generate a reply from the channel's stored recent context.

    Callers store the incoming message(s) to memory first, so the latest turns are already in
    `recent` -- we don't append anything extra here. `profile_user_id` selects whose long-term
    profile to fold in (typically the most recent speaker); pass None to skip it.
    """
    state = await mem.get_channel_state(channel_id)
    profile = await mem.get_profile(profile_user_id) if profile_user_id else None
    recent = await mem.get_recent_messages(channel_id, config.SHORT_TERM_TURNS)

    system_prompt = build_system_prompt(mood=state["mood"], profile_summary=profile)

    # consent-gated web lookup: propose a search and wait for approval before ever hitting the
    # network. "ask" -> return the permission question as the whole reply; "context" -> the user
    # approved, so fold the fetched results into the prompt.
    latest_user_msg = next((content for role, content, _u in reversed(recent) if role == "user"), None)
    action, block = await resolve_web_lookup(channel_id, latest_user_msg)
    if action == "ask":
        return random.choice(_SEARCH_ASK_LINES)
    if action == "context" and block:
        system_prompt += (
            "\n\n" + block +
            "\n\nThe web results above are fresher than your built-in knowledge. Use them for any "
            "current-event or post-2023 facts, work the info into your own voice, and don't dump "
            "raw links unless someone asks. If the results are thin or conflicting, say so plainly "
            "instead of guessing."
        )

    messages = [{"role": "system", "content": system_prompt}]
    for role, content, uname in recent:
        # Prefix user messages with who said them so the model can track multiple speakers
        if role == "user" and uname:
            messages.append({"role": "user", "content": f"{uname}: {content}"})
        else:
            messages.append({"role": role, "content": content})

    # Sandwich in a short injection reminder right before the last user turn -- cheap
    # defense-in-depth on top of INJECTION_GUARD at the top of the system prompt.
    messages = sandwich_messages(messages)

    reply = await llm_client.chat(messages)
    return reply


async def maybe_summarize(channel_id: int, user_id: int, username: str):
    state = await mem.get_channel_state(channel_id)
    latest_id = await mem.latest_message_id(channel_id)
    if latest_id - state["last_summarized_id"] < config.SUMMARIZE_EVERY:
        return

    new_msgs = await mem.messages_since(channel_id, state["last_summarized_id"])
    if not new_msgs:
        return

    existing = await mem.get_profile(user_id) or "(no prior profile)"
    transcript = "\n".join(f"{uname or role}: {content}" for _id, role, content, uname in new_msgs)

    messages = [
        {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Existing profile:\n{existing}\n\nNew messages:\n{transcript}"},
    ]
    try:
        new_summary = await llm_client.chat(messages, temperature=0.3, max_tokens=250)
        await mem.upsert_profile(user_id, new_summary)
        await mem.set_last_summarized_id(channel_id, latest_id)
        log.info(f"Updated profile for user {user_id}")
    except Exception as e:
        log.warning(f"Summarization failed: {e}")

    # lightweight mood update off the same transcript
    try:
        mood_messages = [
            {"role": "system", "content": MOOD_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        mood = await llm_client.chat(mood_messages, temperature=0.5, max_tokens=20)
        await mem.set_mood(channel_id, mood)
    except Exception as e:
        log.warning(f"Mood update failed: {e}")


@bot.event
async def setup_hook():
    # setup_hook runs after login but BEFORE the gateway starts dispatching events, so any
    # message that arrives is guaranteed to see an initialized DB. Doing this in on_ready
    # instead races: on_ready runs concurrently with early message events, and get_channel_state
    # would blow up with 'NoneType has no attribute execute' if a message beat the init await.
    await mem.init()


@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user}")
    if config.AUTONOMY_ENABLED and not autonomy_loop.is_running():
        autonomy_loop.start()
    if config.ATTENTION_ENABLED and not attention_loop.is_running():
        attention_loop.start()


async def _coalesced_reply(channel_id: int):
    """Debounced worker: after a short window, answer everything buffered for a channel at once.

    Incoming addressed messages are stored to memory immediately (preserving order) and pushed
    onto `_pending_msgs`; the first one spawns this task. We wait COALESCE_WINDOW_SECONDS so a
    burst settles, then generate a single reply from the accumulated context. The loop re-checks
    the buffer so messages that arrive mid-generation get folded into a follow-up batch rather
    than spawning a competing reply. The buffer-empty check and marker removal happen with no
    await between them, so an enqueue can't slip through and get stranded.
    """
    if config.COALESCE_WINDOW_SECONDS > 0:
        await asyncio.sleep(config.COALESCE_WINDOW_SECONDS)

    async with _channel_lock(channel_id):
        while True:
            pending = _pending_msgs.get(channel_id) or []
            if not pending:
                _debounce_tasks.pop(channel_id, None)  # atomic with the check above (no await)
                return
            _pending_msgs[channel_id] = []

            anchor, _ = pending[-1]  # most recent message: used for profile, thread anchor, guild
            row_ids = [rid for _, rid in pending]
            channel = anchor.channel
            async with channel.typing():
                try:
                    reply = await generate_reply(channel_id, anchor.author.id)
                except Exception:
                    log.exception("Generation failed")
                    await channel.send("(brain fog -- something broke on my end, try again in a sec)")
                    continue

                # Model can decline to speak on an addressed turn (persona allows this rarely).
                # Skip the send + memory-add for the assistant turn, but still fall through to
                # maybe_summarize below since the user's message is worth folding into memory.
                if reply.strip().upper() == "[SKIP]":
                    log.info(f"Model chose to skip an addressed reply in channel {channel_id}")
                else:
                    # the model can ask to open a thread by leading with [[thread: title]]
                    thread_name, reply = extract_thread_directive(reply)
                    target: discord.abc.Messageable = channel
                    if thread_name and _can_create_thread(channel):
                        try:
                            target = await anchor.create_thread(name=thread_name)
                            _last_thread_ts[channel_id] = time.time()
                            log.info(f"Created thread '{thread_name}' from message in channel {channel_id}")
                            # move the triggering messages into the thread's context so the parent
                            # channel isn't left with a dangling unanswered request (which the model
                            # would otherwise re-answer next time it's addressed in the parent)
                            await mem.reassign_messages(row_ids, target.id)
                        except discord.HTTPException as e:
                            log.warning(f"Couldn't create thread '{thread_name}': {e}")
                            target = channel

                    await mem.add_message(getattr(target, "id", channel_id), bot.user.id,
                                           str(bot.user), "assistant", reply)
                    await send_message(target, reply, guild=anchor.guild)

            asyncio.create_task(
                maybe_summarize(channel_id, anchor.author.id, anchor.author.display_name)
            )


@bot.event
async def on_message(message: discord.Message):
    await bot.process_commands(message)
    if message.author.bot:
        return

    # ignore messages in channels the bot isn't allowed to respond in for this guild
    if not _channel_allowed(message.channel):
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    text = strip_mention(message.content)
    addressed = should_respond(message) or await is_conversation_followup(message)

    # Vision is a cheap no-op for the vast majority of messages (no attachments -> no network
    # or model call at all, see vision.describe_attachments). When a message does carry images,
    # Qwen's description gets folded into the plain text right here, before anything else
    # touches it -- so memory storage, generate_reply, follow-up detection on future turns,
    # injection scanning, and summarization all just see it as more text and need no changes.
    # VISION_ANALYZE_PASSIVE controls whether that also happens for images the bot wasn't
    # addressed on (so it can still reference them later via attention/autonomy), vs. only
    # ever spending a Qwen call on images in messages that directly address the bot.
    if message.attachments and (addressed or config.VISION_ANALYZE_PASSIVE):
        image_desc = await vision.describe_attachments(message.attachments)
        if image_desc:
            text = f"{text}\n{image_desc}" if text else image_desc

    if addressed:
        # directly addressed (pinged, replied to, DM) or a natural follow-up to something
        # the bot just said -- respond in full
        if not text:
            return
        # store immediately so context stays in real order, then buffer for a coalesced reply:
        # a burst of messages (many people, or quick successive lines) is answered together in
        # one reply instead of one-reply-per-message. The first message spawns the debounce
        # worker; the append + task check below run with no await between them, so it's race-free.
        row_id = await mem.add_message(message.channel.id, message.author.id,
                                        message.author.display_name, "user", text)
        # Defense-in-depth safeguard: log if the incoming message looks like a prompt-injection
        # attempt. Runs alongside reply generation so it never delays the response.
        asyncio.create_task(_check_injection(text, message))
        _pending_msgs.setdefault(message.channel.id, []).append((message, row_id))
        if message.channel.id not in _debounce_tasks:
            _debounce_tasks[message.channel.id] = asyncio.create_task(
                _coalesced_reply(message.channel.id)
            )
        return

    # not directly addressed: still track it for context. Whether to react or chime in is
    # decided by attention_loop / autonomy_loop on their own ticks, NOT by message arrival --
    # so the bot's initiative isn't gated on someone sending something new.
    if is_dm or not text:
        return

    await mem.add_message(message.channel.id, message.author.id, message.author.display_name, "user", text)


@tasks.loop(minutes=config.AUTONOMY_CHECK_INTERVAL_MINUTES)
async def autonomy_loop():
    """Self-initiated behavior: on every tick, consider each channel the bot knows about and
    decide whether to speak up on its own -- independent of any user message triggering it.

    Two modes, per channel:
      - IDLE: the channel has been quiet for a long time (AUTONOMY_MIN_IDLE_MINUTES). Roll
        AUTONOMY_CHANCE. This is the "haven't heard from you all in a while" nudge.
      - ACTIVE: the channel has fresh activity (within AUTONOMY_ACTIVE_MAX_IDLE_MINUTES) but
        the bot hasn't chimed in autonomously here recently. Roll AUTONOMY_ACTIVE_CHANCE.
        This is the "just felt like saying something" chime-in that isn't wired to any
        specific incoming message.

    Either way the LLM gets the [SKIP] escape hatch (via AUTONOMY_PROMPT_SUFFIX and the
    formatting instruction added in build_system_prompt), so the model still has final veto.
    """
    channels = await mem.all_known_channels()
    now = time.time()
    for channel_id in channels:
        if config.AUTONOMY_CHANNEL_WHITELIST and channel_id not in config.AUTONOMY_CHANNEL_WHITELIST:
            continue

        state = await mem.get_channel_state(channel_id)
        last_ts = state["last_message_ts"]
        if last_ts is None:
            continue
        idle_minutes = (now - last_ts) / 60

        channel = bot.get_channel(channel_id)
        if channel is None:
            continue

        # -- decide which mode (if any) this channel is eligible for --
        mode: str | None = None
        if idle_minutes >= config.AUTONOMY_MIN_IDLE_MINUTES:
            if random.random() <= config.AUTONOMY_CHANCE:
                mode = "idle"
        elif (
            config.AUTONOMY_ACTIVE_ENABLED
            and not isinstance(channel, discord.DMChannel)  # DMs already always get replies
            and idle_minutes <= config.AUTONOMY_ACTIVE_MAX_IDLE_MINUTES
        ):
            last_self = _last_autonomous_ts.get(channel_id, 0)
            gap_minutes = (now - last_self) / 60
            if gap_minutes >= config.AUTONOMY_ACTIVE_MIN_SINCE_LAST_BOT_MINUTES:
                if random.random() <= config.AUTONOMY_ACTIVE_CHANCE:
                    mode = "active"
        if mode is None:
            continue

        recent = await mem.get_recent_messages(channel_id, config.SHORT_TERM_TURNS)
        if not recent:
            continue

        system_prompt = build_system_prompt(mood=state["mood"], profile_summary=None) + AUTONOMY_PROMPT_SUFFIX
        messages = [{"role": "system", "content": system_prompt}]
        for role, content, uname in recent:
            if role == "user" and uname:
                messages.append({"role": "user", "content": f"{uname}: {content}"})
            else:
                messages.append({"role": role, "content": content})
        messages = sandwich_messages(messages)

        try:
            reply = await llm_client.chat(messages)
        except Exception as e:
            log.warning(f"Autonomy generation failed: {e}")
            continue

        if reply.strip() == "[SKIP]" or not reply.strip():
            log.info(f"Autonomy skipped ({mode}) in channel {channel_id}")
            continue

        await mem.add_message(channel_id, bot.user.id, str(bot.user), "assistant", reply)
        await send_message(channel, reply, guild=getattr(channel, "guild", None))
        _last_autonomous_ts[channel_id] = time.time()
        log.info(f"Sent autonomous message ({mode}) in channel {channel_id}")


@autonomy_loop.before_loop
async def before_autonomy():
    await bot.wait_until_ready()


_ATTENTION_REACT_RE = re.compile(r"REACT\s+(\d+)\s+(\S+)", re.IGNORECASE)


@tasks.loop(seconds=config.ATTENTION_CHECK_INTERVAL_SECONDS)
async def attention_loop():
    """Tick-driven reaction pass: periodically look at each active channel and let the model
    decide whether to react to any recent message with an emoji. Reactions are the bot's own
    initiative on its own schedule, not a response to any specific message arriving.

    Per tick, per channel:
      - skip if channel isn't active enough (older than ATTENTION_MAX_IDLE_MINUTES)
      - skip with probability (1 - ATTENTION_CHANCE) so most ticks are silent even when eligible
      - fetch last ATTENTION_LOOKBACK messages via Discord, filter to ones newer than the last
        seen watermark and not authored by the bot itself
      - ask the LLM: REACT <n> <emoji> or NONE
      - if REACT, add that emoji to that message; either way advance the watermark so we don't
        keep re-considering the same messages next tick
    """
    channels = await mem.all_known_channels()
    now = time.time()
    for channel_id in channels:
        if config.AUTONOMY_CHANNEL_WHITELIST and channel_id not in config.AUTONOMY_CHANNEL_WHITELIST:
            continue

        state = await mem.get_channel_state(channel_id)
        last_ts = state["last_message_ts"]
        if last_ts is None:
            continue
        idle_minutes = (now - last_ts) / 60
        if idle_minutes > config.ATTENTION_MAX_IDLE_MINUTES:
            continue

        channel = bot.get_channel(channel_id)
        if channel is None or isinstance(channel, discord.DMChannel):
            continue

        if random.random() > config.ATTENTION_CHANCE:
            continue

        # Pull recent messages from Discord itself (not memory) so we can react to the actual
        # Message objects. history() returns newest-first; we reverse so the numbered list the
        # LLM sees runs oldest -> newest, matching normal reading order.
        try:
            history = [m async for m in channel.history(limit=config.ATTENTION_LOOKBACK)]
        except (discord.Forbidden, discord.HTTPException) as e:
            log.warning(f"Attention: couldn't fetch history in channel {channel_id}: {e}")
            continue
        history.reverse()

        last_seen = _last_attention_msg_id.get(channel_id, 0)
        candidates = [
            m for m in history
            if m.id > last_seen and not m.author.bot and (m.content or "").strip()
        ]
        if not candidates:
            continue
        # Advance watermark up-front so we don't reconsider these messages next tick even if
        # the LLM call fails or returns NONE. Reactions are a one-shot judgement.
        _last_attention_msg_id[channel_id] = candidates[-1].id

        numbered = "\n".join(
            f"[{i + 1}] {m.author.display_name}: {m.content}"
            for i, m in enumerate(candidates)
        )
        prompt_messages = [
            {"role": "system", "content": f"{ATTENTION_DECISION_PROMPT}\n\nYour current mood: {state['mood']}"},
            {"role": "user", "content": f"Recent messages:\n{numbered}"},
        ]
        try:
            result = await llm_client.chat(prompt_messages, temperature=0.6, max_tokens=20)
        except Exception as e:
            log.warning(f"Attention decision failed in channel {channel_id}: {e}")
            continue

        result = (result or "").strip()
        if not result or result.upper().startswith("NONE"):
            continue
        m = _ATTENTION_REACT_RE.search(result)
        if not m:
            log.info(f"Attention: unparseable response in channel {channel_id}: {result!r}")
            continue
        idx = int(m.group(1)) - 1
        emoji = m.group(2)
        if not (0 <= idx < len(candidates)):
            log.info(f"Attention: out-of-range index {idx + 1} for {len(candidates)} candidates")
            continue
        target = candidates[idx]
        try:
            await target.add_reaction(emoji)
            log.info(f"Attention: reacted {emoji!r} to msg {target.id} in channel {channel_id}")
        except discord.HTTPException as e:
            log.warning(f"Attention: couldn't add reaction {emoji!r}: {e}")


@attention_loop.before_loop
async def before_attention():
    await bot.wait_until_ready()


if __name__ == "__main__":
    if not config.DISCORD_TOKEN:
        raise SystemExit("Set DISCORD_TOKEN in your .env file")
    bot.run(config.DISCORD_TOKEN)
