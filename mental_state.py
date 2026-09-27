"""
mental_state.py

A lightweight "internal state" layer for khronic, meant to sit alongside
persona.py / memory.py and give the bot a continuously-running affective
substrate instead of pure stimulus -> response behavior.

Core idea
---------
Real organisms don't sit frozen waiting for input. Mood, energy, and
"want to engage" all drift on their own, get nudged by events, and settle
back toward a baseline over time. We model each dimension as an
Ornstein-Uhlenbeck process:

    x(t + dt) = x(t) + theta * (mu - x(t)) * dt + sigma * sqrt(dt) * N(0, 1)

  - mu     : baseline the value drifts back toward (personality trait)
  - theta  : how strongly/quickly it reverts to baseline (0 = never, 1 = snaps back)
  - sigma  : how much random wander/noise per tick (personality "volatility")

This gives you variables that wander organically, never fully settle,
and never spike wildly -- which is exactly the texture you want for
"feels alive" rather than "reactive script."

Nothing here talks to the LLM directly. It exposes:
  - tick()                    -> call periodically (e.g. every loop iteration)
  - on_message(...)           -> call when a message arrives in a channel
  - on_bot_spoke()            -> call after the bot sends a message
  - render_prompt_fragment()  -> natural-language state description for the persona prompt
  - speak_urge() / react_urge() -> 0..1 scores to gate the autonomy/attention loops
  - save() / load()           -> SQLite persistence
"""

from __future__ import annotations

import asyncio
import math
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Per-dimension OU process
# ---------------------------------------------------------------------------

@dataclass
class OUVariable:
    value: float
    mu: float       # baseline / resting value
    theta: float    # mean-reversion strength (per second)
    sigma: float    # noise scale (per sqrt-second)
    lo: float = 0.0
    hi: float = 1.0

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        drift = self.theta * (self.mu - self.value) * dt
        noise = self.sigma * math.sqrt(dt) * random.gauss(0.0, 1.0)
        self.value = min(self.hi, max(self.lo, self.value + drift + noise))

    def nudge(self, amount: float) -> None:
        """External event pushes the value directly (clamped to range)."""
        self.value = min(self.hi, max(self.lo, self.value + amount))


# ---------------------------------------------------------------------------
# The state machine itself
# ---------------------------------------------------------------------------

class MentalState:
    """
    One instance per persona (global), or one per guild if you want the bot
    to feel differently "in the mood" per-server. Global is simpler and more
    true to "one character living across servers" -- matches your existing
    per-user (not per-guild) memory model.
    """

    def __init__(self, db_path: str = "khronic_state.db", seed_traits: Optional[dict] = None, graph=None):
        self.db_path = db_path
        self.graph = graph  # optional AssociativeGraph instance, see graph_memory.py
        traits = seed_traits or {}

        # Baselines (mu) are where PERSONALITY lives. Tune these per-character.
        # theta/sigma control how "even-keeled" vs. "moody" the character feels.
        self.energy = OUVariable(
            value=traits.get("energy", 0.6),
            mu=traits.get("energy_mu", 0.55),
            theta=traits.get("energy_theta", 0.00006),   # slow drift
            sigma=traits.get("energy_sigma", 0.004),
        )
        self.valence = OUVariable(
            value=traits.get("valence", 0.6),
            mu=traits.get("valence_mu", 0.6),
            theta=traits.get("valence_theta", 0.00004),
            sigma=traits.get("valence_sigma", 0.003),
        )
        self.social_battery = OUVariable(
            value=traits.get("social_battery", 0.7),
            mu=traits.get("social_battery_mu", 0.65),
            theta=traits.get("social_battery_theta", 0.00005),
            sigma=traits.get("social_battery_sigma", 0.003),
        )
        self.curiosity = OUVariable(
            value=traits.get("curiosity", 0.5),
            mu=traits.get("curiosity_mu", 0.5),
            theta=traits.get("curiosity_theta", 0.00003),
            sigma=traits.get("curiosity_sigma", 0.005),
        )
        # Boredom is hierarchical: each guild is one independent scope and
        # each DM channel is its own scope. Scope boredom decides WHETHER to
        # speak; channel boredom only helps choose WHERE within that scope.
        # One active channel therefore settles its guild, but never another
        # guild or an unrelated DM.
        self.boredom_rate = traits.get("boredom_rate", 0.00002)  # per second of silence
        self._last_scope_activity: Dict[str, float] = {}
        self._last_channel_activity: Dict[int, float] = {}

        # Reference point for channels we've never seen a message in yet --
        # treated as "just started watching", not literal infinite silence.
        self._watch_start = time.time()

        # Defensive circuit breaker: suppress unprompted speaking for a
        # short window after (re)connecting, so a reconnect burst or any
        # future bug in the boredom/activity math can't cause a synchronized
        # blast of unprompted messages across every channel at once.
        self._boot_time = time.time()
        self.startup_grace_s = traits.get("startup_grace_s", 300)

        # Update cadence decoupling: tick() can be called as often as your
        # loop happens to spin, but the OU variables only actually step
        # once real time has accumulated past this interval. This keeps
        # sigma/theta calibrated against a stable timescale instead of
        # however fast the caller happens to loop, and avoids sampling the
        # fine-grained jitter of the underlying random walk.
        self.min_tick_interval_s = traits.get("min_tick_interval_s", 1.0)
        self._pending_dt = 0.0

        # Guards concurrent mutation (tick/on_message/etc.) against a
        # snapshot read from a different thread (see save_async). CPython's
        # GIL already makes a single attribute write atomic, so this isn't
        # about preventing memory corruption -- it's about preventing a
        # snapshot from mixing pre- and post-update values across fields.
        self._lock = threading.Lock()

        self._last_tick = time.time()

    # -- core loop -----------------------------------------------------

    def tick(self) -> None:
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now
        if dt <= 0:
            return

        with self._lock:
            self._pending_dt += dt
            if self._pending_dt < self.min_tick_interval_s:
                return
            step_dt = self._pending_dt
            self._pending_dt = 0.0

            self.energy.step(step_dt)
            self.valence.step(step_dt)
            self.social_battery.step(step_dt)
            self.curiosity.step(step_dt)

    # -- event hooks -----------------------------------------------------

    def on_message(
        self,
        channel_id: int,
        scope_id: str,
        sentiment: float = 0.0,
        user_id: Optional[int] = None,
        text: Optional[str] = None,
    ) -> None:
        """
        Call this on every incoming message, addressed or not.
        sentiment: rough -1..1 estimate of the message's tone (optional;
        0.0 is fine if you don't want to bother classifying it).
        user_id/text: if provided and self.graph is set, feeds the
        associative graph (see graph_memory.py) so topics/users light up
        and can seed unprompted messages later.
        """
        with self._lock:
            self._last_scope_activity[scope_id] = time.time()
            self._last_channel_activity[channel_id] = time.time()   # resets THIS channel's boredom
            self.energy.nudge(0.01)                                # mild stimulation
            self.valence.nudge(sentiment * 0.03)                   # tone rubs off a little
            self.social_battery.nudge(-0.01)                       # engaging costs a little battery

        if self.graph is not None and user_id is not None and text:
            from graph_memory import extract_topics  # local import keeps this module standalone if unused
            topics = [f"topic:{scope_id}:{topic}" for topic in extract_topics(text)]
            self.graph.record_interaction(
                user_id=f"user:{scope_id}:{user_id}",
                topics=topics,
                channel_id=f"channel:{scope_id}:{channel_id}",
            )
            # a topic that's genuinely warm right now feeds curiosity directly
            if topics:
                with self._lock:
                    self.curiosity.nudge(0.03)

    def on_bot_spoke(self, channel_id: int, scope_id: str, unprompted: bool = False) -> None:
        """Call right after the bot sends a message in a given channel."""
        with self._lock:
            self._last_scope_activity[scope_id] = time.time()
            self._last_channel_activity[channel_id] = time.time()   # resets THIS channel's boredom
            self.social_battery.nudge(-0.05 if not unprompted else -0.03)
            self.curiosity.nudge(-0.1)   # scratched the itch, temporarily
            self.energy.nudge(-0.02)

    def on_reacted_well(self) -> None:
        """Optional: call if a message got positive reactions/replies back."""
        with self._lock:
            self.valence.nudge(0.05)
            self.social_battery.nudge(0.03)  # positive social feedback is recharging, not costly

    def seconds_since_activity(self, channel_id: int) -> float:
        """How long since real activity in this channel. Falls back to
        'since we started watching' for a channel we've never seen a
        message in -- NOT infinity, so a fresh boot or a brand-new channel
        can't read as maximally, permanently silent."""
        last = self._last_channel_activity.get(channel_id, self._watch_start)
        return max(0.0, time.time() - last)

    def seconds_since_scope_activity(self, scope_id: str) -> float:
        """How long the guild or independent DM scope has been quiet."""
        last = self._last_scope_activity.get(scope_id, self._watch_start)
        return max(0.0, time.time() - last)

    def scope_boredom(self, scope_id: str) -> float:
        return min(1.0, self.boredom_rate * self.seconds_since_scope_activity(scope_id))

    def channel_boredom(self, channel_id: int) -> float:
        """Channel-level silence used to rank destinations within a scope."""
        return min(1.0, self.boredom_rate * self.seconds_since_activity(channel_id))

    # -- decision gating -----------------------------------------------------

    def speak_urge(self, channel_id: int, scope_id: str) -> float:
        """
        0..1 score for 'how much does the bot want to say something
        unprompted right now'. Feed this into autonomy_loop instead of
        a flat probability constant.
        """
        with self._lock:
            silence_factor = min(1.0, self.scope_boredom(scope_id) * 1.3)
            willingness = (
                0.40 * silence_factor
                + 0.25 * self.curiosity.value
                + 0.20 * self.energy.value
                + 0.15 * self.social_battery.value
            )
        # a warm topic sitting unaddressed is an extra little pull, on top
        # of the general curiosity/boredom math above
        if self.graph is not None:
            seed = self.graph.something_on_your_mind(
                channel_id=f"channel:{scope_id}:{channel_id}"
            )
            if seed:
                willingness += 0.10
        return max(0.0, min(1.0, willingness))

    def topic_seed(self, channel_id: int, scope_id: str) -> Optional[str]:
        """What the bot would organically bring up right now, if anything.
        Returns a bare topic string (e.g. 'kubernetes') or None."""
        if self.graph is None:
            return None
        node = self.graph.something_on_your_mind(
            channel_id=f"channel:{scope_id}:{channel_id}"
        )
        if node and node.startswith("topic:"):
            return node.rsplit(":", 1)[-1]
        return None

    def react_urge(self) -> float:
        """0..1 score for 'how likely to drop an emoji reaction right now'."""
        return max(0.0, min(1.0, 0.5 * self.energy.value + 0.5 * self.valence.value))

    def should_speak_unprompted(
        self,
        channel_id: int,
        scope_id: str,
        threshold: float = 0.55,
    ) -> bool:
        # Hard circuit breaker: never speak unprompted in the first few
        # minutes after (re)connecting, regardless of what the urge score
        # says. This is deliberately independent of the boredom math above --
        # belt and suspenders against exactly the "reconnect burst" failure
        # mode, even if some future change reintroduces it.
        if time.time() - self._boot_time < self.startup_grace_s:
            return False
        return self.speak_urge(channel_id, scope_id) + random.uniform(-0.08, 0.08) > threshold

    # -- persona prompt integration -----------------------------------------------------

    def render_prompt_fragment(
        self,
        channel_id: Optional[int] = None,
        scope_id: Optional[str] = None,
    ) -> str:
        """
        Turns numeric state into a short natural-language aside for the
        system/persona prompt. Keep this SHORT -- it's flavor, not a wall
        of stats, and it should never mention numbers. Pass channel_id to
        get the boredom line for that specific channel; omit it (e.g. for
        an addressed reply where boredom isn't the relevant thing) and
        that line is simply skipped.
        """
        with self._lock:
            e, v, s, c = (
                self.energy.value,
                self.valence.value,
                self.social_battery.value,
                self.curiosity.value,
            )
            b = self.scope_boredom(scope_id) if scope_id is not None else 0.0

        parts = []

        if e < 0.35:
            parts.append("you're running low on energy, a bit worn out")
        elif e > 0.75:
            parts.append("you're feeling energetic and a little restless")

        if v < 0.4:
            parts.append("your mood is a bit flat or irritable right now")
        elif v > 0.75:
            parts.append("you're in a genuinely good mood")

        if s < 0.35:
            parts.append("you don't have much social battery left, keep it brief")
        elif s > 0.8:
            parts.append("you're feeling talkative and open")

        if c > 0.7:
            parts.append("something's been nagging at your curiosity")

        if b > 0.6:
            parts.append("it's been quiet for a while and it's starting to itch at you")

        seed = self.topic_seed(channel_id, scope_id) if channel_id is not None and scope_id is not None else None
        if seed:
            parts.append(f"you keep thinking back to something about \"{seed}\" that came up earlier")

        if not parts:
            return "You feel pretty even and unremarkable right now -- nothing pulling at you either way."

        return "Right now, " + "; ".join(parts) + "."

    # -- persistence -----------------------------------------------------

    def _snapshot(self) -> dict:
        """Take one atomic, consistent snapshot of everything that needs
        to be persisted. Called on the calling thread, under the lock --
        cheap (just copying floats/dict entries), so it's fine to do
        synchronously even from async code. The actual slow disk I/O
        happens afterward, off-thread, operating only on this plain
        snapshot -- never touching live mutable state -- so there's no
        way for a concurrent tick()/on_message() to produce a torn read."""
        with self._lock:
            return {
                "energy": self.energy.value,
                "valence": self.valence.value,
                "social_battery": self.social_battery.value,
                "curiosity": self.curiosity.value,
                "scope_activity": dict(self._last_scope_activity),
                "channel_activity": dict(self._last_channel_activity),
                "saved_at": time.time(),
            }

    def _write_snapshot(self, snap: dict) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS mental_state (
                   id INTEGER PRIMARY KEY CHECK (id = 0),
                   energy REAL, valence REAL, social_battery REAL,
                   curiosity REAL, saved_at REAL
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS scope_activity (
                   scope_id TEXT PRIMARY KEY, last_active REAL
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS channel_activity (
                   channel_id INTEGER PRIMARY KEY, last_active REAL
               )"""
        )
        conn.execute(
            """INSERT INTO mental_state (id, energy, valence, social_battery, curiosity, saved_at)
               VALUES (0, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   energy=excluded.energy, valence=excluded.valence,
                   social_battery=excluded.social_battery, curiosity=excluded.curiosity,
                   saved_at=excluded.saved_at""",
            (snap["energy"], snap["valence"], snap["social_battery"],
             snap["curiosity"], snap["saved_at"]),
        )
        conn.executemany(
            """INSERT INTO scope_activity (scope_id, last_active) VALUES (?, ?)
               ON CONFLICT(scope_id) DO UPDATE SET last_active=excluded.last_active""",
            list(snap["scope_activity"].items()),
        )
        conn.executemany(
            """INSERT INTO channel_activity (channel_id, last_active) VALUES (?, ?)
               ON CONFLICT(channel_id) DO UPDATE SET last_active=excluded.last_active""",
            list(snap["channel_activity"].items()),
        )
        conn.commit()
        conn.close()

    def save(self) -> None:
        """Synchronous/blocking. Prefer save_async() from asyncio code
        (discord.py etc.) -- see save_async for why."""
        self._write_snapshot(self._snapshot())

    async def save_async(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Non-blocking entry point for asyncio callers. The snapshot is
        taken synchronously here (fast, lock-protected, just copies a few
        floats and a small dict) -- only the actual disk write is handed
        to a worker thread, so the event loop is never blocked and the
        worker thread never touches live mutable state."""
        snap = self._snapshot()
        loop = loop or asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._write_snapshot(snap))

    def load(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT energy, valence, social_battery, curiosity FROM mental_state WHERE id = 0"
            ).fetchone()
            scope_rows = conn.execute(
                "SELECT scope_id, last_active FROM scope_activity"
            ).fetchall()
            activity_rows = conn.execute(
                "SELECT channel_id, last_active FROM channel_activity"
            ).fetchall()
        except sqlite3.OperationalError:
            row = None
            scope_rows = []
            activity_rows = []
        conn.close()

        now = time.time()
        with self._lock:
            # Persist identities, but restart the boredom clocks at boot so
            # downtime cannot produce a synchronized reconnect burst.
            self._last_scope_activity = {scope_id: now for scope_id, _ts in scope_rows}
            self._last_channel_activity = {channel_id: now for channel_id, _ts in activity_rows}
            self._watch_start = now       # channels we've truly never seen start fresh, not "infinitely bored"
            self._boot_time = now         # (re)arm the startup grace window
            self._last_tick = now
            self._pending_dt = 0.0

            if row:
                energy, valence, social_battery, curiosity = row
                self.energy.value = energy
                self.valence.value = valence
                self.social_battery.value = social_battery
                self.curiosity.value = curiosity
