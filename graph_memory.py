"""
graph_memory.py

A lightweight associative network -- users and topics as nodes, weighted
edges between them that strengthen with co-occurrence and decay with time
(classic spreading-activation / Hebbian "fire together, wire together"
network, in the tradition of old cognitive architectures like ACT-R --
not a trained model, just rules).

Two separate notions of "strength" are tracked, and conflating them is
the most common mistake with this kind of model:

  - EDGE WEIGHT   : long-term association strength ("this user talks
                    about this topic a lot"). Decays slowly. This is
                    closer to long-term memory / trait knowledge.

  - ACTIVATION     : short-term "how warm is this right now" heat.
                    Decays fast. This is closer to working memory / what's
                    currently top of mind. Activation SPREADS across
                    edges when something fires, attenuating each hop --
                    that's what makes a mention of one thing bring
                    related things to mind instead of only itself.

No ML, no gradients. Everything here is O(1)-ish per touch and cheap
to run on every message.
"""

from __future__ import annotations

import asyncio
import math
import re
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

NodeId = str  # e.g. "user:123456", "topic:kubernetes", "channel:987"


def _key(a: NodeId, b: NodeId) -> Tuple[NodeId, NodeId]:
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Cheap topic extraction -- swap this out for something smarter
# (an LLM call, a keyword model, whatever) without touching the graph logic.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "for", "with", "that", "this", "it",
    "you", "i", "we", "they", "he", "she", "at", "as", "so", "if", "just",
    "not", "yeah", "like", "im", "its", "your", "my", "me", "u", "lol",
}


def extract_topics(text: str, max_topics: int = 4) -> List[str]:
    """Very crude noun-ish keyword extraction. Good enough as a seed --
    replace with a real extractor (or a cheap LLM call) if you want
    sharper topics."""
    words = re.findall(r"[a-zA-Z']{3,}", text.lower())
    candidates = [w for w in words if w not in _STOPWORDS]
    # keep first occurrence order, dedup
    seen = []
    for w in candidates:
        if w not in seen:
            seen.append(w)
    return seen[:max_topics]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

@dataclass
class _Edge:
    weight: float = 0.0
    last_touched: float = field(default_factory=time.time)


@dataclass
class _Node:
    activation: float = 0.0
    last_active: float = field(default_factory=time.time)
    last_mentioned: float = 0.0  # last time it was actually spoken about, not just activated


class AssociativeGraph:
    def __init__(
        self,
        db_path: str = "khronic_state.db",
        edge_half_life_s: float = 60 * 60 * 24 * 3,   # associations fade over ~3 days
        activation_half_life_s: float = 60 * 20,        # "top of mind" fades over ~20 min
        spread_hops: int = 2,
        spread_attenuation: float = 0.45,
        min_edge_transfer: float = 0.30,
    ):
        self.db_path = db_path
        self.edge_half_life_s = edge_half_life_s
        self.activation_half_life_s = activation_half_life_s
        self.spread_hops = spread_hops
        self.spread_attenuation = spread_attenuation
        # Floor on how much a weak/decayed edge can still carry per hop.
        # Without this, spread_attenuation (generic hop decay) and edge
        # weight (specific association decay) stack multiplicatively and
        # a merely-average edge crushes the signal before hop 2. See:
        # https://en.wikipedia.org/wiki/Spreading_activation -- classic
        # models modulate transfer by association strength, they don't
        # gate it to near-zero on top of a hop decay that's already there.
        self.min_edge_transfer = min_edge_transfer

        self._edges: Dict[Tuple[NodeId, NodeId], _Edge] = {}
        self._adj: Dict[NodeId, set] = defaultdict(set)
        self._nodes: Dict[NodeId, _Node] = {}
        self._lock = threading.RLock()
        self._version = 0

        # Dirty-tracking so save() only writes what actually changed since
        # the last save, instead of rewriting the whole graph every time.
        # Values are mutation versions, allowing save() to avoid clearing a
        # key that changed again while its previous snapshot was being written.
        self._dirty_nodes: Dict[NodeId, int] = {}
        self._dirty_edges: Dict[Tuple[NodeId, NodeId], int] = {}

    def _mark_node_dirty(self, node_id: NodeId) -> None:
        self._version += 1
        self._dirty_nodes[node_id] = self._version

    def _mark_edge_dirty(self, key: Tuple[NodeId, NodeId]) -> None:
        self._version += 1
        self._dirty_edges[key] = self._version

    # -- decay helpers -----------------------------------------------------

    @staticmethod
    def _decay_factor(elapsed_s: float, half_life_s: float) -> float:
        if half_life_s <= 0:
            return 1.0
        return math.pow(0.5, elapsed_s / half_life_s)

    def _node(self, node_id: NodeId) -> _Node:
        if node_id not in self._nodes:
            self._nodes[node_id] = _Node()
        return self._nodes[node_id]

    def _decayed_activation(self, node_id: NodeId) -> float:
        n = self._nodes.get(node_id)
        if not n:
            return 0.0
        elapsed = time.time() - n.last_active
        n.activation *= self._decay_factor(elapsed, self.activation_half_life_s)
        n.last_active = time.time()
        return n.activation

    def _decayed_weight(self, key: Tuple[NodeId, NodeId]) -> float:
        e = self._edges.get(key)
        if not e:
            return 0.0
        elapsed = time.time() - e.last_touched
        e.weight *= self._decay_factor(elapsed, self.edge_half_life_s)
        e.last_touched = time.time()
        return e.weight

    # -- writing -----------------------------------------------------

    def touch_edge(self, a: NodeId, b: NodeId, boost: float = 0.15) -> None:
        """Call whenever two nodes co-occur (a user mentions a topic, two
        users talk in the same thread, etc.). Strengthens their long-term
        association a little each time -- repetition builds the bond."""
        with self._lock:
            if a == b:
                return
            key = _key(a, b)
            current = self._decayed_weight(key)
            e = self._edges.setdefault(key, _Edge())
            e.weight = min(1.0, current + boost)
            e.last_touched = time.time()
            self._adj[a].add(b)
            self._adj[b].add(a)
            self._mark_edge_dirty(key)

    def activate(self, node_id: NodeId, amount: float = 1.0, spread: bool = True) -> None:
        """Fire a node directly (e.g. a user just spoke, a topic was just
        mentioned), and optionally let that activation spread outward
        through associated nodes, attenuating each hop."""
        with self._lock:
            n = self._node(node_id)
            n.activation = min(1.0, self._decayed_activation(node_id) + amount)
            n.last_active = time.time()
            n.last_mentioned = time.time()
            self._mark_node_dirty(node_id)

            if not spread:
                return

            frontier = {node_id}
            seen = {node_id}
            current_amount = amount
            for _ in range(self.spread_hops):
                current_amount *= self.spread_attenuation
                if current_amount < 0.02:
                    break
                next_frontier = set()
                for node in frontier:
                    for neighbor in self._adj.get(node, ()):
                        if neighbor in seen:
                            continue
                        w = self._decayed_weight(_key(node, neighbor))
                        if w <= 0:
                            continue
                        transfer = current_amount * (self.min_edge_transfer + (1 - self.min_edge_transfer) * w)
                        nb = self._node(neighbor)
                        nb.activation = min(1.0, self._decayed_activation(neighbor) + transfer)
                        nb.last_active = time.time()
                        next_frontier.add(neighbor)
                        seen.add(neighbor)
                        self._mark_node_dirty(neighbor)
                frontier = next_frontier
                if not frontier:
                    break

    def record_interaction(self, user_id: NodeId, topics: Iterable[NodeId], channel_id: Optional[NodeId] = None) -> None:
        """Convenience wrapper: call once per message. Links the user to
        each topic mentioned, links topics to each other (co-occurrence),
        and activates everything involved."""
        with self._lock:
            topics = list(topics)
            self.activate(user_id, amount=0.6)
            if channel_id:
                self.touch_edge(user_id, channel_id, boost=0.05)

            for t in topics:
                self.touch_edge(user_id, t, boost=0.15)
                self.activate(t, amount=0.8)
                if channel_id:
                    self.touch_edge(t, channel_id, boost=0.05)

            for i, t1 in enumerate(topics):
                for t2 in topics[i + 1:]:
                    self.touch_edge(t1, t2, boost=0.10)

    # -- reading -----------------------------------------------------

    def warmth(self, node_id: NodeId) -> float:
        """Current 'how top-of-mind is this' score, 0..1."""
        return self._decayed_activation(node_id)

    def association(self, a: NodeId, b: NodeId) -> float:
        """Current long-term association strength between two nodes, 0..1."""
        return self._decayed_weight(_key(a, b))

    def top_active(self, prefix: Optional[str] = None, n: int = 5, exclude_recent_s: float = 0) -> List[Tuple[NodeId, float]]:
        """Highest-activation nodes right now, optionally filtered by id
        prefix (e.g. 'topic:'). exclude_recent_s lets you skip things that
        were JUST mentioned, so you don't repeat yourself immediately."""
        now = time.time()
        scored = []
        for node_id in list(self._nodes.keys()):
            if prefix and not node_id.startswith(prefix):
                continue
            if exclude_recent_s and (now - self._nodes[node_id].last_mentioned) < exclude_recent_s:
                continue
            score = self._decayed_activation(node_id)
            if score > 0.01:
                scored.append((node_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def associated_with(self, node_id: NodeId, n: int = 5) -> List[Tuple[NodeId, float]]:
        """Things most strongly (long-term) associated with a given node --
        e.g. 'what does this user usually talk about'."""
        scored = [(nb, self._decayed_weight(_key(node_id, nb))) for nb in self._adj.get(node_id, ())]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def something_on_your_mind(
        self,
        channel_id: Optional[NodeId] = None,
        min_warmth: float = 0.15,
        quiet_for_s: float = 60 * 10,
    ) -> Optional[str]:
        """Pick a topic that's still warm but hasn't been brought up
        recently -- good seed for an unprompted message. Returns the raw
        node id (e.g. 'topic:kubernetes') or None if nothing qualifies."""
        with self._lock:
            candidates = self.top_active(prefix="topic:", n=50, exclude_recent_s=quiet_for_s)
            if channel_id is not None:
                channel_topics = {
                    node_id for node_id in self._adj.get(channel_id, ())
                    if node_id.startswith("topic:")
                }
                candidates = [
                    (node_id, warmth * (0.5 + 0.5 * self._decayed_weight(_key(channel_id, node_id))))
                    for node_id, warmth in candidates
                    if node_id in channel_topics
                ]
                candidates.sort(key=lambda item: item[1], reverse=True)
            candidates = [candidate for candidate in candidates if candidate[1] >= min_warmth]
            return candidates[0][0] if candidates else None

    # -- persistence -----------------------------------------------------

    def save(self, full: bool = False) -> None:
        """
        Persist to SQLite. By default only writes nodes/edges touched since
        the last save (bounded by recent activity, not total graph size) --
        pass full=True occasionally (e.g. once a day) to also flush any
        entries that only changed via passive decay, and to compact away
        rows for anything pruned out of memory (see prune()).

        This is synchronous/blocking -- call it via save_async() from
        asyncio code (e.g. discord.py) so it never runs on the event loop
        thread.
        """
        with self._lock:
            captured_nodes = dict(self._dirty_nodes)
            captured_edges = dict(self._dirty_edges)
            node_ids = list(self._nodes.keys()) if full else list(captured_nodes)
            edge_keys = list(self._edges.keys()) if full else list(captured_edges)
            node_rows = [
                (nid, self._decayed_activation(nid), self._nodes[nid].last_active, self._nodes[nid].last_mentioned)
                for nid in node_ids if nid in self._nodes
            ]
            edge_rows = [
                (a, b, self._decayed_weight((a, b)), self._edges[(a, b)].last_touched)
                for (a, b) in edge_keys if (a, b) in self._edges
            ]

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")   # lets reads proceed during this write
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS graph_edges (
                   a TEXT, b TEXT, weight REAL, last_touched REAL,
                   PRIMARY KEY (a, b)
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS graph_nodes (
                   node_id TEXT PRIMARY KEY, activation REAL,
                   last_active REAL, last_mentioned REAL
               )"""
        )
        if full:
            conn.execute("DELETE FROM graph_edges")
            conn.execute("DELETE FROM graph_nodes")
        conn.executemany(
            """INSERT INTO graph_edges (a, b, weight, last_touched) VALUES (?, ?, ?, ?)
               ON CONFLICT(a, b) DO UPDATE SET
                   weight=excluded.weight, last_touched=excluded.last_touched""",
            edge_rows,
        )
        conn.executemany(
            """INSERT INTO graph_nodes (node_id, activation, last_active, last_mentioned) VALUES (?, ?, ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET
                   activation=excluded.activation, last_active=excluded.last_active,
                   last_mentioned=excluded.last_mentioned""",
            node_rows,
        )
        conn.commit()
        conn.close()

        with self._lock:
            for node_id, version in captured_nodes.items():
                if self._dirty_nodes.get(node_id) == version:
                    self._dirty_nodes.pop(node_id, None)
            for edge_key, version in captured_edges.items():
                if self._dirty_edges.get(edge_key) == version:
                    self._dirty_edges.pop(edge_key, None)

    async def save_async(self, loop: Optional[asyncio.AbstractEventLoop] = None, full: bool = False) -> None:
        """Non-blocking entry point for asyncio callers (discord.py etc).
        Runs the actual sqlite3 I/O in a worker thread so it can't stall
        the gateway heartbeat even if the write takes a while."""
        loop = loop or asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.save(full=full))

    def prune(self, min_weight: float = 0.01, min_activation: float = 0.005) -> None:
        """Drop edges/nodes that have decayed to near-nothing, so the
        in-memory graph (and eventually the DB, via a following full save)
        doesn't grow unboundedly under real traffic. Call this occasionally
        on a slow timer (e.g. once a day), not on every message."""
        with self._lock:
            dead_edges = [k for k in list(self._edges.keys()) if self._decayed_weight(k) < min_weight]
            for a, b in dead_edges:
                del self._edges[(a, b)]
                self._adj[a].discard(b)
                self._adj[b].discard(a)
                self._dirty_edges.pop((a, b), None)

            dead_nodes = [
                nid for nid in list(self._nodes.keys())
                if self._decayed_activation(nid) < min_activation and not self._adj.get(nid)
            ]
            for nid in dead_nodes:
                del self._nodes[nid]
                self._adj.pop(nid, None)
                self._dirty_nodes.pop(nid, None)

    def load(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            edge_rows = conn.execute("SELECT a, b, weight, last_touched FROM graph_edges").fetchall()
            node_rows = conn.execute("SELECT node_id, activation, last_active, last_mentioned FROM graph_nodes").fetchall()
        except sqlite3.OperationalError:
            conn.close()
            return
        conn.close()

        with self._lock:
            for a, b, weight, last_touched in edge_rows:
                self._edges[_key(a, b)] = _Edge(weight=weight, last_touched=last_touched)
                self._adj[a].add(b)
                self._adj[b].add(a)

            for node_id, activation, last_active, last_mentioned in node_rows:
                self._nodes[node_id] = _Node(
                    activation=activation,
                    last_active=last_active,
                    last_mentioned=last_mentioned,
                )
