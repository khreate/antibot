import time
import aiosqlite

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    user_id INTEGER,
    username TEXT,
    role TEXT NOT NULL,           -- 'user' or 'assistant'
    content TEXT NOT NULL,
    ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    user_id INTEGER PRIMARY KEY,
    summary TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS channel_state (
    channel_id INTEGER PRIMARY KEY,
    mood TEXT DEFAULT 'neutral, settling in',
    last_message_ts REAL,
    last_summarized_id INTEGER DEFAULT 0
);
"""


class Memory:
    def __init__(self, path: str = config.DB_PATH):
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        self._db = await aiosqlite.connect(self.path)
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    # --- messages ---

    async def add_message(self, channel_id: int, user_id: int | None, username: str | None,
                           role: str, content: str) -> int:
        cursor = await self._db.execute(
            "INSERT INTO messages (channel_id, user_id, username, role, content, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (channel_id, user_id, username, role, content, time.time()),
        )
        row_id = cursor.lastrowid
        await self._db.execute(
            "INSERT INTO channel_state (channel_id, last_message_ts) VALUES (?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET last_message_ts=excluded.last_message_ts",
            (channel_id, time.time()),
        )
        await self._db.commit()
        return row_id

    async def reassign_messages(self, message_ids: list[int], new_channel_id: int):
        """Move stored messages to another channel's context (e.g. when a reply spins up a
        thread, the triggering user messages move with it so the parent channel is left clean
        and the thread's context is self-contained)."""
        if not message_ids:
            return
        placeholders = ",".join("?" for _ in message_ids)
        await self._db.execute(
            f"UPDATE messages SET channel_id = ? WHERE id IN ({placeholders})",
            (new_channel_id, *message_ids),
        )
        await self._db.commit()

    async def get_recent_messages(self, channel_id: int, limit: int = config.SHORT_TERM_TURNS):
        cursor = await self._db.execute(
            "SELECT role, content, username FROM messages WHERE channel_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (channel_id, limit),
        )
        rows = await cursor.fetchall()
        return list(reversed(rows))  # chronological order

    async def messages_since(self, channel_id: int, after_id: int):
        cursor = await self._db.execute(
            "SELECT id, role, content, username FROM messages "
            "WHERE channel_id = ? AND id > ? ORDER BY id ASC",
            (channel_id, after_id),
        )
        return await cursor.fetchall()

    async def latest_message_id(self, channel_id: int) -> int:
        cursor = await self._db.execute(
            "SELECT COALESCE(MAX(id), 0) FROM messages WHERE channel_id = ?", (channel_id,)
        )
        row = await cursor.fetchone()
        return row[0]

    # --- profiles (per-user long-term memory) ---

    async def get_profile(self, user_id: int) -> str | None:
        cursor = await self._db.execute(
            "SELECT summary FROM profiles WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def upsert_profile(self, user_id: int, summary: str):
        await self._db.execute(
            "INSERT INTO profiles (user_id, summary, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at",
            (user_id, summary, time.time()),
        )
        await self._db.commit()

    # --- channel state (mood, idle tracking, summarization cursor) ---

    async def get_channel_state(self, channel_id: int) -> dict:
        cursor = await self._db.execute(
            "SELECT mood, last_message_ts, last_summarized_id FROM channel_state WHERE channel_id = ?",
            (channel_id,),
        )
        row = await cursor.fetchone()
        if not row:
            await self._db.execute(
                "INSERT INTO channel_state (channel_id) VALUES (?)", (channel_id,)
            )
            await self._db.commit()
            return {"mood": "neutral, settling in", "last_message_ts": None, "last_summarized_id": 0}
        return {"mood": row[0], "last_message_ts": row[1], "last_summarized_id": row[2]}

    async def set_mood(self, channel_id: int, mood: str):
        await self._db.execute(
            "UPDATE channel_state SET mood = ? WHERE channel_id = ?", (mood, channel_id)
        )
        await self._db.commit()

    async def set_last_summarized_id(self, channel_id: int, msg_id: int):
        await self._db.execute(
            "UPDATE channel_state SET last_summarized_id = ? WHERE channel_id = ?",
            (msg_id, channel_id),
        )
        await self._db.commit()

    async def all_known_channels(self):
        cursor = await self._db.execute("SELECT channel_id FROM channel_state")
        rows = await cursor.fetchall()
        return [r[0] for r in rows]
