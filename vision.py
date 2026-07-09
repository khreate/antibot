import base64
import logging

import httpx

import config
import llm_client

log = logging.getLogger("sentient_bot.vision")

# Kept separate from persona.py's chat-voice prompts on purpose: this call isn't khronic
# talking, it's a plain describe-the-image utility call whose output gets handed to khronic
# as context. Asking for a neutral, factual description keeps it from fighting the persona.
VISION_SYSTEM_PROMPT = (
    "Describe what's in the image factually and concisely, for another AI to use as context in "
    "a Discord conversation. Mention any visible text verbatim. 2-4 sentences. No preamble like "
    "'the image shows' or 'this appears to be' -- just the content."
)

# discord.Attachment.content_type is e.g. "image/png; charset=..." for some clients, so we
# only compare the part before ';'. gif/webp included since Discord serves both commonly.
_SUPPORTED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}


def _image_attachments(attachments) -> list:
    """Filter a message's attachments down to ones that look like images, capped to keep a
    single message from stalling the reply on a big dump of files."""
    images = [
        a for a in attachments
        if (a.content_type or "").split(";")[0].strip().lower() in _SUPPORTED_CONTENT_TYPES
    ]
    return images[:config.VISION_MAX_IMAGES_PER_MESSAGE]


async def _describe_one(client: httpx.AsyncClient, url: str) -> str | None:
    """Download one image and get a description from the vision model. Returns None on any
    failure (bad download, model error, empty response) -- callers treat that as non-fatal,
    same pattern as web_search.search()."""
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        encoded = base64.b64encode(resp.content).decode("ascii")
    except Exception as e:
        log.warning(f"Couldn't download image attachment {url!r}: {e}")
        return None

    messages = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {"role": "user", "content": "Describe this image.", "images": [encoded]},
    ]
    try:
        result = await llm_client.chat(
            messages,
            model=config.VISION_MODEL_NAME,
            temperature=0.2,
            max_tokens=config.VISION_MAX_TOKENS,
        )
    except Exception as e:
        log.warning(f"Vision model call failed for {url!r}: {e}")
        return None
    return result.strip() or None


async def describe_attachments(attachments) -> str | None:
    """Look at a Discord message's attachments and, if any are images, ask Qwen-VL to describe
    them. Returns a compact text block ready to append to the message's text content, or None
    if vision is disabled, there are no image attachments, or every description attempt failed.

    This is the only entry point bot.py needs -- it's a cheap no-op (no network/model call at
    all) for the overwhelming majority of messages that carry no images.
    """
    if not config.VISION_ENABLED:
        return None
    images = _image_attachments(attachments)
    if not images:
        return None

    descriptions: list[str] = []
    async with httpx.AsyncClient(timeout=config.VISION_TIMEOUT_SECONDS) as client:
        for att in images:
            desc = await _describe_one(client, att.url)
            if desc:
                descriptions.append(desc)

    if not descriptions:
        return None

    if len(descriptions) == 1:
        return f"[attached image -- {descriptions[0]}]"
    lines = "\n".join(f"  {i + 1}. {d}" for i, d in enumerate(descriptions))
    return f"[attached images:\n{lines}\n]"
