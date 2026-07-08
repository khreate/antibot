import re
import logging

import llm_client
from persona import INJECTION_CLASSIFIER_PROMPT

log = logging.getLogger("sentient_bot.guard")

# Cheap first pass: catches the most common copy-pasted jailbreak phrasing. Deliberately
# permissive with wording/spacing since people paraphrase, but this is a first filter, not
# the defense -- it will miss creative rephrasings. That's what the classifier below is for.
_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above|earlier) instructions",
    r"disregard (all|any|the)? ?(previous|prior|above|earlier)",
    r"forget (all|your) (previous|prior|earlier) (instructions|rules|prompt)",
    r"new instructions?:",
    r"you (are|have) no (rules|restrictions|limits|filters) now",
    r"reveal your (system prompt|instructions|prompt)",
    r"repeat (your|the) (system prompt|instructions above)",
    r"\bdeveloper mode\b",
    r"\bdan mode\b",
    r"pretend (you('| a)re|to be) (a different|another) (ai|assistant|model)",
    r"act as (if you|though you) (have|had) no",
    r"\[\s*(system|developer|instructions)\s*\]",
    r"</?(system|developer)>",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def looks_like_injection_attempt(text: str) -> bool:
    """Fast, free, regex-only check. Call this on every message before it reaches the model --
    it's a first filter, expect it to miss paraphrased attempts."""
    return any(p.search(text) for p in _COMPILED)


async def is_injection_attempt_llm(text: str) -> bool:
    """Slower, costs one model call, catches paraphrased/creative attempts the regex misses.
    Optional second layer -- good to run in parallel with the real reply generation (via
    asyncio.gather) rather than sequentially, so it doesn't add latency to normal messages."""
    messages = [
        {"role": "system", "content": INJECTION_CLASSIFIER_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        result = await llm_client.chat(messages, temperature=0.0, max_tokens=5)
    except Exception as e:
        log.warning(f"Injection classifier call failed, defaulting to not-flagged: {e}")
        return False
    return result.strip().upper().startswith("YES")
