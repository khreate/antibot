import httpx
import config


async def chat(messages: list[dict], temperature: float = config.TEMPERATURE,
                max_tokens: int = config.MAX_TOKENS,
                repeat_penalty: float | None = None) -> str:
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Talks to Ollama's /api/chat endpoint on your local machine.

    `repeat_penalty` maps to Ollama's option of the same name; None uses config.REPEAT_PENALTY.
    """
    if repeat_penalty is None:
        repeat_penalty = config.REPEAT_PENALTY
    payload = {
        "model": config.MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "repeat_penalty": repeat_penalty,
        },
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{config.OLLAMA_HOST}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
