"""STEP 5: Ask the LLM to write an answer using ONLY the retrieved chunks.

We use Ollama, a free program that runs open-source LLMs (here llama3.2)
on your own computer. Ollama listens at http://localhost:11434, and we
talk to it with a plain HTTP request.

The prompt gives the LLM numbered sources and strict rules. The rules
reduce "hallucination" (the LLM inventing facts that are not in the book).
"""

import httpx

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about books.
Rules:
1. Use ONLY the information in the numbered sources below. Do not use outside knowledge.
2. After each fact, cite the source number in square brackets, like [1] or [2].
3. If the sources do not contain the answer, reply exactly:
   "I could not find the answer in the books."
4. Keep the answer clear and short."""


class LLMError(Exception):
    """Raised when the LLM cannot be reached or gives a bad reply."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Put the retrieved chunks and the question together into one message."""
    parts = []
    for number, chunk in enumerate(chunks, start=1): # enumerate the chunks starting at 1
        parts.append(f"[{number}] (from {chunk['file']}, page {chunk['page']})\n{chunk['text']}")
    sources = "\n\n".join(parts)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}\nAnswer:"


def ask_llm(prompt: str, config: dict) -> str:
    request_body = {
        "model": config["llm_model"],
        "stream": False,  # wait for the full answer instead of word by word
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": config["temperature"],
            "num_predict": config["max_answer_tokens"],
        },
    }
    url = f"{config['ollama_url']}/api/chat"

    try:
        response = httpx.post(url, json=request_body, timeout=config["llm_timeout_seconds"])
        response.raise_for_status()
    except httpx.ConnectError:
        raise LLMError("Cannot connect to Ollama. Start it with:  ollama serve")
    except httpx.TimeoutException:
        raise LLMError("The LLM took too long to answer. Try again or raise llm_timeout_seconds.")
    except httpx.HTTPStatusError as error:
        raise LLMError(f"Ollama returned an error: {error.response.text}")

    answer = response.json().get("message", {}).get("content", "").strip()
    if not answer:
        raise LLMError("The LLM returned an empty answer.")
    return answer
