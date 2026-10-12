"""Claude LLM wrapper with retry/backoff and token tracking."""
import time
import threading
import anthropic

from config import ANTHROPIC_API_KEY, DEFAULT_MODEL, LLM_MAX_RETRIES, LLM_INITIAL_DELAY, LLM_MAX_TOKENS

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_token_lock = threading.Lock()
_total_input_tokens = 0
_total_output_tokens = 0
_total_calls = 0


class LLMOverloadedError(Exception):
    pass


def get_token_usage() -> dict:
    with _token_lock:
        return {
            "input_tokens": _total_input_tokens,
            "output_tokens": _total_output_tokens,
            "total_calls": _total_calls,
        }


def reset_token_usage():
    global _total_input_tokens, _total_output_tokens, _total_calls
    with _token_lock:
        _total_input_tokens = 0
        _total_output_tokens = 0
        _total_calls = 0


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Call Claude with a prompt. Returns the text response.
    Retries on 429/500/503/529 with exponential backoff.
    """
    global _total_input_tokens, _total_output_tokens, _total_calls
    delay = LLM_INITIAL_DELAY
    for attempt in range(LLM_MAX_RETRIES):
        try:
            message = _client.messages.create(
                model=model,
                max_tokens=LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(message, 'usage'):
                with _token_lock:
                    _total_input_tokens += message.usage.input_tokens
                    _total_output_tokens += message.usage.output_tokens
                    _total_calls += 1
            return "".join(b.text for b in message.content if b.type == "text")
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 500, 503, 529):
                print(f"WARNING: LLM status {e.status_code}, retry {attempt+1}/{LLM_MAX_RETRIES} in {delay}s...")
                if attempt < LLM_MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise LLMOverloadedError(
                        f"LLM failed after {LLM_MAX_RETRIES} attempts: {e}"
                    )
            else:
                raise
        except Exception as e:
            print(f"ERROR: Unexpected LLM error: {e}")
            raise
    return None
