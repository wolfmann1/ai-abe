"""Model providers, called over plain HTTPS.

Each provider turns (system prompt, user message) into a Completion. They use
`requests` directly rather than vendor SDKs, which keeps the dependency list short
and makes the request each provider sends easy to read and test.

- azure_openai       Azure OpenAI chat completions (deployment name goes in `model`)
- openai_compatible  OpenAI's API, or anything that speaks it: Ollama, LM Studio, vLLM
- anthropic          Anthropic Messages API
- extractive         No model. Returns the best-matching passages verbatim. Useful as an
                     offline baseline, for CI, and for showing whether an LLM adds value.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests

from .spec import ProviderConfig

TIMEOUT_SECONDS = 60
ANTHROPIC_VERSION = "2023-06-01"
OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(RuntimeError):
    pass


class Provider:
    kind = "base"
    extractive = False

    def __init__(self, config: ProviderConfig):
        self.config = config

    def complete(self, system: str, user: str) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError

    def _key(self, required: bool = True) -> str:
        name = self.config.api_key_env
        value = os.environ.get(name, "") if name else ""
        if required and not value:
            raise ProviderError(
                f"No API key found. Set the environment variable named in provider.api_key_env "
                f"({name or 'not set'})."
            )
        return value

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        try:
            response = requests.post(url, headers=headers, json=body, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise ProviderError(f"Request to {url} failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"{self.kind} returned HTTP {response.status_code}: {response.text[:500]}")
        return response.json()


class AzureOpenAIProvider(Provider):
    kind = "azure_openai"

    def complete(self, system: str, user: str) -> Completion:
        if not self.config.endpoint or not self.config.model:
            raise ProviderError("Azure OpenAI needs provider.endpoint and provider.model (the deployment name).")
        url = (
            f"{self.config.endpoint.rstrip('/')}/openai/deployments/{self.config.model}"
            f"/chat/completions?api-version={self.config.api_version}"
        )
        data = self._post(
            url,
            {"api-key": self._key(), "Content-Type": "application/json"},
            {
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
        )
        return _openai_completion(data)


class OpenAICompatibleProvider(Provider):
    kind = "openai_compatible"

    def complete(self, system: str, user: str) -> Completion:
        if not self.config.model:
            raise ProviderError("provider.model is required (e.g. a model name pulled into Ollama).")
        base = (self.config.endpoint or OLLAMA_DEFAULT_URL).rstrip("/")
        headers = {"Content-Type": "application/json"}
        key = self._key(required=False)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = self._post(
            f"{base}/chat/completions",
            headers,
            {
                "model": self.config.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
        )
        return _openai_completion(data)


class AnthropicProvider(Provider):
    kind = "anthropic"

    def complete(self, system: str, user: str) -> Completion:
        if not self.config.model:
            raise ProviderError("provider.model is required for Anthropic.")
        base = (self.config.endpoint or "https://api.anthropic.com").rstrip("/")
        data = self._post(
            f"{base}/v1/messages",
            {
                "x-api-key": self._key(),
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        return Completion(text, usage.get("input_tokens", 0), usage.get("output_tokens", 0))


class ExtractiveProvider(Provider):
    """Answers by quoting the passages most relevant to the question."""

    kind = "extractive"
    extractive = True

    def answer(self, question: str, passages: list[tuple[int, str]], max_sentences: int = 3) -> Completion:
        import math

        from .index import tokenize

        wanted = set(tokenize(question))
        sentences: list[tuple[int, int, str, set[str]]] = []
        for number, text in passages:
            for position, sentence in enumerate(re.split(r"(?<=[.!?])\s+|\n+", text)):
                sentence = sentence.strip()
                if len(sentence) >= 3:
                    sentences.append((number, position, sentence, set(tokenize(sentence))))
        if not sentences:
            return Completion("")
        # Weight each question term by how rare it is among the candidate sentences.
        df = {t: sum(t in terms for *_, terms in sentences) for t in wanted}
        idf = {t: math.log(1 + len(sentences) / d) for t, d in df.items() if d}
        scored = []
        for number, position, sentence, terms in sentences:
            score = sum(idf.get(t, 0.0) for t in terms & wanted)
            if score:
                scored.append((score, number, position, sentence))
        best = sorted(scored, key=lambda s: (-s[0], s[1], s[2]))[:max_sentences]
        best.sort(key=lambda s: (s[1], s[2]))
        return Completion(" ".join(f"{sentence} [{number}]" for _s, number, _p, sentence in best))


PROVIDERS: dict[str, type[Provider]] = {
    "azure_openai": AzureOpenAIProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "extractive": ExtractiveProvider,
}


def make_provider(config: ProviderConfig) -> Provider:
    try:
        return PROVIDERS[config.kind](config)
    except KeyError as exc:
        raise ProviderError(f"Unknown provider kind: {config.kind}") from exc


def _openai_completion(data: dict) -> Completion:
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as exc:
        raise ProviderError(f"Unexpected response shape: {str(data)[:300]}") from exc
    usage = data.get("usage") or {}
    return Completion(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
