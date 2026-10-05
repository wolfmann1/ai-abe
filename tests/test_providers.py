import os
import unittest
from unittest import mock

from abe.providers import (
    AnthropicProvider,
    AzureOpenAIProvider,
    OpenAICompatibleProvider,
    ProviderError,
    make_provider,
)
from abe.spec import ProviderConfig


def fake_response(status=200, payload=None):
    response = mock.Mock()
    response.status_code = status
    response.json.return_value = payload or {}
    response.text = str(payload)
    return response


OPENAI_SHAPE = {
    "choices": [{"message": {"content": "Answer [1]."}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 8},
}


class ProviderRequestTests(unittest.TestCase):
    @mock.patch.dict(os.environ, {"AZ_KEY": "secret"})
    @mock.patch("abe.providers.requests.post")
    def test_azure_request(self, post):
        post.return_value = fake_response(payload=OPENAI_SHAPE)
        provider = AzureOpenAIProvider(ProviderConfig(
            kind="azure_openai", model="gpt-4o-mini", endpoint="https://r.openai.azure.com/", api_key_env="AZ_KEY"))
        completion = provider.complete("sys", "user")
        url = post.call_args.args[0]
        self.assertEqual(
            url,
            "https://r.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-21",
        )
        self.assertEqual(post.call_args.kwargs["headers"]["api-key"], "secret")
        self.assertEqual(post.call_args.kwargs["json"]["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual((completion.text, completion.input_tokens, completion.output_tokens),
                         ("Answer [1].", 120, 8))

    @mock.patch("abe.providers.requests.post")
    def test_ollama_needs_no_key(self, post):
        post.return_value = fake_response(payload=OPENAI_SHAPE)
        provider = OpenAICompatibleProvider(ProviderConfig(kind="openai_compatible", model="llama3.1"))
        provider.complete("sys", "user")
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/v1/chat/completions")
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])
        self.assertEqual(post.call_args.kwargs["json"]["model"], "llama3.1")

    @mock.patch.dict(os.environ, {"A_KEY": "k"})
    @mock.patch("abe.providers.requests.post")
    def test_anthropic_request(self, post):
        post.return_value = fake_response(payload={
            "content": [{"type": "text", "text": "Hello"}],
            "usage": {"input_tokens": 50, "output_tokens": 2},
        })
        provider = AnthropicProvider(ProviderConfig(kind="anthropic", model="claude-x", api_key_env="A_KEY"))
        completion = provider.complete("sys", "user")
        self.assertEqual(post.call_args.args[0], "https://api.anthropic.com/v1/messages")
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["x-api-key"], "k")
        self.assertIn("anthropic-version", headers)
        self.assertEqual(post.call_args.kwargs["json"]["system"], "sys")
        self.assertEqual((completion.text, completion.input_tokens), ("Hello", 50))

    @mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key"})
    @mock.patch("abe.providers.requests.post")
    def test_openrouter_from_form(self, post):
        from abe.web import blueprint_from_form

        post.return_value = fake_response(payload=OPENAI_SHAPE)
        blueprint = blueprint_from_form({"name": "X", "provider": "openrouter", "model": "openai/gpt-5.2"})
        config = blueprint.spec.provider
        self.assertEqual((config.endpoint, config.api_key_env), ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"))
        OpenAICompatibleProvider(config).complete("sys", "user")
        self.assertEqual(post.call_args.args[0], "https://openrouter.ai/api/v1/chat/completions")
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer or-key")
        self.assertEqual(headers["X-OpenRouter-Title"], "ai-abe")
        self.assertIn("HTTP-Referer", headers)

    def test_missing_key_is_a_clear_error(self):
        provider = AzureOpenAIProvider(ProviderConfig(
            kind="azure_openai", model="m", endpoint="https://e", api_key_env="DEFINITELY_NOT_SET_123"))
        with self.assertRaises(ProviderError) as ctx:
            provider.complete("s", "u")
        self.assertIn("DEFINITELY_NOT_SET_123", str(ctx.exception))

    @mock.patch("abe.providers.requests.post")
    def test_http_error_surfaces(self, post):
        post.return_value = fake_response(status=429, payload={"error": "rate limited"})
        provider = OpenAICompatibleProvider(ProviderConfig(kind="openai_compatible", model="m"))
        with self.assertRaises(ProviderError) as ctx:
            provider.complete("s", "u")
        self.assertIn("429", str(ctx.exception))

    def test_factory(self):
        self.assertTrue(make_provider(ProviderConfig(kind="extractive")).extractive)


if __name__ == "__main__":
    unittest.main()
