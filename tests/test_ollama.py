"""Tests against a small fake Ollama server running on a local port."""

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from abe import ollama

from .helpers import EXAMPLE, TempDir


class FakeOllama(BaseHTTPRequestHandler):
    installed = ["llama3.1:latest", "qwen2.5:7b"]
    fail_pulls = False

    def log_message(self, *args):
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": n} for n in self.installed]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/pull":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            if self.fail_pulls:
                self.wfile.write(b'{"error":"pull model manifest: file does not exist"}\n')
                return
            for event in (
                {"status": "pulling manifest"},
                {"status": "downloading", "total": 100, "completed": 50},
                {"status": "downloading", "total": 100, "completed": 100},
                {"status": "success"},
            ):
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
            type(self).installed = [*self.installed, body["model"]]
        elif self.path == "/v1/chat/completions":
            self._json({
                "choices": [{"message": {"content": "Release builds are kept for 365 days [1]."}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 12},
            })
        else:
            self._json({"error": "not found"}, 404)


class OllamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeOllama.installed = ["llama3.1:latest", "qwen2.5:7b"]
        FakeOllama.fail_pulls = False
        ollama._pulls.clear()
        self.env = mock.patch.dict(os.environ, {"OLLAMA_HOST": self.base})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def wait_for(self, model):
        for _ in range(100):
            status = ollama.pull_status(model)
            if status and not status.active:
                return status
            time.sleep(0.02)
        self.fail("pull did not finish")

    def test_host_and_listing(self):
        self.assertEqual(ollama.host(), self.base)
        self.assertEqual(ollama.openai_endpoint(), f"{self.base}/v1")
        self.assertEqual(ollama.list_models(), ["llama3.1:latest", "qwen2.5:7b"])

    def test_unreachable_returns_none(self):
        self.assertIsNone(ollama.list_models("http://127.0.0.1:9"))

    def test_latest_tag_matching(self):
        installed = ["llama3.1:latest", "qwen2.5:7b"]
        self.assertTrue(ollama.is_installed("llama3.1", installed))
        self.assertTrue(ollama.is_installed("qwen2.5:7b", installed))
        self.assertFalse(ollama.is_installed("qwen2.5", installed))

    def test_pull_success_and_failure(self):
        status = self.wait_for(ollama.start_pull("mistral").model)
        self.assertEqual((status.state, status.percent), ("done", 100))
        self.assertIn("mistral", ollama.list_models())

        FakeOllama.fail_pulls = True
        status = self.wait_for(ollama.start_pull("not-a-model").model)
        self.assertEqual(status.state, "error")
        self.assertIn("does not exist", status.message)

    def test_form_lists_models_and_build_starts_pull(self):
        from starlette.testclient import TestClient

        from abe.web import create_app

        with TempDir() as tmp:
            client = TestClient(create_app(tmp))
            page = client.get("/")
            self.assertIn('data-model="qwen2.5:7b"', page.text)
            self.assertIn("ollama pull", page.text)

            files = [("documents", (p.name, p.read_bytes(), "text/plain")) for p in (EXAMPLE / "docs").iterdir()]
            response = client.post("/build", data={"name": "Local Helper", "provider": "ollama", "model": "mistral"},
                                   files=files, follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            self.assertEqual(self.wait_for("mistral").state, "done")

            page = client.get("/agents/local-helper")
            self.assertIn("Ready.", page.text)
            answer = client.post("/agents/local-helper/ask", data={"question": "How long are release builds kept?"})
            self.assertIn("365 days", answer.text)

    def test_failed_pull_shows_manual_instructions(self):
        from starlette.testclient import TestClient

        from abe.web import create_app

        FakeOllama.fail_pulls = True
        with TempDir() as tmp:
            client = TestClient(create_app(tmp))
            files = [("documents", (p.name, p.read_bytes(), "text/plain")) for p in (EXAMPLE / "docs").iterdir()]
            client.post("/build", data={"name": "Local Helper", "provider": "ollama", "model": "typo-model"},
                        files=files)
            self.wait_for("typo-model")
            page = client.get("/agents/local-helper")
            self.assertIn("failed", page.text)
            self.assertIn("ollama pull typo-model", page.text)


if __name__ == "__main__":
    unittest.main()
