import asyncio
import io
import unittest

from .helpers import EXAMPLE, TempDir, build_example


class WebTests(unittest.TestCase):
    def client(self, workspace):
        from starlette.testclient import TestClient

        from abe.web import create_app

        return TestClient(create_app(workspace))

    def test_form_renders(self):
        with TempDir() as tmp:
            page = self.client(tmp).get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("Build an agent", page.text)
            self.assertIn("Minimum viable accuracy", page.text)

    def test_build_ask_eval_download(self):
        with TempDir() as tmp:
            client = self.client(tmp)
            files = [
                ("documents", (p.name, p.read_bytes(), "text/plain")) for p in sorted((EXAMPLE / "docs").iterdir())
            ]
            form = {
                "name": "Stash Helper",
                "problem": "Runbooks are hard to find during incidents.",
                "provider": "extractive",
                "require_citations": "on",
                "data_readiness_score": "4",
                "q1": "How long are release builds kept?",
                "q1_mention": "365 days",
                "q1_source": "faq.txt",
                "users": "10",
                "questions_per_day": "3",
            }
            response = client.post("/build", data=form, files=files, follow_redirects=False)
            self.assertEqual(response.status_code, 303, response.text)
            self.assertEqual(response.headers["location"], "/agents/stash-helper")

            page = client.get("/agents/stash-helper")
            self.assertIn("Stash Helper", page.text)

            answer = client.post("/agents/stash-helper/ask", data={"question": "How long are release builds kept?"})
            self.assertIn("365 days", answer.text)

            report = client.post("/agents/stash-helper/eval")
            self.assertIn("PASS", report.text)
            self.assertTrue((tmp / "stash-helper" / "evals" / "report.md").exists())

            archive = client.get("/agents/stash-helper/download")
            self.assertEqual(archive.headers["content-type"], "application/zip")

    def test_rejects_bad_file_type_and_missing_name(self):
        with TempDir() as tmp:
            client = self.client(tmp)
            bad = client.post("/build", data={"name": "X"},
                              files=[("documents", ("sheet.xlsx", io.BytesIO(b"x"), "application/octet-stream"))])
            self.assertEqual(bad.status_code, 400)
            unnamed = client.post("/build", data={"name": ""})
            self.assertEqual(unnamed.status_code, 400)

    def test_unknown_agent_and_path_tricks(self):
        with TempDir() as tmp:
            client = self.client(tmp)
            self.assertEqual(client.get("/agents/nope").status_code, 404)
            self.assertEqual(client.get("/agents/..%2F..%2Fetc").status_code, 404)

    def test_safe_filename(self):
        from abe.web import safe_filename

        self.assertEqual(safe_filename("..\\..\\evil/Lease Abstract.PDF"), "Lease Abstract.pdf")
        self.assertEqual(safe_filename("a$b?.txt"), "a_b_.txt")


class MCPTests(unittest.TestCase):
    def test_tools(self):
        try:
            from abe.mcp_server import create_server
        except ImportError:
            self.skipTest("mcp not installed")
        with TempDir() as tmp:
            server = create_server(build_example(tmp))

            async def run():
                names = [t.name for t in await server.list_tools()]
                sources = await server.call_tool("list_sources", {})
                return names, str(sources)

            names, sources = asyncio.run(run())
            self.assertEqual(sorted(names), ["ask", "list_sources", "search_knowledge"])
            self.assertIn("faq.txt", sources)

    def test_retrieval_only_and_rules_prompt(self):
        try:
            from abe.mcp_server import create_server
        except ImportError:
            self.skipTest("mcp not installed")
        with TempDir() as tmp:
            server = create_server(build_example(tmp), retrieval_only=True)

            async def run():
                names = [t.name for t in await server.list_tools()]
                prompts = [p.name for p in await server.list_prompts()]
                rules = await server.get_prompt("agent_rules", {})
                return names, prompts, str(rules)

            names, prompts, rules = asyncio.run(run())
            self.assertEqual(sorted(names), ["list_sources", "search_knowledge"])
            self.assertEqual(prompts, ["agent_rules"])
            self.assertIn("Stash Ops Assistant", rules)
            self.assertIn("search_knowledge", rules)


if __name__ == "__main__":
    unittest.main()
