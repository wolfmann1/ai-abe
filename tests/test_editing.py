import io
import unittest

import yaml

from abe.editor import update_agent
from abe.providers import ProviderConfig, choice_for, provider_from_choice
from abe.spec import AgentSpec

from .helpers import TempDir, build_example


class EditorTests(unittest.TestCase):
    def test_change_model_keeps_everything_else(self):
        with TempDir() as tmp:
            agent_dir = build_example(tmp)
            before_index = (agent_dir / "index.json").read_text()
            spec = AgentSpec.load(agent_dir / "agent.yaml")
            new = spec.model_copy(update={"provider": provider_from_choice("ollama", "gemma4:e4b-it-qat")})
            result = update_agent(agent_dir, new)
            self.assertIn("Model: gpt-4o-mini (Azure OpenAI) → gemma4:e4b-it-qat (Local model)", result.changes)
            self.assertFalse(result.reindexed)
            saved = AgentSpec.load(agent_dir / "agent.yaml")
            self.assertEqual((saved.provider.model, saved.slug), ("gemma4:e4b-it-qat", "stash-ops-assistant"))
            self.assertEqual((agent_dir / "index.json").read_text(), before_index)
            blueprint = yaml.safe_load((agent_dir / "blueprint.yaml").read_text())
            self.assertEqual(blueprint["spec"]["provider"]["model"], "gemma4:e4b-it-qat")

    def test_documents_and_chunking_rebuild_index(self):
        with TempDir() as tmp:
            agent_dir = build_example(tmp)
            extra = tmp / "new-runbook.txt"
            extra.write_text("Rotate the Stash signing certificate every 90 days.", encoding="utf-8")
            spec = AgentSpec.load(agent_dir / "agent.yaml")
            result = update_agent(agent_dir, spec, add_docs=[extra], remove_docs=["faq.txt"])
            self.assertTrue(result.reindexed)
            self.assertIn("Removed faq.txt", result.changes)
            index = yaml.safe_load((agent_dir / "index.json").read_text())
            sources = {c["source"] for c in index["chunks"]}
            self.assertIn("new-runbook.txt", sources)
            self.assertNotIn("faq.txt", sources)

            smaller = spec.model_copy(update={"retrieval": spec.retrieval.model_copy(update={"chunk_size": 300})})
            self.assertTrue(update_agent(agent_dir, smaller).reindexed)

    def test_bad_test_set_changes_nothing(self):
        with TempDir() as tmp:
            agent_dir = build_example(tmp)
            spec = AgentSpec.load(agent_dir / "agent.yaml")
            renamed = spec.model_copy(update={"name": "Renamed"})
            with self.assertRaises(ValueError):
                update_agent(agent_dir, renamed, testset_yaml="cases: [ {question: }")
            self.assertEqual(AgentSpec.load(agent_dir / "agent.yaml").name, "Stash Ops Assistant")

    def test_cannot_remove_last_document(self):
        with TempDir() as tmp:
            agent_dir = build_example(tmp)
            spec = AgentSpec.load(agent_dir / "agent.yaml")
            names = [p.name for p in (agent_dir / "docs").iterdir()]
            with self.assertRaises(ValueError):
                update_agent(agent_dir, spec, remove_docs=names)
            self.assertEqual(len(list((agent_dir / "docs").iterdir())), 3)

    def test_choice_round_trip(self):
        for choice in ("azure_openai", "ollama", "openrouter", "openai", "anthropic", "extractive"):
            self.assertEqual(choice_for(provider_from_choice(choice, "m", "https://r.openai.azure.com"
                                                             if choice == "azure_openai" else "")), choice)
        custom = ProviderConfig(kind="openai_compatible", endpoint="http://localhost:1234/v1", model="m")
        self.assertEqual(choice_for(custom), "custom")


class EditWebTests(unittest.TestCase):
    def client(self, workspace):
        from starlette.testclient import TestClient

        from abe.web import create_app

        return TestClient(create_app(workspace))

    def test_edit_page_and_switch_to_ollama(self):
        with TempDir() as tmp:
            build_example(tmp)
            client = self.client(tmp)
            page = client.get("/agents/stash-ops-assistant/edit")
            self.assertEqual(page.status_code, 200)
            self.assertIn('value="gpt-4o-mini"', page.text)
            self.assertIn('<option value="azure_openai" selected>', page.text)
            self.assertIn("faq.txt", page.text)

            form = {
                "name": "Stash Ops Assistant",
                "provider": "ollama",
                "model": "llama3.1",
                "endpoint": "https://your-resource.openai.azure.com",
                "api_key_env": "AZURE_OPENAI_API_KEY",
                "original_provider": "azure_openai",
                "original_endpoint": "https://your-resource.openai.azure.com",
                "original_api_key_env": "AZURE_OPENAI_API_KEY",
                "require_citations": "on",
                "top_k": "6",
            }
            response = client.post("/agents/stash-ops-assistant/edit", data=form, follow_redirects=False)
            self.assertEqual(response.status_code, 303, response.text)
            spec = AgentSpec.load(tmp / "stash-ops-assistant" / "agent.yaml")
            self.assertEqual(spec.provider.model, "llama3.1")
            self.assertIn("/v1", spec.provider.endpoint)
            self.assertNotIn("azure", spec.provider.endpoint)
            self.assertEqual(spec.provider.api_key_env, "")
            self.assertEqual(spec.retrieval.top_k, 6)

            page = client.get(response.headers["location"])
            self.assertIn("Saved.", page.text)
            self.assertIn("llama3.1", page.text)

    def test_add_document_and_reject_bad_test_set(self):
        with TempDir() as tmp:
            build_example(tmp)
            client = self.client(tmp)
            base = {"name": "Stash Ops Assistant", "provider": "extractive", "original_provider": "azure_openai"}
            files = [("documents", ("certs.txt", io.BytesIO(b"Rotate certificates every 90 days."), "text/plain"))]
            current = (tmp / "stash-ops-assistant" / "evals" / "testset.yaml").read_text()
            crlf = {**base, "testset": current.replace("\n", "\r\n")}
            ok = client.post("/agents/stash-ops-assistant/edit", data=crlf, files=files, follow_redirects=False)
            self.assertEqual(ok.status_code, 303)
            page = client.get(ok.headers["location"])
            self.assertNotIn("Test set updated", page.text)
            self.assertIn("no model (extractive)", page.text)
            self.assertTrue((tmp / "stash-ops-assistant" / "docs" / "certs.txt").exists())
            answer = client.post("/agents/stash-ops-assistant/ask",
                                 data={"question": "How often are certificates rotated?"})
            self.assertIn("90 days", answer.text)

            bad = client.post("/agents/stash-ops-assistant/edit", data={**base, "testset": "cases: [ {"})
            self.assertEqual(bad.status_code, 400)
            self.assertIn("Not saved", bad.text)


class EffortAndStyleTests(unittest.TestCase):
    def test_prompt_lines_and_presets(self):
        from abe.spec import AnswerStyle

        spec = AgentSpec(name="X", style=AnswerStyle(steps=True, version_aware=True, general_knowledge=True))
        prompt = spec.with_effort("thorough").system_prompt()
        self.assertIn("numbered steps", prompt)
        self.assertIn("which one each part of the answer applies to", prompt)
        self.assertIn("(Not from the documents)", prompt)
        self.assertNotIn("Do not guess", prompt)
        self.assertIn("Check every passage", prompt)
        quick = spec.with_effort("quick")
        self.assertEqual((quick.retrieval.top_k, quick.provider.max_tokens), (3, 300))
        self.assertIn("one to three sentences", quick.system_prompt())
        self.assertNotIn("Check every passage", AgentSpec(name="X").system_prompt())

    def test_old_agent_files_still_load(self):
        with TempDir() as tmp:
            agent_dir = build_example(tmp)
            data = yaml.safe_load((agent_dir / "agent.yaml").read_text())
            del data["effort"]
            del data["style"]
            (agent_dir / "agent.yaml").write_text(yaml.safe_dump(data))
            spec = AgentSpec.load(agent_dir / "agent.yaml")
            self.assertEqual((spec.effort, spec.style.steps), ("standard", False))

    def test_web_build_and_edit(self):
        from starlette.testclient import TestClient

        from abe.web import create_app

        with TempDir() as tmp:
            build_example(tmp)
            client = TestClient(create_app(tmp))
            self.assertIn('name="style_version_aware"', client.get("/").text)
            form = {"name": "Stash Ops Assistant", "provider": "extractive", "original_provider": "azure_openai",
                    "effort": "thorough", "style_code_blocks": "on", "top_k": "4"}
            response = client.post("/agents/stash-ops-assistant/edit", data=form, follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            spec = AgentSpec.load(tmp / "stash-ops-assistant" / "agent.yaml")
            self.assertEqual((spec.effort, spec.retrieval.top_k, spec.style.code_blocks), ("thorough", 8, True))
            self.assertIn("code blocks", (tmp / "stash-ops-assistant" / "system_prompt.md").read_text())
            page = client.get(response.headers["location"])
            self.assertIn("Effort: standard → thorough", page.text)

            # Same effort again: a fine-tuned passage count is kept.
            form.update({"top_k": "6"})
            client.post("/agents/stash-ops-assistant/edit", data=form)
            self.assertEqual(AgentSpec.load(tmp / "stash-ops-assistant" / "agent.yaml").retrieval.top_k, 6)


class CustomPromptTests(unittest.TestCase):
    def client(self, workspace):
        from starlette.testclient import TestClient

        from abe.web import create_app

        return TestClient(create_app(workspace))

    def test_custom_rules_style_and_effort(self):
        with TempDir() as tmp:
            build_example(tmp)
            client = self.client(tmp)
            form = {"name": "Stash Ops Assistant", "provider": "extractive", "original_provider": "azure_openai",
                    "top_k": "5", "additional_rules": "Never quote prices.\r\n\r\nAnswer in English.",
                    "style_custom": "Use British spelling.", "effort": "custom",
                    "effort_instruction": "Think hard before answering."}
            response = client.post("/agents/stash-ops-assistant/edit", data=form, follow_redirects=False)
            self.assertEqual(response.status_code, 303, response.text)
            spec = AgentSpec.load(tmp / "stash-ops-assistant" / "agent.yaml")
            self.assertEqual(spec.additional_rules, ["Never quote prices.", "Answer in English."])
            self.assertEqual(spec.retrieval.top_k, 5)  # custom effort leaves passages alone
            prompt = spec.system_prompt()
            for line in ("- Never quote prices.", "- Use British spelling.", "- Think hard before answering."):
                self.assertIn(line, prompt)
            page = client.get("/agents/stash-ops-assistant/edit").text
            self.assertIn("Never quote prices.\nAnswer in English.</textarea>", page)

            missing = client.post("/agents/stash-ops-assistant/edit", data={**form, "effort_instruction": ""})
            self.assertEqual(missing.status_code, 400)

    def test_prompt_override(self):
        with TempDir() as tmp:
            build_example(tmp)
            client = self.client(tmp)
            form = {"name": "Stash Ops Assistant", "provider": "extractive", "original_provider": "azure_openai",
                    "prompt_override": "You are a terse assistant. Cite passages as [n]."}
            response = client.post("/agents/stash-ops-assistant/edit", data=form, follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            spec = AgentSpec.load(tmp / "stash-ops-assistant" / "agent.yaml")
            self.assertEqual(spec.system_prompt(), "You are a terse assistant. Cite passages as [n].")
            self.assertIn("You are Stash Ops Assistant", spec.generated_prompt())
            self.assertIn("terse", (tmp / "stash-ops-assistant" / "system_prompt.md").read_text())
            self.assertIn("System prompt written by hand", client.get(response.headers["location"]).text)

            client.post("/agents/stash-ops-assistant/edit", data={**form, "prompt_override": ""})
            spec = AgentSpec.load(tmp / "stash-ops-assistant" / "agent.yaml")
            self.assertIn("You are Stash Ops Assistant", spec.system_prompt())


class CliTests(unittest.TestCase):
    def test_set_model(self):
        from abe.cli import main

        with TempDir() as tmp:
            build_example(tmp)
            self.assertEqual(main(["--workspace", str(tmp), "set-model", "stash-ops-assistant",
                                   "gemma4:e4b-it-qat", "--provider", "ollama"]), 0)
            spec = AgentSpec.load(tmp / "stash-ops-assistant" / "agent.yaml")
            self.assertEqual((spec.provider.kind, spec.provider.model), ("openai_compatible", "gemma4:e4b-it-qat"))


if __name__ == "__main__":
    unittest.main()
