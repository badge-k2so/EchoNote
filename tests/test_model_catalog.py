import tempfile
import unittest
from pathlib import Path

from otoweave_app.model_catalog import ModelDisclosure, model_disclosures


class ModelDisclosureTests(unittest.TestCase):
    def test_all_keys_are_unique(self):
        keys = [model.key for model in model_disclosures(Path.cwd())]
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_ai_component_is_disclosed(self):
        keys = {model.key for model in model_disclosures(Path.cwd())}
        expected = {
            # モデル
            "qwen3.5-4b",
            "qwen3.5-9b",
            "qwen3.5-2b",
            "reazonspeech-k2-v2",
            "parakeet-tdt-v2-int8",
            "speechbrain-voxlingua107",
            "pyannote-segmentation-3.0",
            "3dspeaker-eres2net",
            "windows-tts-haruka",
            "macos-tts-kyoko",
            # 実行エンジン
            "sherpa-onnx",
            "llama-cpp-python",
            "onnxruntime",
        }
        self.assertTrue(expected.issubset(keys), expected - keys)

    def test_every_entry_has_license_and_source(self):
        for model in model_disclosures(Path.cwd()):
            with self.subTest(key=model.key):
                self.assertTrue(model.name.strip())
                self.assertTrue(model.purpose.strip())
                self.assertTrue(model.license_name.strip())
                self.assertTrue(model.source_url.strip())

    def test_9b_availability_follows_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()

            def find_9b() -> ModelDisclosure:
                return next(
                    m for m in model_disclosures(root) if m.key == "qwen3.5-9b"
                )

            self.assertFalse(find_9b().available)
            (root / "models" / "Qwen3.5-9B-Q4_K_M.gguf").write_bytes(b"x")
            self.assertTrue(find_9b().available)

    def test_status_labels(self):
        base = dict(
            key="k",
            name="n",
            purpose="p",
            license_name="l",
            source_url="s",
        )
        self.assertEqual(
            ModelDisclosure(**base, required=True, available=True).status,
            "利用可能",
        )
        self.assertEqual(
            ModelDisclosure(**base, required=True, available=False).status,
            "未配置",
        )
        self.assertEqual(
            ModelDisclosure(**base, required=False, available=False).status,
            "未配置（任意）",
        )


if __name__ == "__main__":
    unittest.main()
