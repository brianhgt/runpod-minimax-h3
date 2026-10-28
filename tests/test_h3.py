import base64
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import handler  # noqa: E402
import manifest_tools  # noqa: E402

WF = ROOT / "workflows-api" / "h3" / "video_minimax_h3_r2v.json"
PNG = base64.b64encode(b"\x89PNG fake").decode()


def load():
    return json.loads(WF.read_text(encoding="utf-8"))


def fake_upload(name, data):
    return name


class ApplyInputsTest(unittest.TestCase):
    def test_full_mapping(self):
        p = load()
        warnings, seed = handler.apply_inputs(p, {
            "Input Text (Prompt)": "hello",
            "INPUT - Frame Count": 121,
            "Resolution Selector (Size)": {"aspect_ratio": "9:16 (Portrait Widescreen)", "megapixels": 0.4},
            "Seed": 42,
            "Reference Image": [PNG, PNG],
        }, fake_upload, "j")
        self.assertEqual(p["138"]["inputs"]["value"], "hello")
        self.assertEqual(p["115"]["inputs"]["aspect_ratio"], "9:16 (Portrait Widescreen)")
        self.assertEqual(p["115"]["inputs"]["multiple"], 32)
        self.assertEqual(p["129"]["inputs"]["noise_seed"], 42)
        self.assertEqual(seed, 42)
        self.assertEqual(p["137"]["inputs"]["image"], "j_ref0.png")
        self.assertEqual(p["139"]["inputs"]["image"], "j_ref1.png")
        self.assertEqual(len(warnings), 1)
        self.assertIn("INPUT - Frame Count", warnings[0])

    def test_unused_image_slot_is_pruned(self):
        p = load()
        handler.apply_inputs(p, {"Reference Image": PNG}, fake_upload)
        self.assertNotIn("139", p)
        self.assertNotIn("ref_images.ref_image_1", p["136"]["inputs"])
        self.assertIn("ref_images.ref_image_0", p["136"]["inputs"])

    def test_too_many_images(self):
        with self.assertRaises(handler.JobError):
            handler.apply_inputs(load(), {"Reference Image": [PNG] * 3}, fake_upload)

    def test_random_seed_when_omitted_or_minus_one(self):
        for inputs in ({}, {"Seed": -1}):
            p = load()
            _, seed = handler.apply_inputs(p, inputs, fake_upload)
            self.assertEqual(p["129"]["inputs"]["noise_seed"], seed)
            self.assertNotEqual(seed, 261662374822964)

    def test_validation(self):
        bad = [
            {"Input Text (Prompt)": ""},
            {"Input Text (Prompt)": 5},
            {"Resolution Selector (Size)": "16:9"},
            {"Resolution Selector (Size)": {"megapixels": -1}},
            {"Seed": "abc"},
            {"Seed": -5},
        ]
        for inputs in bad:
            with self.subTest(inputs=inputs), self.assertRaises(handler.JobError):
                handler.apply_inputs(load(), inputs, fake_upload)
        with self.assertRaises(handler.JobError):
            handler.apply_inputs(load(), ["x"], fake_upload)

    def test_frame_count_with_node(self):
        p = load()
        p["200"] = {"inputs": {"value": 5}, "class_type": "PrimitiveInt", "_meta": {"title": "INPUT - Frame Count"}}
        handler.apply_inputs(p, {"INPUT - Frame Count": "97"}, fake_upload)
        self.assertEqual(p["200"]["inputs"]["value"], 97)
        with self.assertRaises(handler.JobError):
            handler.apply_inputs(p, {"INPUT - Frame Count": 0}, fake_upload)

    def test_generic_title(self):
        p = load()
        handler.apply_inputs(p, {"Int (Full)": 30}, fake_upload)
        self.assertEqual(p["143"]["inputs"]["value"], 30)


class ProvisionSpecTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        handler.MANIFESTS_DIR = ROOT / "models"

    def tearDown(self):
        self.tmp.cleanup()

    def resolve(self, spec):
        return handler._resolve_manifest_spec(spec, self.tmp.name)

    def test_named_and_list(self):
        self.assertEqual(len(self.resolve("ref2va")), 1)
        self.assertEqual(len(self.resolve(["ref2va", "t2va"])), 2)
        self.assertEqual(len(self.resolve({"include": "ref2va"})), 1)

    def test_rejects_bad_names(self):
        for spec in ("../all", "nope", ["a/b"], {"x": 1}, 5, []):
            with self.subTest(spec=spec), self.assertRaises(handler.JobError):
                self.resolve(spec)

    def test_inline_rows(self):
        paths = self.resolve([["loras", "x.safetensors", "https://huggingface.co/a/b/resolve/main/x.safetensors"]])
        entries, errors = manifest_tools.load_manifests(paths)
        self.assertEqual((len(entries), errors), (1, []))


class ManifestToolsTest(unittest.TestCase):
    URL = "https://huggingface.co/a/b/resolve/main/x.safetensors"

    def parse(self, line):
        return manifest_tools.parse_manifest(line)

    def test_shipped_manifests_valid(self):
        entries, errors = manifest_tools.load_manifests(sorted(str(p) for p in (ROOT / "models").glob("*.txt")))
        self.assertEqual(errors, [])
        self.assertTrue(entries)

    def test_workflows_listed(self):
        args = type("A", (), {"dir": str(WF.parent), "manifests": [str(ROOT / "models" / "all.txt")]})
        self.assertEqual(manifest_tools.cmd_workflows(args), [])
        args.manifests = [str(ROOT / "models" / "t2va.txt")]
        self.assertTrue(manifest_tools.cmd_workflows(args))

    def test_every_workflow_covered_by_all(self):
        entries, _ = manifest_tools.load_manifests([ROOT / "models" / "all.txt"])
        have = {(e.subdir, e.name) for e in entries}
        for wf in WF.parent.glob("*.json"):
            for name, subdirs in manifest_tools.workflow_models(wf):
                self.assertTrue(any((s, name) in have for s in subdirs), f"{wf.name}: {name}")

    def test_subsets_cover_their_workflow(self):
        for subset, wf in (("ref2va", "video_minimax_h3_r2v.json"), ("i2va", "video_minimax_h3_i2v.json")):
            entries, _ = manifest_tools.load_manifests([ROOT / "models" / f"{subset}.txt"])
            have = {(e.subdir, e.name) for e in entries}
            for name, subdirs in manifest_tools.workflow_models(WF.parent / wf):
                self.assertTrue(any((s, name) in have for s in subdirs), f"{subset}: {name}")

    def test_bad_lines(self):
        sha = "a" * 64
        cases = {
            "too few": "vae\tx.safetensors",
            "bad subdir": f"nope\tx.safetensors\t{self.URL}",
            "bad name": f"vae\t../x\t{self.URL}",
            "http": "vae\tx.safetensors\thttp://huggingface.co/a/b",
            "bad host": "vae\tx.safetensors\thttps://evil.example/x",
            "bad sha": f"vae\tx.safetensors\t{self.URL}\tzz",
            "bad size": f"vae\tx.safetensors\t{self.URL}\t{sha}\t-5",
        }
        for label, line in cases.items():
            with self.subTest(label):
                entries, errors = self.parse(line)
                self.assertTrue(errors)
                self.assertFalse(entries)
        entries, errors = self.parse(f"vae\tx.safetensors\t{self.URL}\t-\t-")
        self.assertEqual((len(entries), errors), (1, []))

    def test_conflicting_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d, "a.txt"), Path(d, "b.txt")
            a.write_text(f"vae\tx.safetensors\t{self.URL}\n")
            b.write_text(f"vae\tx.safetensors\t{self.URL}\t{'b' * 64}\n")
            _, errors = manifest_tools.load_manifests([a, b])
            self.assertTrue(errors)

    def test_check_file(self):
        data = b"hello world"
        sha = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d, "vae", "x.safetensors")
            f.parent.mkdir()
            entry = manifest_tools.Entry("vae", "x.safetensors", self.URL, sha, len(data), "t", 1)
            self.assertEqual(manifest_tools.check_file(f, entry), "missing")
            f.write_bytes(b"")
            self.assertEqual(manifest_tools.check_file(f, entry), "empty file")
            f.write_bytes(data[:-1])
            self.assertIn("size", manifest_tools.check_file(f, entry))
            f.write_bytes(b"hello worle")
            self.assertIsNone(manifest_tools.check_file(f, entry))
            self.assertIn("sha256", manifest_tools.check_file(f, entry, with_hash=True))
            f.write_bytes(data)
            self.assertIsNone(manifest_tools.check_file(f, entry, with_hash=True))

    def test_verify_cli(self):
        data = b"hello world"
        with tempfile.TemporaryDirectory() as d:
            m = Path(d, "m.txt")
            m.write_text(f"vae\tx.safetensors\t{self.URL}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\n")
            Path(d, "vae").mkdir()
            self.assertEqual(manifest_tools.main(["verify", d, str(m)]), 1)
            Path(d, "vae", "x.safetensors").write_bytes(data)
            self.assertEqual(manifest_tools.main(["verify", d, str(m), "--hash"]), 0)


if __name__ == "__main__":
    unittest.main()
