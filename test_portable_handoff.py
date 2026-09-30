"""交接检查：随仓库模型、CPU/不可用CUDA回退、安装模式和Windows命令文件字节。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analyze_single_fly_video import resolve_device
from setup_env import install_plan

ROOT = Path(__file__).resolve().parent


class HandoffTests(unittest.TestCase):
    def test_bundled_model_matches_manifest(self):
        manifest = json.loads((ROOT / "assets/model_info.json").read_text(encoding="utf-8"))
        model = ROOT / "assets" / manifest["filename"]
        self.assertEqual(model.stat().st_size, manifest["size_bytes"])
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), manifest["sha256"])
        self.assertEqual(manifest["classes"], {"0": "fly"})

    def test_auto_device_on_cpu_only_computer(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
        self.assertEqual(resolve_device("auto", fake), "cpu")
        self.assertEqual(resolve_device("cpu", fake), "cpu")
        with self.assertRaises(RuntimeError):
            resolve_device("0", fake)

    def test_unusable_cuda_does_not_block_auto_cpu(self):
        def failed_kernel(*args, **kwargs):
            raise RuntimeError("driver or CUDA architecture not supported")

        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1), ones=failed_kernel)
        self.assertEqual(resolve_device("auto", fake), "cpu")
        with self.assertRaises(RuntimeError):
            resolve_device("0", fake)

    def test_cuda_reported_available_but_no_visible_device(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 0))
        self.assertEqual(resolve_device("auto", fake), "cpu")

    def test_no_nvidia_installs_cpu_build(self):
        with patch("setup_env.shutil.which", return_value=None):
            self.assertEqual(install_plan("auto"), "cpu")
        self.assertEqual(install_plan("cpu"), "cpu")
        self.assertEqual(install_plan("cuda"), "cuda")

    def test_cmd_files_keep_windows_line_endings(self):
        for name in ["run_single_fly.cmd", "setup_env.cmd", "check_install.cmd"]:
            data = (ROOT / name).read_bytes()
            self.assertGreater(data.count(b"\r\n"), 0)
            self.assertNotIn(b"\n", data.replace(b"\r\n", b""))
        self.assertIn("*.cmd -text", (ROOT / ".gitattributes").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
