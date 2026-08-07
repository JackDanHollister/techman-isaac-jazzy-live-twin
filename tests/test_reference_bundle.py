from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import yaml


REPO_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_DIR / "reference/seven_pin"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceBundleTests(unittest.TestCase):
    def test_config_hashes_match_bundled_reference_files(self) -> None:
        config = yaml.safe_load(
            (REPO_DIR / "config/isaac_multi_pin_verticalization.yaml").read_text(
                encoding="utf-8"
            )
        )
        plan = config["multi_pin_plan"]
        self.assertEqual(
            sha256_file(REPO_DIR / plan["path"]),
            plan["sha256"],
        )
        asset = config["articulated_asset"]
        for path_field, hash_field in (
            ("import_report", "import_report_sha256"),
            ("staged_manifest", "staged_manifest_sha256"),
            ("tool_metadata", "tool_metadata_sha256"),
        ):
            self.assertEqual(
                sha256_file(REPO_DIR / asset[path_field]),
                asset[hash_field],
            )

    def test_import_report_paths_are_repository_relative(self) -> None:
        report = json.loads(
            (REFERENCE_DIR / "isaac_import_report.json").read_text(
                encoding="utf-8"
            )
        )
        for field in ("source_urdf", "output_directory", "output_usd"):
            path = Path(report[field])
            self.assertFalse(path.is_absolute(), field)
            self.assertFalse(".." in path.parts, field)
        output_dir = REPO_DIR / report["output_directory"]
        for relative_path, evidence in report["asset_artifacts"].items():
            artifact = output_dir / relative_path
            self.assertTrue(artifact.is_file(), artifact)
            self.assertEqual(artifact.stat().st_size, evidence["size_bytes"])
            self.assertEqual(sha256_file(artifact), evidence["sha256"])

    def test_reference_metadata_has_no_workstation_identity(self) -> None:
        forbidden = (
            "/home/",
            "/tmp/",
            "157.140.",
            "192.168.",
        )
        offenders: list[str] = []
        for path in REFERENCE_DIR.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(value in text for value in forbidden):
                offenders.append(str(path.relative_to(REPO_DIR)))
        self.assertEqual(offenders, [])

    def test_binary_reference_assets_have_no_embedded_workstation_identity(
        self,
    ) -> None:
        forbidden = (
            b"/home/",
            b"/tmp/",
            b"157.140.",
            b"192.168.",
        )
        offenders: list[str] = []
        for path in REFERENCE_DIR.rglob("*"):
            if not path.is_file():
                continue
            data = path.read_bytes()
            if any(value in data for value in forbidden):
                offenders.append(str(path.relative_to(REPO_DIR)))
        self.assertEqual(offenders, [])

    def test_execution_fixture_hashes_match_staging_script(self) -> None:
        source = (
            REPO_DIR / "scripts/stage_reference_execution.sh"
        ).read_text(encoding="utf-8")
        expected = {
            "retimed_seven_pin_air_replay.json": (
                "8f24ba8c8cf6f814ba12f33e8202cf214b4fd89cd7d9017d11f75d075c5400fb"
            ),
            "tool_aware_ready_ingress.json": (
                "5c13f72b209781417448f48098c222077a5065809a05b7c39e46d898e713b018"
            ),
        }
        for name, digest in expected.items():
            self.assertIn(name, source)
            self.assertIn(digest, source)
            self.assertEqual(
                sha256_file(REFERENCE_DIR / "execution" / name),
                digest,
            )


if __name__ == "__main__":
    unittest.main()
