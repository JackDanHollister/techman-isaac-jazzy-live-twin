from __future__ import annotations

import copy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from scripts.prepare_watson_multi_pin_air_replay import (
    EXPECTED_PLAN_SHA256,
    EXPECTED_SAMPLE_SHA256,
    MANIFEST_STATUS,
    build_manifest,
    canonical_digest,
    resolve_plan,
    validate_plan,
    validate_live_check,
    write_private_manifest,
)


ARENA_DIR = Path(__file__).resolve().parents[1]
CONFIG = ARENA_DIR / "config/isaac_multi_pin_verticalization.yaml"
LIVE_CHECK_FIXTURE = (
    ARENA_DIR / "tests/fixtures/watson_pre_multi_pin_check.json"
)


class WatsonMultiPinAirReplayPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.live_check = Path(self.temporary_directory.name) / "live-check.json"
        self.live_check.write_bytes(LIVE_CHECK_FIXTURE.read_bytes())
        os.chmod(self.live_check, 0o600)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def live_time(self) -> datetime:
        report = json.loads(self.live_check.read_text(encoding="utf-8"))
        return datetime.fromisoformat(report["timestamp_utc"]) + timedelta(seconds=1)

    def private_json(self, directory: Path, name: str, payload: dict) -> Path:
        path = directory / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
        return path

    def test_shipped_live_check_builds_blocked_non_executable_intent(self) -> None:
        manifest = build_manifest(CONFIG, self.live_check, now=self.live_time())
        self.assertEqual(manifest["status"], MANIFEST_STATUS)
        self.assertEqual(manifest["source_plan_sha256"], EXPECTED_PLAN_SHA256)
        self.assertEqual(
            manifest["source_numeric_sample_sha256"], EXPECTED_SAMPLE_SHA256
        )
        self.assertEqual(manifest["stage_count"], 49)
        self.assertEqual(len(manifest["reviewed_isaac_stage_intents"]), 49)
        self.assertEqual(
            manifest["source_plan_metrics"]["control_sample_count"], 18102
        )
        self.assertAlmostEqual(
            manifest["maximum_live_to_ready_delta_rad"],
            0.4779895339900244,
            places=14,
        )
        self.assertEqual(
            manifest["ingress_intent"]["status"],
            "unplanned_not_a_controller_trajectory",
        )
        self.assertEqual(
            manifest["egress_intent"]["status"],
            "unplanned_not_a_controller_trajectory",
        )
        for field in (
            "commands_gripper",
            "controller_trajectory_created",
            "command_path_created",
            "ros_used",
            "watson_connected",
            "network_connection_opened",
            "real_robot_commanded",
            "motion_commanded",
            "execution_authorized",
            "arm_token_accepted",
        ):
            self.assertIs(manifest[field], False, field)
        self.assertEqual(manifest["report_payload_sha256"], canonical_digest(manifest))
        self.assertTrue(
            any("live-start" in blocker for blocker in manifest["blockers"])
        )
        self.assertTrue(
            any(">=25 ms" in blocker for blocker in manifest["blockers"])
        )

    def test_private_writer_refuses_overwrite(self) -> None:
        manifest = build_manifest(CONFIG, self.live_check, now=self.live_time())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intent.json"
            write_private_manifest(path, manifest)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["report_payload_sha256"], canonical_digest(stored))
            with self.assertRaises(FileExistsError):
                write_private_manifest(path, manifest)

    def test_rejects_non_private_live_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy_path = Path(directory) / "check.json"
            copy_path.write_bytes(self.live_check.read_bytes())
            os.chmod(copy_path, 0o644)
            with self.assertRaisesRegex(ValueError, "private mode 0600"):
                validate_live_check(copy_path, now=self.live_time())

    def test_rejects_symlinked_or_stale_live_evidence(self) -> None:
        original = json.loads(self.live_check.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(original["timestamp_utc"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = self.private_json(root, "real.json", original)
            symlink = root / "link.json"
            symlink.symlink_to(real)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                validate_live_check(symlink, now=timestamp)
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_live_check(real, now=timestamp + timedelta(minutes=31))

    def test_rejects_execute_or_motion_evidence(self) -> None:
        original = json.loads(self.live_check.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execute = copy.deepcopy(original)
            execute["mode"] = "execute"
            execute["report_payload_sha256"] = canonical_digest(execute)
            with self.assertRaisesRegex(ValueError, "read-only check"):
                validate_live_check(
                    self.private_json(root, "execute.json", execute),
                    now=self.live_time(),
                )

            moved = copy.deepcopy(original)
            moved["motion_commanded"] = True
            moved["report_payload_sha256"] = canonical_digest(moved)
            with self.assertRaisesRegex(ValueError, "motion_commanded false"):
                validate_live_check(
                    self.private_json(root, "moved.json", moved),
                    now=self.live_time(),
                )

    def test_rejects_promoted_or_digest_tampered_live_evidence(self) -> None:
        original = json.loads(self.live_check.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            promoted = copy.deepcopy(original)
            promoted["controller_tool_settings_promotion_passed"] = True
            promoted["controller_tool_audit"]["promotion_passed"] = True
            promoted["report_payload_sha256"] = canonical_digest(promoted)
            with self.assertRaisesRegex(ValueError, "uncommissioned tool"):
                validate_live_check(
                    self.private_json(root, "promoted.json", promoted),
                    now=self.live_time(),
                )

            tampered = copy.deepcopy(original)
            tampered["stable_health"]["project_speed"] = 100
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_live_check(
                    self.private_json(root, "tampered.json", tampered),
                    now=self.live_time(),
                )

    def test_rejects_failed_plan_acceptance_or_clearance_gates(self) -> None:
        _, _, original = resolve_plan(CONFIG)
        failed = copy.deepcopy(original)
        failed["specimens"][0]["stages"][0]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "failed stage gate"):
            validate_plan(failed)

        unsafe = copy.deepcopy(original)
        unsafe["validation"]["minimum_sampled_sphere_clearance_m"] = 0.003
        with self.assertRaisesRegex(ValueError, "top-level"):
            validate_plan(unsafe)


if __name__ == "__main__":
    unittest.main()
