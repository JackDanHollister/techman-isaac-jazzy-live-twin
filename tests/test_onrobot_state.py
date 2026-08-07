from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from pin_axis_3d_sim.onrobot_state import (
    LIVE_CONFIRMATION,
    QualificationError,
    build_qualification_report,
    canonical_digest,
    capture_live_read_only,
    decode_capture,
    normalise_2fg7_state,
    write_private_report,
)
from scripts.qualify_onrobot_2fg7_state import main as qualifier_main


NOW = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)


def healthy_device(**state_overrides):
    state = {
        "finger_orientation_outward": False,
        "min_external_width": 1.0,
        "max_external_width": 39.0,
        "current_external_width": 39.0,
        "min_internal_width": 11.0,
        "max_internal_width": 49.0,
        "current_internal_width": 49.0,
        "busy": False,
        "grip_detected": False,
        "status": 0,
        "not_calibrated": False,
        "linear_sensor_error": False,
    }
    state.update(state_overrides)
    return {
        "deviceId": 0,
        "deviceType": 17,
        "constants": {"productCode": 192},
        "variable": state,
    }


def healthy_capture(**state_overrides):
    return {
        "captured_at_utc": NOW.isoformat(),
        "devices": [healthy_device(**state_overrides)],
    }


class OnRobotStateParsingTests(unittest.TestCase):
    def test_normalises_nested_healthy_state(self) -> None:
        state = normalise_2fg7_state(healthy_capture())
        self.assertEqual(state["device_id"], 0)
        self.assertEqual(state["device_type"], 17)
        self.assertEqual(state["product_code"], 192)
        self.assertFalse(state["finger_orientation_outward"])
        self.assertEqual(state["external_width_mm"]["current"], 39.0)
        self.assertEqual(state["internal_width_mm"]["current"], 49.0)
        self.assertFalse(state["busy"])
        self.assertEqual(state["errors"]["status_code"], 0)

    def test_accepts_socketio_message_with_json_string_payload(self) -> None:
        packet = "42" + json.dumps(
            ["message", json.dumps(healthy_capture(), separators=(",", ":"))],
            separators=(",", ":"),
        )
        decoded = decode_capture(packet)
        self.assertEqual(normalise_2fg7_state(decoded)["device_id"], 0)

    def test_ignores_non_message_packets_but_requires_one_message(self) -> None:
        capture = healthy_capture()
        packet = "\x1e".join(
            (
                '0{"sid":"abc"}',
                "40",
                '42["mqtt_message",{"ignored":true}]',
                "42" + json.dumps(["message", capture], separators=(",", ":")),
            )
        )
        decoded = decode_capture(packet)
        self.assertEqual(normalise_2fg7_state(decoded)["product_code"], 192)
        with self.assertRaisesRegex(QualificationError, "no inbound"):
            decode_capture('42["mqtt_message",{}]')

    def test_rejects_ambiguous_2fg7_devices(self) -> None:
        capture = {
            "captured_at_utc": NOW.isoformat(),
            "devices": [healthy_device(), healthy_device()],
        }
        with self.assertRaisesRegex(QualificationError, "ambiguous"):
            normalise_2fg7_state(capture)

    def test_rejects_missing_required_state(self) -> None:
        capture = healthy_capture()
        del capture["devices"][0]["variable"]["busy"]
        with self.assertRaisesRegex(QualificationError, "missing required field busy"):
            normalise_2fg7_state(capture)

    def test_rejects_conflicting_error_representations(self) -> None:
        with self.assertRaisesRegex(QualificationError, "disagrees"):
            normalise_2fg7_state(
                healthy_capture(status=8, not_calibrated=False)
            )

    def test_derives_documented_error_bits_when_boolean_fields_absent(self) -> None:
        capture = healthy_capture(status=24)
        state_map = capture["devices"][0]["variable"]
        del state_map["not_calibrated"]
        del state_map["linear_sensor_error"]
        state = normalise_2fg7_state(capture)
        self.assertTrue(state["errors"]["not_calibrated"])
        self.assertTrue(state["errors"]["linear_sensor_error"])


class OnRobotQualificationTests(unittest.TestCase):
    def test_healthy_fresh_open_state_qualifies(self) -> None:
        report = build_qualification_report(
            healthy_capture(),
            mode="offline_capture",
            captured_at=None,
            now=NOW + timedelta(seconds=1),
            max_age_seconds=5.0,
        )
        self.assertEqual(report["status"], "qualified")
        self.assertTrue(report["ready_for_air_demo_sync"])
        self.assertEqual(report["failures"], [])
        self.assertFalse(report["safety_evidence"]["watson_contacted"])
        self.assertFalse(report["safety_evidence"]["gripper_commanded"])
        digest = report.pop("report_payload_sha256")
        self.assertEqual(digest, canonical_digest(report))

    def test_float32_open_endpoint_qualifies_with_readback_tolerance(
        self,
    ) -> None:
        report = build_qualification_report(
            healthy_capture(
                current_external_width=39.000003814697266,
                current_internal_width=49.000003814697266,
            ),
            mode="offline_capture",
            captured_at=None,
            now=NOW,
        )
        self.assertEqual(report["status"], "qualified")
        self.assertEqual(report["failures"], [])

    def test_material_open_endpoint_excursion_remains_blocked(self) -> None:
        report = build_qualification_report(
            healthy_capture(
                current_external_width=39.5001,
                current_internal_width=49.5001,
            ),
            mode="offline_capture",
            captured_at=None,
            now=NOW,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(
            any("advertised range" in item for item in report["failures"])
        )

    def test_stale_capture_fails_closed(self) -> None:
        report = build_qualification_report(
            healthy_capture(),
            mode="offline_capture",
            captured_at=None,
            now=NOW + timedelta(seconds=6),
            max_age_seconds=5.0,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("stale" in failure for failure in report["failures"]))

    def test_busy_error_or_unsynchronised_width_blocks(self) -> None:
        for overrides, expected in (
            ({"busy": True}, "busy"),
            ({"status": 16, "linear_sensor_error": True}, "error bits"),
            ({"current_external_width": 20.0}, "not synchronized"),
            ({"grip_detected": True}, "active grip"),
        ):
            with self.subTest(overrides=overrides):
                report = build_qualification_report(
                    healthy_capture(**overrides),
                    mode="offline_capture",
                    captured_at=None,
                    now=NOW,
                )
                self.assertEqual(report["status"], "blocked")
                self.assertTrue(
                    any(expected in failure for failure in report["failures"]),
                    report["failures"],
                )

    def test_low_three_operational_status_bits_are_not_errors(self) -> None:
        report = build_qualification_report(
            healthy_capture(status=7),
            mode="offline_capture",
            captured_at=None,
            now=NOW,
        )
        self.assertEqual(report["status"], "qualified")
        self.assertEqual(report["state"]["errors"]["operational_status_bits"], 7)
        self.assertEqual(report["state"]["errors"]["error_bits"], 0)

    def test_missing_trusted_timestamp_blocks(self) -> None:
        report = build_qualification_report(
            healthy_device(),
            mode="offline_capture",
            captured_at=None,
            now=NOW,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertIn("trusted envelope timestamp", report["failures"][0])


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


class FakeSocketOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "body": request.data,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("unexpected extra HTTP request")
        return FakeResponse(self.responses.pop(0).encode("utf-8"))


class OnRobotLiveReadOnlyTests(unittest.TestCase):
    def test_requires_exact_explicit_confirmation_before_network(self) -> None:
        opener = FakeSocketOpener([])
        with self.assertRaisesRegex(QualificationError, "exact confirmation"):
            capture_live_read_only(confirmation="", opener=opener)
        self.assertEqual(opener.requests, [])

    def test_only_connects_transport_and_receives_message(self) -> None:
        event = "40" + "\x1e" + "42" + json.dumps(
            ["message", healthy_capture()], separators=(",", ":")
        )
        opener = FakeSocketOpener(
            [
                '0{"sid":"safe_session","upgrades":[]}',
                "ok",
                event,
            ]
        )
        live = capture_live_read_only(
            confirmation=LIVE_CONFIRMATION,
            opener=opener,
            monotonic=lambda: 0.0,
        )
        self.assertEqual(normalise_2fg7_state(live.payload)["device_id"], 0)
        self.assertEqual(
            [request["method"] for request in opener.requests],
            ["GET", "POST", "GET"],
        )
        self.assertEqual(
            [request["body"] for request in opener.requests],
            [None, b"40", None],
        )
        for request in opener.requests:
            self.assertIn("http://192.0.2.1/socket.io/", request["url"])
            self.assertNotIn("/api/dc/", request["url"])
            self.assertNotIn("grip", request["url"].lower())
            self.assertNotIn("release", request["url"].lower())
            self.assertNotIn("stop", request["url"].lower())
        self.assertEqual(live.transport["application_events_emitted"], [])

    def test_engine_ping_gets_protocol_pong_not_application_event(self) -> None:
        event = "42" + json.dumps(
            ["message", healthy_capture()], separators=(",", ":")
        )
        opener = FakeSocketOpener(
            [
                '0{"sid":"safe_session","upgrades":[]}',
                "ok",
                "2",
                "ok",
                event,
            ]
        )
        live = capture_live_read_only(
            confirmation=LIVE_CONFIRMATION,
            opener=opener,
            monotonic=lambda: 0.0,
        )
        self.assertEqual(
            [request["body"] for request in opener.requests],
            [None, b"40", None, b"3", None],
        )
        self.assertEqual(live.transport["transport_protocol_bodies_sent"], ["40", "3"])
        self.assertEqual(live.transport["application_events_emitted"], [])


class OnRobotPrivateReportTests(unittest.TestCase):
    def test_private_report_is_mode_0600_and_never_overwritten(self) -> None:
        report = build_qualification_report(
            healthy_capture(),
            mode="offline_capture",
            captured_at=None,
            now=NOW,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "report.json"
            self.assertEqual(write_private_report(target, report), target)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            loaded = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(loaded["report_payload_sha256"], report["report_payload_sha256"])
            with self.assertRaises(FileExistsError):
                write_private_report(target, report)

    def test_refuses_existing_symlink(self) -> None:
        report = {"status": "blocked"}
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            destination = directory / "destination.json"
            destination.write_text("keep", encoding="utf-8")
            link = directory / "report.json"
            os.symlink(destination, link)
            with self.assertRaises(FileExistsError):
                write_private_report(link, report)
            self.assertTrue(link.is_symlink())
            self.assertEqual(destination.read_text(encoding="utf-8"), "keep")


class OnRobotQualifierCliTests(unittest.TestCase):
    def test_offline_mode_never_opens_live_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            capture = healthy_capture()
            capture["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
            capture_path = directory / "capture.json"
            capture_path.write_text(json.dumps(capture), encoding="utf-8")
            report_path = directory / "report.json"
            with mock.patch(
                "scripts.qualify_onrobot_2fg7_state.capture_live_read_only",
                side_effect=AssertionError("offline mode attempted network"),
            ), mock.patch("builtins.print"):
                result = qualifier_main(
                    [
                        "--capture",
                        str(capture_path),
                        "--report",
                        str(report_path),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(stat.S_IMODE(report_path.stat().st_mode), 0o600)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "offline_capture")
            self.assertFalse(report["transport"]["network_connection_opened"])
            self.assertFalse(report["safety_evidence"]["watson_contacted"])
            self.assertFalse(report["safety_evidence"]["gripper_commanded"])


if __name__ == "__main__":
    unittest.main()
