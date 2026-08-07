from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
from urllib.request import ProxyHandler

from pin_axis_3d_sim.onrobot_control import (
    AIR_REPLAY_CLOSE_EXTERNAL_WIDTH_MM,
    ALLOWED_COMMAND_URLS,
    COMPUTE_BOX_ORIGIN,
    CONTROL_CONFIRMATION,
    CommandResponse,
    ControlTiming,
    FORCE_N,
    FixedComputeBoxTransport,
    GripperAction,
    OPEN_EXTERNAL_WIDTH_MM,
    SPEED_PERCENT,
    command_spec,
    run_fixed_recovery_stop,
    run_guarded_command,
)
from pin_axis_3d_sim.onrobot_state import (
    LiveCapture,
    QualificationError,
    canonical_digest,
)


NOW = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)
FAST_TIMING = ControlTiming(
    state_capture_timeout_seconds=0.1,
    command_timeout_seconds=0.1,
    poll_timeout_seconds=0.2,
    poll_interval_seconds=0.1,
    max_state_age_seconds=1.0,
)


def state_payload(
    external_width: float,
    *,
    busy: bool = False,
    grip_detected: bool = False,
    status: int = 0,
    device_id: int = 0,
    device_type: int = 17,
    product_code: int = 192,
    outward: bool = False,
) -> dict:
    return {
        "devices": [
            {
                "deviceId": device_id,
                "deviceType": device_type,
                "constants": {"productCode": product_code},
                "variable": {
                    "finger_orientation_outward": outward,
                    "min_external_width": 1.0,
                    "max_external_width": 39.0,
                    "current_external_width": external_width,
                    "min_internal_width": 11.0,
                    "max_internal_width": 49.0,
                    "current_internal_width": external_width + 10.0,
                    "busy": busy,
                    "grip_detected": grip_detected,
                    "status": status,
                    "not_calibrated": bool(status & 8),
                    "linear_sensor_error": bool(status & 16),
                },
            }
        ]
    }


def capture(external_width: float, **kwargs) -> LiveCapture:
    return LiveCapture(
        payload=state_payload(external_width, **kwargs),
        received_at_utc=NOW,
        transport={"network_connection_opened": True},
    )


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeTransport:
    def __init__(
        self,
        states: list[LiveCapture],
        *,
        responses: dict[str, CommandResponse] | None = None,
    ) -> None:
        self.states = list(states)
        self.responses = responses or {}
        self.calls: list[tuple[str, str | float]] = []

    def passive_state(self, *, timeout_seconds: float) -> LiveCapture:
        self.calls.append(("state", timeout_seconds))
        if not self.states:
            raise AssertionError("unexpected passive-state call")
        return self.states.pop(0)

    def command_get(
        self,
        *,
        url: str,
        timeout_seconds: float,
    ) -> CommandResponse:
        self.calls.append(("command", url))
        if url not in self.responses:
            raise AssertionError(f"unexpected command URL: {url}")
        return self.responses[url]


def success_response(action: GripperAction) -> CommandResponse:
    spec = command_spec(action)
    return CommandResponse(status_code=200, body="0", final_url=spec.url)


class OnRobotFixedProfileTests(unittest.TestCase):
    def test_fixed_profile_matches_verified_watson_commands(self) -> None:
        self.assertEqual(COMPUTE_BOX_ORIGIN, "http://192.0.2.1")
        self.assertEqual(OPEN_EXTERNAL_WIDTH_MM, 39.0)
        self.assertEqual(AIR_REPLAY_CLOSE_EXTERNAL_WIDTH_MM, 1.0)
        self.assertEqual(FORCE_N, 20)
        self.assertEqual(SPEED_PERCENT, 10)
        self.assertEqual(
            command_spec("open").url,
            "http://192.0.2.1/api/dc/twofg/grip_external/0/39/20/10",
        )
        self.assertEqual(
            command_spec("close").url,
            "http://192.0.2.1/api/dc/twofg/grip_external/0/1/20/10",
        )
        self.assertEqual(
            command_spec("stop").url,
            "http://192.0.2.1/api/dc/twofg/stop/0",
        )
        self.assertEqual(len(ALLOWED_COMMAND_URLS), 3)

    def test_dry_run_cannot_touch_injected_transport(self) -> None:
        transport = FakeTransport([])
        report = run_guarded_command("close", transport=transport)
        self.assertEqual(report["status"], "dry_run")
        self.assertFalse(report["completed"])
        self.assertFalse(report["transport_evidence"]["network_contacted"])
        self.assertEqual(transport.calls, [])

    def test_no_call_without_exact_confirmation(self) -> None:
        for token in (None, "", CONTROL_CONFIRMATION.lower(), CONTROL_CONFIRMATION + " "):
            with self.subTest(token=token):
                transport = FakeTransport([])
                report = run_guarded_command(
                    "close",
                    execute=True,
                    confirmation=token,
                    transport=transport,
                )
                self.assertEqual(report["status"], "blocked")
                self.assertFalse(report["authorization"]["accepted"])
                self.assertEqual(transport.calls, [])

    def test_wrong_identity_or_orientation_blocks_before_command(self) -> None:
        variants = (
            {"device_id": 1},
            {"device_type": 18},
            {"product_code": 193},
            {"outward": True},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                transport = FakeTransport([capture(39.0, **variant)])
                report = run_guarded_command(
                    "close",
                    execute=True,
                    confirmation=CONTROL_CONFIRMATION,
                    transport=transport,
                    wall_clock=lambda: NOW,
                )
                self.assertEqual(report["status"], "blocked")
                self.assertEqual(
                    [call[0] for call in transport.calls],
                    ["state"],
                )


class FakeHttpResponse(io.BytesIO):
    status = 200

    def __init__(self, body: bytes, url: str) -> None:
        super().__init__(body)
        self._url = url

    def geturl(self) -> str:
        return self._url


class RecordingOpener:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, request, timeout):
        self.calls.append(request.full_url)
        return FakeHttpResponse(b"0", request.full_url)


class OnRobotUrlAllowlistTests(unittest.TestCase):
    def test_default_transport_explicitly_disables_environment_proxies(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"http_proxy": "http://127.0.0.1:9"},
        ):
            transport = FixedComputeBoxTransport()
        opener = transport._opener.__self__
        proxy_handlers = [
            handler
            for handler in opener.handlers
            if isinstance(handler, ProxyHandler)
        ]
        self.assertEqual(proxy_handlers, [])

    def test_transport_rejects_arbitrary_host_device_force_speed_and_path(self) -> None:
        opener = RecordingOpener()
        transport = FixedComputeBoxTransport(opener=opener)
        variants = (
            "http://example.com/api/dc/twofg/stop/0",
            "http://192.0.2.1/api/dc/twofg/stop/1",
            "http://192.0.2.1/api/dc/twofg/grip_external/0/1/21/10",
            "http://192.0.2.1/api/dc/twofg/grip_external/0/1/20/11",
            "http://192.0.2.1/api/dc/twofg/grip_internal/0/1/20/10",
            "http://192.0.2.1/api/dc/twofg/set_finger_orientation/0/true",
        )
        for url in variants:
            with self.subTest(url=url), self.assertRaises(QualificationError):
                transport.command_get(url=url, timeout_seconds=1.0)
        self.assertEqual(opener.calls, [])

    def test_allowlisted_get_uses_exact_url(self) -> None:
        opener = RecordingOpener()
        transport = FixedComputeBoxTransport(opener=opener)
        spec = command_spec("stop")
        response = transport.command_get(url=spec.url, timeout_seconds=1.0)
        self.assertEqual(response.body, "0")
        self.assertEqual(opener.calls, [spec.url])


class OnRobotCommandFlowTests(unittest.TestCase):
    def test_dispatch_guard_contains_final_abort_check_and_actuator_get(self) -> None:
        close = command_spec("close")
        events = []

        class OrderedTransport(FakeTransport):
            def command_get(self, *, url: str, timeout_seconds: float):
                events.append(("command", url))
                return super().command_get(
                    url=url,
                    timeout_seconds=timeout_seconds,
                )

        @contextmanager
        def dispatch_guard(label: str):
            events.append(("enter", label))
            try:
                yield
            finally:
                events.append(("exit", label))

        transport = OrderedTransport(
            [capture(39.0), capture(1.0)],
            responses={
                close.url: success_response(GripperAction.CLOSE),
            },
        )
        report = run_guarded_command(
            "close",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            wall_clock=lambda: NOW,
            abort_requested=lambda: False,
            dispatch_guard=dispatch_guard,
        )
        self.assertTrue(report["completed"])
        self.assertEqual(
            events,
            [
                ("enter", "2fg7_close"),
                ("command", close.url),
                ("exit", "2fg7_close"),
            ],
        )

    def test_abort_during_prestate_blocks_before_actuator_get(self) -> None:
        stop_requested = False

        class AbortAfterStateTransport(FakeTransport):
            def passive_state(self, *, timeout_seconds: float) -> LiveCapture:
                nonlocal stop_requested
                result = super().passive_state(timeout_seconds=timeout_seconds)
                stop_requested = True
                return result

        transport = AbortAfterStateTransport([capture(39.0)])
        report = run_guarded_command(
            "close",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            wall_clock=lambda: NOW,
            abort_requested=lambda: stop_requested,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["safety_evidence"]["gripper_commanded"])
        self.assertEqual(
            [call for call in transport.calls if call[0] == "command"],
            [],
        )

    def test_abort_after_actuator_get_sends_fixed_recovery_stop(self) -> None:
        clock = FakeClock()
        close = command_spec("close")
        stop = command_spec("stop")
        stop_requested = False

        class AbortAfterCommandTransport(FakeTransport):
            def command_get(self, *, url: str, timeout_seconds: float):
                nonlocal stop_requested
                result = super().command_get(
                    url=url,
                    timeout_seconds=timeout_seconds,
                )
                if url == close.url:
                    stop_requested = True
                return result

        transport = AbortAfterCommandTransport(
            [capture(39.0), capture(25.0)],
            responses={
                close.url: success_response(GripperAction.CLOSE),
                stop.url: success_response(GripperAction.STOP),
            },
        )
        report = run_guarded_command(
            "close",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: NOW,
            abort_requested=lambda: stop_requested,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["recovery_stop"]["attempted"])
        commands = [call[1] for call in transport.calls if call[0] == "command"]
        self.assertEqual(commands, [close.url, stop.url])

    def test_recovery_stop_does_not_require_passive_prestate(self) -> None:
        stop = command_spec("stop")

        class BrokenStateTransport(FakeTransport):
            def passive_state(self, *, timeout_seconds: float) -> LiveCapture:
                self.calls.append(("state", timeout_seconds))
                raise OSError("state channel unavailable")

        transport = BrokenStateTransport(
            [],
            responses={stop.url: success_response(GripperAction.STOP)},
        )
        report = run_fixed_recovery_stop(
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            wall_clock=lambda: NOW,
        )
        self.assertFalse(report["completed"])
        self.assertEqual(
            [call for call in transport.calls if call[0] == "command"],
            [("command", stop.url)],
        )
        self.assertEqual(
            report["status"],
            "stop_sent_or_attempted_but_unverified",
        )

    def test_recovery_stop_wrong_token_touches_no_transport(self) -> None:
        transport = FakeTransport([])
        report = run_fixed_recovery_stop(
            confirmation="wrong",
            transport=transport,
            wall_clock=lambda: NOW,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(transport.calls, [])

    def test_correct_close_flow(self) -> None:
        clock = FakeClock()
        close = command_spec("close")
        transport = FakeTransport(
            [
                capture(39.000003814697266),
                capture(25.0, busy=True),
                capture(0.9999999403953552),
            ],
            responses={close.url: success_response(GripperAction.CLOSE)},
        )
        report = run_guarded_command(
            "close",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: NOW,
        )
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["completed"])
        self.assertEqual(report["poll_count"], 2)
        self.assertEqual(
            report["state_after"]["state"]["external_width_mm"]["current"],
            0.9999999403953552,
        )
        self.assertEqual(
            [call for call in transport.calls if call[0] == "command"],
            [("command", close.url)],
        )

    def test_correct_open_flow(self) -> None:
        clock = FakeClock()
        opened = command_spec("open")
        transport = FakeTransport(
            [
                capture(1.0),
                capture(28.0, busy=True),
                capture(39.000003814697266),
            ],
            responses={opened.url: success_response(GripperAction.OPEN)},
        )
        report = run_guarded_command(
            "open",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: NOW,
        )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(
            report["state_after"]["state"]["external_width_mm"]["current"],
            39.000003814697266,
        )

    def test_close_accepts_near_closed_fingertip_contact(self) -> None:
        close = command_spec("close")
        transport = FakeTransport(
            [
                capture(38.5),
                capture(1.7, grip_detected=True),
            ],
            responses={close.url: success_response(GripperAction.CLOSE)},
        )
        report = run_guarded_command(
            "close",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            wall_clock=lambda: NOW,
        )
        self.assertTrue(report["completed"])
        self.assertTrue(
            report["state_after"]["state"]["grip_detected"]
        )
        self.assertEqual(
            report["state_after"]["state"]["external_width_mm"]["current"],
            1.7,
        )

    def test_open_accepts_grip_latched_prestate_and_clears_it(self) -> None:
        opened = command_spec("open")
        transport = FakeTransport(
            [
                capture(1.7, grip_detected=True),
                capture(20.0, busy=True, grip_detected=True),
                capture(38.5),
            ],
            responses={opened.url: success_response(GripperAction.OPEN)},
        )
        report = run_guarded_command(
            "open",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            wall_clock=lambda: NOW,
        )
        self.assertTrue(report["completed"])
        self.assertFalse(
            report["state_after"]["state"]["grip_detected"]
        )

    def test_material_endpoint_excursion_still_fails_and_stops(self) -> None:
        clock = FakeClock()
        opened = command_spec("open")
        stop = command_spec("stop")
        transport = FakeTransport(
            [
                capture(4.4),
                capture(39.5001),
                capture(39.0),
            ],
            responses={
                opened.url: success_response(GripperAction.OPEN),
                stop.url: success_response(GripperAction.STOP),
            },
        )
        report = run_guarded_command(
            "open",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: NOW,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertIn("39.500100mm", report["failures"][0])
        self.assertEqual(report["recovery_stop"]["failures"], [])
        commands = [call[1] for call in transport.calls if call[0] == "command"]
        self.assertEqual(commands, [opened.url, stop.url])

    def test_explicit_stop_accepts_busy_prestate_and_proves_stationary(self) -> None:
        clock = FakeClock()
        stopped = command_spec("stop")
        transport = FakeTransport(
            [
                capture(39.000003814697266, busy=True),
                capture(39.000003814697266),
            ],
            responses={stopped.url: success_response(GripperAction.STOP)},
        )
        report = run_guarded_command(
            "stop",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: NOW,
        )
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["state_after"]["state"]["busy"])

    def test_recovery_stop_accepts_stationary_latched_grip(self) -> None:
        stop = command_spec("stop")
        transport = FakeTransport(
            [capture(1.7, grip_detected=True)],
            responses={stop.url: success_response(GripperAction.STOP)},
        )
        report = run_fixed_recovery_stop(
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            wall_clock=lambda: NOW,
        )
        self.assertTrue(report["completed"])
        self.assertTrue(
            report["state_after"]["state"]["grip_detected"]
        )

    def test_timeout_sends_fixed_stop_and_records_recovery(self) -> None:
        clock = FakeClock()
        close = command_spec("close")
        stop = command_spec("stop")
        transport = FakeTransport(
            [
                capture(39.0),
                capture(25.0, busy=True),
                capture(25.0, busy=True),
                capture(25.0),
            ],
            responses={
                close.url: success_response(GripperAction.CLOSE),
                stop.url: success_response(GripperAction.STOP),
            },
        )
        report = run_guarded_command(
            "close",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: NOW,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["recovery_stop"]["attempted"])
        self.assertEqual(report["recovery_stop"]["failures"], [])
        commands = [call[1] for call in transport.calls if call[0] == "command"]
        self.assertEqual(commands, [close.url, stop.url])

    def test_error_state_sends_fixed_stop(self) -> None:
        clock = FakeClock()
        close = command_spec("close")
        stop = command_spec("stop")
        transport = FakeTransport(
            [
                capture(39.0),
                capture(20.0, status=16),
                capture(20.0),
            ],
            responses={
                close.url: success_response(GripperAction.CLOSE),
                stop.url: success_response(GripperAction.STOP),
            },
        )
        report = run_guarded_command(
            "close",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: NOW,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(
            any("error bits" in failure for failure in report["failures"])
        )
        commands = [call[1] for call in transport.calls if call[0] == "command"]
        self.assertEqual(commands, [close.url, stop.url])

    def test_unexpected_air_grip_sends_fixed_stop(self) -> None:
        clock = FakeClock()
        close = command_spec("close")
        stop = command_spec("stop")
        transport = FakeTransport(
            [
                capture(39.0),
                capture(18.0, grip_detected=True),
                capture(18.0),
            ],
            responses={
                close.url: success_response(GripperAction.CLOSE),
                stop.url: success_response(GripperAction.STOP),
            },
        )
        report = run_guarded_command(
            "close",
            execute=True,
            confirmation=CONTROL_CONFIRMATION,
            transport=transport,
            timing=FAST_TIMING,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: NOW,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["recovery_stop"]["attempted"])
        self.assertTrue(
            any("above the fixed 2mm" in item for item in report["failures"])
        )


class OnRobotControlReportTests(unittest.TestCase):
    def test_private_report_is_0600_digest_bound_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "control.json"
            report = run_guarded_command("close", report_path=report_path)
            self.assertEqual(stat.S_IMODE(report_path.stat().st_mode), 0o600)
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            digest = loaded.pop("report_payload_sha256")
            self.assertEqual(digest, canonical_digest(loaded))
            self.assertEqual(report["status"], "dry_run")
            with self.assertRaises(FileExistsError):
                run_guarded_command("close", report_path=report_path)


if __name__ == "__main__":
    unittest.main()
