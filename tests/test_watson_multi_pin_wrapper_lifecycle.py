from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import textwrap
import time
import unittest


ARENA_DIR = Path(__file__).resolve().parents[1]
WRAPPER = ARENA_DIR / "scripts/run_watson_multi_pin_air_replay.sh"


def write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    path.chmod(0o755)


def wait_for_path(path: Path, timeout_s: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


class WrapperLifecycleTests(unittest.TestCase):
    def make_wrapper(self, root: Path) -> Path:
        scripts = root / "scripts"
        scripts.mkdir()
        source = WRAPPER.read_text(encoding="utf-8")
        source = source.replace(
            'RUNNER="$SCRIPT_DIR/run_watson_multi_pin_air_replay.py"',
            'RUNNER="${WATSON_TEST_RUNNER:?}"',
        )
        source = source.replace(
            'carrier_file="/sys/class/net/$ROBOT_INTERFACE/carrier"',
            'carrier_file="${WATSON_TEST_CARRIER:?}"',
        )
        source = source.replace(
            "source_setup /opt/ros/jazzy/setup.bash",
            'source_setup "${WATSON_TEST_ROS_SETUP:?}"',
        )
        wrapper = scripts / WRAPPER.name
        wrapper.write_text(source, encoding="utf-8")
        wrapper.chmod(0o755)
        return wrapper

    def base_environment(self, root: Path) -> dict[str, str]:
        mock_bin = root / "mock-bin"
        mock_bin.mkdir()
        write_executable(
            mock_bin / "ip",
            """\
            #!/usr/bin/env bash
            echo "$4 dev enp1s0 src 192.0.2.100"
            """,
        )
        write_executable(
            mock_bin / "pgrep",
            """\
            #!/usr/bin/env bash
            exit 1
            """,
        )
        carrier = root / "carrier"
        carrier.write_text("1\n", encoding="utf-8")
        ros_setup = root / "ros_setup.bash"
        ros_setup.write_text(":\n", encoding="utf-8")
        tm_setup = root / "home/tm2_ws_apt/install/setup.bash"
        tm_setup.parent.mkdir(parents=True)
        tm_setup.write_text(":\n", encoding="utf-8")
        return {
            **os.environ,
            "HOME": str(root / "home"),
            "PATH": f"{mock_bin}:{os.environ['PATH']}",
            "WATSON_TEST_CARRIER": str(carrier),
            "WATSON_TEST_ROS_SETUP": str(ros_setup),
            "WATSON_TEST_EVENT_DIR": str(root / "events"),
        }

    def test_sigint_waits_for_runner_then_stops_orphaned_launch_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper = self.make_wrapper(root)
            environment = self.base_environment(root)
            events = root / "events"
            events.mkdir()
            mock_bin = root / "mock-bin"

            write_executable(
                mock_bin / "ros2",
                """\
                #!/usr/bin/python3
                import os
                from pathlib import Path
                import signal
                import sys
                import time

                events = Path(os.environ["WATSON_TEST_EVENT_DIR"])

                def record(name):
                    with (events / "order.log").open("a", encoding="utf-8") as stream:
                        stream.write(f"{time.monotonic_ns()} {name}\\n")

                if sys.argv[1:3] == ["node", "list"]:
                    if (events / "launch_leader.pid").exists():
                        print("/watson/tm_driver_node")
                        print("/watson/move_group")
                        print("/watson/robot_state_publisher")
                    raise SystemExit(0)

                if len(sys.argv) > 1 and sys.argv[1] == "launch":
                    child = os.fork()
                    if child == 0:
                        (events / "launch_child.pid").write_text(
                            str(os.getpid()), encoding="utf-8"
                        )
                        def stop(signum, _frame):
                            record(f"launch_child_signal_{signum}")
                            raise SystemExit(0)
                        for handled in (
                            signal.SIGINT,
                            signal.SIGTERM,
                            signal.SIGHUP,
                        ):
                            signal.signal(handled, stop)
                        while True:
                            time.sleep(0.05)
                    (events / "launch_leader.pid").write_text(
                        str(os.getpid()), encoding="utf-8"
                    )
                    # Leave enough time for the wrapper readiness probe even
                    # when the full suite is running under scheduler load.
                    time.sleep(2.0)
                    record("launch_leader_exit")
                    os._exit(0)

                raise SystemExit(3)
                """,
            )
            runner = root / "runner.py"
            write_executable(
                runner,
                """\
                #!/usr/bin/python3
                import os
                from pathlib import Path
                import signal
                import time

                events = Path(os.environ["WATSON_TEST_EVENT_DIR"])
                (events / "runner.pid").write_text(
                    str(os.getpid()), encoding="utf-8"
                )

                def record(name):
                    with (events / "order.log").open("a", encoding="utf-8") as stream:
                        stream.write(f"{time.monotonic_ns()} {name}\\n")

                def stop(signum, _frame):
                    record(f"runner_signal_{signum}")
                    time.sleep(0.20)
                    record("runner_guarded_exit")
                    raise SystemExit(0)

                for handled in (
                    signal.SIGINT,
                    signal.SIGTERM,
                    signal.SIGHUP,
                ):
                    signal.signal(handled, stop)
                while True:
                    time.sleep(0.05)
                """,
            )
            environment["WATSON_TEST_RUNNER"] = str(runner)

            process = subprocess.Popen(
                ["/usr/bin/bash", str(wrapper)],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            leader_pid = None
            runner_pid = None
            child_pid = None
            try:
                wait_for_path(events / "runner.pid")
                wait_for_path(events / "launch_leader.pid")
                wait_for_path(events / "launch_child.pid")
                runner_pid = int(
                    (events / "runner.pid").read_text(encoding="utf-8")
                )
                leader_pid = int(
                    (events / "launch_leader.pid").read_text(encoding="utf-8")
                )
                child_pid = int(
                    (events / "launch_child.pid").read_text(encoding="utf-8")
                )
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    order_path = events / "order.log"
                    order = (
                        order_path.read_text(encoding="utf-8")
                        if order_path.exists()
                        else ""
                    )
                    if "launch_leader_exit" in order:
                        break
                    time.sleep(0.02)
                else:
                    self.fail("mock launch leader did not exit")

                teardown_started = time.monotonic()
                os.kill(process.pid, signal.SIGINT)
                stdout, stderr = process.communicate(timeout=8.0)
                teardown_duration = time.monotonic() - teardown_started
                self.assertEqual(
                    process.returncode,
                    130,
                    msg=f"stdout={stdout}\nstderr={stderr}",
                )
                self.assertLess(teardown_duration, 3.0)
                self.assertNotIn("forcing cleanup", stderr)
                self.assertNotIn(
                    "guarded runner did not exit after SIGINT",
                    stderr,
                )
                order = [
                    line.split(" ", 1)[1]
                    for line in (events / "order.log")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertLess(
                    order.index(f"runner_signal_{signal.SIGINT}"),
                    order.index("runner_guarded_exit"),
                )
                self.assertLess(
                    order.index("runner_guarded_exit"),
                    order.index(f"launch_child_signal_{signal.SIGTERM}"),
                )
                with self.assertRaises(ProcessLookupError):
                    os.kill(runner_pid, 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                with self.assertRaises(ProcessLookupError):
                    os.killpg(leader_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2.0)
                for pid in (runner_pid, child_pid):
                    if pid is not None:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                if leader_pid is not None:
                    try:
                        os.killpg(leader_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_help_and_offline_routes_do_not_run_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper = self.make_wrapper(root)
            environment = self.base_environment(root)
            events = root / "events"
            events.mkdir()
            runner = root / "runner.py"
            write_executable(
                runner,
                """\
                #!/usr/bin/python3
                import os
                from pathlib import Path
                import sys

                events = Path(os.environ["WATSON_TEST_EVENT_DIR"])
                with (events / "runner_args.log").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(" ".join(sys.argv[1:]) + "\\n")
                """,
            )
            environment["WATSON_TEST_RUNNER"] = str(runner)
            environment["WATSON_TEST_CARRIER"] = str(root / "missing-carrier")
            environment["WATSON_TEST_ROS_SETUP"] = str(root / "missing-setup")

            for argument in ("--help", "--offline-validate"):
                with self.subTest(argument=argument):
                    completed = subprocess.run(
                        ["/usr/bin/bash", str(wrapper), argument],
                        cwd=root,
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=3.0,
                    )
                    self.assertEqual(
                        completed.returncode,
                        0,
                        msg=completed.stderr,
                    )
            self.assertEqual(
                (events / "runner_args.log")
                .read_text(encoding="utf-8")
                .splitlines(),
                ["--help", "--offline-validate"],
            )

    def test_runner_force_kill_follows_full_ninety_second_grace(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        attempts_match = re.search(
            r"^RUNNER_GUARDED_GRACE_ATTEMPTS=(\d+)$",
            source,
            flags=re.MULTILINE,
        )
        delay_match = re.search(
            r"^RUNNER_GUARDED_GRACE_DELAY_SECONDS=([0-9.]+)$",
            source,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(attempts_match)
        self.assertIsNotNone(delay_match)
        grace_seconds = int(attempts_match.group(1)) * float(
            delay_match.group(1)
        )
        self.assertGreaterEqual(grace_seconds, 90.0)

        stop_runner = source.split("stop_runner() {", 1)[1].split(
            "\n}\n\nstop_owned_stack()",
            1,
        )[0]
        self.assertIn('kill "-$first_signal" "$pid"', stop_runner)
        guarded_wait = (
            'wait_for_pid_exit \\\n'
            '      "$pid" \\\n'
            '      "$RUNNER_GUARDED_GRACE_ATTEMPTS"'
        )
        self.assertIn(guarded_wait, stop_runner)
        self.assertLess(
            stop_runner.index(guarded_wait),
            stop_runner.index('kill -KILL "$pid"'),
        )

    def test_execute_rejects_reused_stack_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper = self.make_wrapper(root)
            environment = self.base_environment(root)
            environment["WATSON_TEST_RUNNER"] = str(root / "missing-runner")
            environment["WATSON_TEST_CARRIER"] = str(root / "missing-carrier")
            completed = subprocess.run(
                [
                    "/usr/bin/bash",
                    str(wrapper),
                    "--use-existing-stack",
                    "--mode",
                    "execute",
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("execute mode must use the wrapper-owned", completed.stderr)

    def test_abbreviated_mode_is_rejected_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrapper = self.make_wrapper(root)
            environment = self.base_environment(root)
            environment["WATSON_TEST_RUNNER"] = str(root / "missing-runner")
            environment["WATSON_TEST_CARRIER"] = str(root / "missing-carrier")
            completed = subprocess.run(
                [
                    "/usr/bin/bash",
                    str(wrapper),
                    "--use-existing-stack",
                    "--mo",
                    "execute",
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "unknown or abbreviated option: --mo",
                completed.stderr,
            )
            self.assertNotIn("does not have carrier", completed.stderr)


if __name__ == "__main__":
    unittest.main()
