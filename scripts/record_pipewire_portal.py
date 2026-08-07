#!/usr/bin/env /usr/bin/python3
"""Record a GNOME Wayland screen stream through xdg-desktop-portal + PipeWire."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid

import dbus
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib


PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"
REQUEST_IFACE = "org.freedesktop.portal.Request"
SESSION_IFACE = "org.freedesktop.portal.Session"


def token(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def wait_for_response(bus: dbus.SessionBus, request_path: str, timeout_s: int) -> dict:
    loop = GLib.MainLoop()
    state: dict = {}

    def on_response(response: int, results: dict) -> None:
        state["response"] = int(response)
        state["results"] = dict(results)
        loop.quit()

    def on_timeout() -> bool:
        state["timeout"] = True
        loop.quit()
        return False

    bus.add_signal_receiver(
        on_response,
        signal_name="Response",
        dbus_interface=REQUEST_IFACE,
        path=request_path,
    )
    GLib.timeout_add_seconds(timeout_s, on_timeout)
    loop.run()

    if state.get("timeout"):
        raise RuntimeError(f"Timed out waiting for portal response: {request_path}")

    response = state.get("response")
    if response != 0:
        raise RuntimeError(f"Portal request failed or was cancelled: response={response}")

    return state["results"]


def create_portal_session(bus: dbus.SessionBus, timeout_s: int) -> tuple[dbus.proxies.Interface, str]:
    portal_object = bus.get_object(PORTAL_BUS, PORTAL_PATH)
    portal = dbus.Interface(portal_object, SCREENCAST_IFACE)

    request_path = portal.CreateSession(
        {
            "handle_token": dbus.String(token("create")),
            "session_handle_token": dbus.String(token("session")),
        }
    )
    results = wait_for_response(bus, request_path, timeout_s)
    return portal, str(results["session_handle"])


def select_sources(
    bus: dbus.SessionBus,
    portal: dbus.proxies.Interface,
    session_path: str,
    source_types: int,
    timeout_s: int,
) -> None:
    request_path = portal.SelectSources(
        dbus.ObjectPath(session_path),
        {
            "handle_token": dbus.String(token("select")),
            "types": dbus.UInt32(source_types),
            "multiple": dbus.Boolean(False),
            "cursor_mode": dbus.UInt32(2),
        },
    )
    wait_for_response(bus, request_path, timeout_s)


def start_stream(
    bus: dbus.SessionBus,
    portal: dbus.proxies.Interface,
    session_path: str,
    timeout_s: int,
) -> tuple[int, int]:
    request_path = portal.Start(
        dbus.ObjectPath(session_path),
        "",
        {"handle_token": dbus.String(token("start"))},
    )
    results = wait_for_response(bus, request_path, timeout_s)
    streams = results.get("streams")
    if not streams:
        raise RuntimeError("Portal did not return any PipeWire streams")

    node_id = int(streams[0][0])
    fd_obj = portal.OpenPipeWireRemote(dbus.ObjectPath(session_path), {})
    fd = fd_obj.take() if hasattr(fd_obj, "take") else int(fd_obj)
    return fd, node_id


def run_gstreamer(fd: int, node_id: int, output: str) -> subprocess.Popen:
    cmd = [
        "gst-launch-1.0",
        "-e",
        "pipewiresrc",
        f"fd={fd}",
        f"path={node_id}",
        "do-timestamp=true",
        "!",
        "queue",
        "!",
        "videoconvert",
        "!",
        "vp8enc",
        "deadline=1",
        "cpu-used=4",
        "!",
        "webmmux",
        "!",
        "filesink",
        f"location={output}",
    ]
    return subprocess.Popen(cmd, pass_fds=(fd,))


def close_session(bus: dbus.SessionBus, session_path: str) -> None:
    try:
        session_object = bus.get_object(PORTAL_BUS, session_path)
        dbus.Interface(session_object, SESSION_IFACE).Close()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Output WebM file")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument(
        "--source",
        choices=("monitor", "window", "both"),
        default="monitor",
        help="Portal source type to request",
    )
    args = parser.parse_args()

    source_types = {"monitor": 1, "window": 2, "both": 3}[args.source]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    try:
        os.unlink(args.output)
    except FileNotFoundError:
        pass

    DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()
    portal = None
    session_path = ""
    gst_proc = None
    stopping = False

    def stop(_signum=None, _frame=None) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        if gst_proc and gst_proc.poll() is None:
            gst_proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        portal, session_path = create_portal_session(bus, args.timeout_s)
        select_sources(bus, portal, session_path, source_types, args.timeout_s)
        fd, node_id = start_stream(bus, portal, session_path, args.timeout_s)
        print(f"PipeWire stream node: {node_id}", flush=True)
        print(f"Recording WebM: {args.output}", flush=True)

        gst_proc = run_gstreamer(fd, node_id, args.output)
        while gst_proc.poll() is None and not stopping:
            time.sleep(0.2)

        if gst_proc.poll() is None:
            gst_proc.send_signal(signal.SIGINT)
            try:
                gst_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gst_proc.terminate()
                gst_proc.wait(timeout=5)

        return int(gst_proc.returncode or 0)
    except Exception as exc:
        print(f"Portal recording failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if session_path:
            close_session(bus, session_path)


if __name__ == "__main__":
    raise SystemExit(main())
