from __future__ import annotations

import importlib
from types import ModuleType
import sys
import unittest
from unittest.mock import patch

from pin_axis_3d_sim import isaac_hil_timeline


class FakeTimeline:
    def __init__(self) -> None:
        self.event_key = {"name": ""}
        self.event_key_calls = 0

    def get_event_key(self):
        self.event_key_calls += 1
        return self.event_key


class FakeDispatcher:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def observe_event(self, **kwargs):
        handle = object()
        self.calls.append({**kwargs, "handle": handle})
        return handle


def fake_isaac_modules(dispatcher: FakeDispatcher) -> dict[str, ModuleType]:
    omni_module = ModuleType("omni")
    omni_module.__path__ = []
    timeline_module = ModuleType("omni.timeline")
    timeline_module.GLOBAL_EVENT_PLAY = "omni.timeline:timeline:play"
    timeline_module.GLOBAL_EVENT_PAUSE = "omni.timeline:timeline:pause"
    timeline_module.GLOBAL_EVENT_STOP = "omni.timeline:timeline:stop"
    omni_module.timeline = timeline_module

    carb_module = ModuleType("carb")
    carb_module.__path__ = []
    dispatcher_module = ModuleType("carb.eventdispatcher")
    dispatcher_module.get_eventdispatcher = lambda: dispatcher
    carb_module.eventdispatcher = dispatcher_module
    return {
        "omni": omni_module,
        "omni.timeline": timeline_module,
        "carb": carb_module,
        "carb.eventdispatcher": dispatcher_module,
    }


class IsaacHilTimelineTests(unittest.TestCase):
    def test_module_import_does_not_require_isaac(self) -> None:
        blocked_modules = {
            "omni": None,
            "omni.timeline": None,
            "carb": None,
            "carb.eventdispatcher": None,
        }
        with patch.dict(sys.modules, blocked_modules):
            importlib.reload(isaac_hil_timeline)

    def test_subscribes_exact_play_pause_stop_events(self) -> None:
        dispatcher = FakeDispatcher()
        timeline = FakeTimeline()
        received: list[tuple[str, object]] = []

        def on_play(event) -> None:
            received.append(("play", event))

        def on_stop(event) -> None:
            received.append(("stop", event))

        with patch.dict(sys.modules, fake_isaac_modules(dispatcher)):
            handles = isaac_hil_timeline.subscribe_hil_timeline(
                timeline,
                on_play,
                on_stop,
            )

        self.assertEqual(timeline.event_key_calls, 1)
        self.assertEqual(
            [call["event_name"] for call in dispatcher.calls],
            [
                "omni.timeline:timeline:play",
                "omni.timeline:timeline:pause",
                "omni.timeline:timeline:stop",
            ],
        )
        self.assertEqual(
            [call["on_event"] for call in dispatcher.calls],
            [on_play, on_stop, on_stop],
        )
        self.assertTrue(
            all(call["filter"] is timeline.event_key for call in dispatcher.calls)
        )
        self.assertEqual(
            handles,
            tuple(call["handle"] for call in dispatcher.calls),
        )
        self.assertEqual(
            len({call["observer_name"] for call in dispatcher.calls}),
            3,
        )

        play_event = object()
        pause_event = object()
        stop_event = object()
        for call, event in zip(
            dispatcher.calls,
            (play_event, pause_event, stop_event),
        ):
            call["on_event"](event)
        self.assertEqual(
            received,
            [
                ("play", play_event),
                ("stop", pause_event),
                ("stop", stop_event),
            ],
        )


if __name__ == "__main__":
    unittest.main()
