"""Lazy Isaac timeline subscription bridge for the Watson HIL GUI."""

from __future__ import annotations

from typing import Any, Callable


TimelineCallback = Callable[[Any], None]
ObserverHandles = tuple[Any, Any, Any]


def subscribe_hil_timeline(
    timeline: Any,
    on_play: TimelineCallback,
    on_stop: TimelineCallback,
) -> ObserverHandles:
    """Subscribe to the default HIL Play/Pause/Stop timeline transitions.

    Isaac modules are deliberately imported only when this function is called,
    after ``SimulationApp`` has started. Pause uses the stop callback because
    Watson has no physical pause semantic.

    The returned tuple strongly owns the Play, Pause, and Stop observers, in
    that order. The caller must retain it for as long as callbacks are needed.
    """

    import omni.timeline
    from carb.eventdispatcher import get_eventdispatcher

    event_filter = timeline.get_event_key()
    dispatcher = get_eventdispatcher()
    subscriptions = (
        dispatcher.observe_event(
            observer_name="pin_axis_3d_sim.hil_timeline.play",
            filter=event_filter,
            event_name=omni.timeline.GLOBAL_EVENT_PLAY,
            on_event=on_play,
        ),
        dispatcher.observe_event(
            observer_name="pin_axis_3d_sim.hil_timeline.pause",
            filter=event_filter,
            event_name=omni.timeline.GLOBAL_EVENT_PAUSE,
            on_event=on_stop,
        ),
        dispatcher.observe_event(
            observer_name="pin_axis_3d_sim.hil_timeline.stop",
            filter=event_filter,
            event_name=omni.timeline.GLOBAL_EVENT_STOP,
            on_event=on_stop,
        ),
    )
    return subscriptions
