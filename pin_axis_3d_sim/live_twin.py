"""Pure validation and state tracking for Watson's read-only Isaac mirror."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import time
from typing import Sequence


EXPECTED_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
JOINT_STATE_TOPIC = "/watson/joint_states"
STALE_AFTER_SECONDS = 0.25


@dataclass(frozen=True)
class LiveJointSample:
    """One name-normalised, finite Watson joint-state observation."""

    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    received_monotonic_s: float
    source_stamp_ns: int


def normalise_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    *,
    joint_limits: Sequence[Sequence[float]] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate a JointState payload and return it in imported DOF order."""

    names_tuple = tuple(str(name) for name in names)
    if len(names_tuple) != len(EXPECTED_JOINT_NAMES):
        raise ValueError("joint state must contain exactly six names")
    if len(set(names_tuple)) != len(names_tuple):
        raise ValueError("joint state contains duplicate names")
    if set(names_tuple) != set(EXPECTED_JOINT_NAMES):
        raise ValueError(
            f"joint state names must be exactly {list(EXPECTED_JOINT_NAMES)}"
        )
    if len(positions) != len(names_tuple):
        raise ValueError("joint state must contain one position per joint")
    if len(velocities) != len(names_tuple):
        raise ValueError("joint state must contain one velocity per joint")

    position_by_name = {
        name: float(positions[index]) for index, name in enumerate(names_tuple)
    }
    velocity_by_name = {
        name: float(velocities[index]) for index, name in enumerate(names_tuple)
    }
    ordered_positions = tuple(
        position_by_name[name] for name in EXPECTED_JOINT_NAMES
    )
    ordered_velocities = tuple(
        velocity_by_name[name] for name in EXPECTED_JOINT_NAMES
    )
    if not all(isfinite(value) for value in ordered_positions + ordered_velocities):
        raise ValueError("joint state contains a non-finite position or velocity")

    if joint_limits is not None:
        limits = tuple(tuple(float(value) for value in pair) for pair in joint_limits)
        if len(limits) != len(EXPECTED_JOINT_NAMES) or any(
            len(pair) != 2 or not all(isfinite(value) for value in pair)
            for pair in limits
        ):
            raise ValueError("imported joint limits must contain six finite pairs")
        for index, (lower, upper) in enumerate(limits):
            if lower > upper:
                raise ValueError("imported joint limit lower bound exceeds upper bound")
            position = ordered_positions[index]
            if not lower <= position <= upper:
                raise ValueError(
                    f"{EXPECTED_JOINT_NAMES[index]} position {position:.6f}rad is "
                    f"outside imported limits [{lower:.6f}, {upper:.6f}]rad"
                )

    return ordered_positions, ordered_velocities


def source_stamp_ns(seconds: int, nanoseconds: int) -> int:
    """Validate and combine a ROS time stamp without using ROS packages."""

    if isinstance(seconds, bool) or isinstance(nanoseconds, bool):
        raise ValueError("joint-state source stamp must be integer-valued")
    seconds = int(seconds)
    nanoseconds = int(nanoseconds)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("joint-state source stamp is outside ROS time bounds")
    return seconds * 1_000_000_000 + nanoseconds


class LiveJointStateBuffer:
    """Hold only the latest valid observation and classify missing/stale data."""

    def __init__(self, *, stale_after_seconds: float = STALE_AFTER_SECONDS) -> None:
        if not isfinite(stale_after_seconds) or stale_after_seconds <= 0.0:
            raise ValueError("stale threshold must be finite and positive")
        self.stale_after_seconds = float(stale_after_seconds)
        self.sample: LiveJointSample | None = None
        self.valid_messages = 0
        self.invalid_messages = 0
        self.last_invalid_reason: str | None = None
        self.first_received_monotonic_s: float | None = None

    def accept(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        velocities: Sequence[float],
        *,
        stamp_seconds: int,
        stamp_nanoseconds: int,
        joint_limits: Sequence[Sequence[float]],
        received_monotonic_s: float | None = None,
    ) -> LiveJointSample:
        received = (
            time.monotonic()
            if received_monotonic_s is None
            else float(received_monotonic_s)
        )
        if not isfinite(received) or received < 0.0:
            raise ValueError("joint-state receipt time must be finite and non-negative")
        try:
            ordered_positions, ordered_velocities = normalise_joint_state(
                names,
                positions,
                velocities,
                joint_limits=joint_limits,
            )
            stamp = source_stamp_ns(stamp_seconds, stamp_nanoseconds)
        except ValueError as exc:
            self.invalid_messages += 1
            self.last_invalid_reason = str(exc)
            raise

        sample = LiveJointSample(
            positions=ordered_positions,
            velocities=ordered_velocities,
            received_monotonic_s=received,
            source_stamp_ns=stamp,
        )
        self.sample = sample
        self.valid_messages += 1
        self.last_invalid_reason = None
        if self.first_received_monotonic_s is None:
            self.first_received_monotonic_s = received
        return sample

    def age_seconds(self, *, now_monotonic_s: float | None = None) -> float | None:
        if self.sample is None:
            return None
        now = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
        if not isfinite(now) or now < self.sample.received_monotonic_s:
            raise ValueError("current monotonic time precedes the latest sample")
        return now - self.sample.received_monotonic_s

    def status(self, *, now_monotonic_s: float | None = None) -> str:
        age = self.age_seconds(now_monotonic_s=now_monotonic_s)
        if age is None:
            return "WAITING"
        if age > self.stale_after_seconds:
            return "STALE"
        return "LIVE"

    def observed_rate_hz(self) -> float:
        if (
            self.valid_messages < 2
            or self.first_received_monotonic_s is None
            or self.sample is None
        ):
            return 0.0
        elapsed = self.sample.received_monotonic_s - self.first_received_monotonic_s
        return 0.0 if elapsed <= 0.0 else (self.valid_messages - 1) / elapsed
