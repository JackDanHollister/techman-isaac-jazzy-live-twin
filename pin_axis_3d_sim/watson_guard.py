"""Pure safety checks for deliberately small, supervised Watson motions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, radians, sqrt
from types import MappingProxyType
from typing import Sequence


JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
MAX_FIRST_MOTION_EXCURSION_RAD = radians(1.0)
MAX_PLANNED_NON_TARGET_EXCURSION_RAD = 0.00001
MAX_LIVE_NON_TARGET_EXCURSION_RAD = 0.003
MAX_LIVE_VELOCITY_RAD_S = 0.10
MAX_REVERSE_VELOCITY_RAD_S = 1e-6
# Watson's stationary TM feedback reports a persistent 1.4099e-6 rad/s J6
# quantisation value. Permit it only for the raw live seed of the first cubic;
# the exact reverse arc remains part of the travel and excursion proofs.
MAX_LIVE_START_REVERSE_VELOCITY_RAD_S = 2e-6
MAX_LIVE_START_REVERSE_EXCURSION_RAD = 1e-8
TM_DRIVER_MIN_SEGMENT_DURATION_S = 0.025
FIRST_MOTION_PROFILE = "first_motion"
J6_QUALIFICATION_PROFILE = "j6_qualification"
J6_SHOWCASE_PROFILE = "j6_showcase"


@dataclass(frozen=True)
class HealthSnapshot:
    """Fields required before a real-robot trajectory may be considered."""

    is_svr_connected: bool
    is_sct_connected: bool
    tmsrv_cperr: int
    tmscript_cperr: int
    tmsrv_dataerr: int
    tmscript_dataerr: int
    is_data_table_correct: bool
    robot_link: bool
    robot_error: bool
    project_run: bool
    project_pause: bool
    safetyguard_a: bool
    e_stop: bool
    error_code: int
    project_speed: int
    ma_mode: int
    robot_light: int
    joint_positions: tuple[float, ...]
    feedback_joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    feedback_age_s: float
    joint_state_age_s: float


@dataclass(frozen=True)
class TrajectorySample:
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    accelerations: tuple[float, ...]
    time_s: float


@dataclass(frozen=True)
class J6GuardProfile:
    """Immutable physical envelope for one explicitly named J6-only motion."""

    name: str
    requested_amplitude_deg: float
    hard_excursion_deg: float
    max_planned_velocity_rad_s: float
    max_planned_acceleration_rad_s2: float
    max_live_velocity_rad_s: float
    max_sample_step_rad: float
    min_duration_s: float
    max_duration_s: float
    velocity_scaling: float
    acceleration_scaling: float
    max_project_speed: int

    @property
    def requested_amplitude_rad(self) -> float:
        return radians(self.requested_amplitude_deg)

    @property
    def hard_excursion_rad(self) -> float:
        return radians(self.hard_excursion_deg)


J6_GUARD_PROFILES = MappingProxyType(
    {
        FIRST_MOTION_PROFILE: J6GuardProfile(
            name=FIRST_MOTION_PROFILE,
            requested_amplitude_deg=0.9,
            hard_excursion_deg=1.0,
            max_planned_velocity_rad_s=0.10,
            max_planned_acceleration_rad_s2=0.10,
            max_live_velocity_rad_s=0.10,
            max_sample_step_rad=0.03,
            min_duration_s=0.25,
            max_duration_s=30.0,
            velocity_scaling=0.01,
            acceleration_scaling=0.01,
            max_project_speed=5,
        ),
        J6_QUALIFICATION_PROFILE: J6GuardProfile(
            name=J6_QUALIFICATION_PROFILE,
            requested_amplitude_deg=6.0,
            hard_excursion_deg=7.0,
            max_planned_velocity_rad_s=0.18,
            max_planned_acceleration_rad_s2=0.40,
            max_live_velocity_rad_s=0.18,
            max_sample_step_rad=radians(7.0),
            min_duration_s=0.75,
            max_duration_s=5.0,
            velocity_scaling=0.02,
            acceleration_scaling=0.02,
            max_project_speed=5,
        ),
        J6_SHOWCASE_PROFILE: J6GuardProfile(
            name=J6_SHOWCASE_PROFILE,
            requested_amplitude_deg=12.0,
            hard_excursion_deg=13.0,
            max_planned_velocity_rad_s=0.35,
            max_planned_acceleration_rad_s2=0.75,
            max_live_velocity_rad_s=0.35,
            max_sample_step_rad=radians(13.0),
            min_duration_s=0.75,
            max_duration_s=4.0,
            velocity_scaling=0.04,
            acceleration_scaling=0.04,
            max_project_speed=5,
        ),
    }
)


def get_j6_guard_profile(name: str) -> J6GuardProfile:
    """Resolve only a built-in profile; callers cannot construct wider limits."""

    try:
        return J6_GUARD_PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown guarded J6 motion profile: {name!r}") from exc


def health_failures(
    snapshot: HealthSnapshot,
    *,
    max_project_speed: int = 5,
    max_stationary_velocity_rad_s: float = 0.01,
    max_message_age_s: float = 0.5,
    max_position_source_delta_rad: float = 0.005,
    require_stationary: bool = True,
    require_auto_mode: bool = True,
) -> list[str]:
    """Return every failed gate instead of stopping at the first one."""

    failures: list[str] = []
    boolean_gates = (
        (snapshot.is_svr_connected, "Ethernet Slave is not connected"),
        (snapshot.is_sct_connected, "Listen Node is not connected"),
        (snapshot.is_data_table_correct, "TMflow Ethernet Slave data table is not correct"),
        (snapshot.robot_link, "robot controller link is not healthy"),
        (not snapshot.robot_error, "robot_error is set"),
        (snapshot.project_run, "TMflow project is not running"),
        (not snapshot.project_pause, "TMflow project is paused"),
        (not snapshot.safetyguard_a, "safeguard input is active"),
        (not snapshot.e_stop, "E-stop is active"),
        (snapshot.error_code == 0, f"robot error_code is {snapshot.error_code}"),
    )
    failures.extend(message for passed, message in boolean_gates if not passed)

    communication_errors = {
        "tmsrv_cperr": snapshot.tmsrv_cperr,
        "tmscript_cperr": snapshot.tmscript_cperr,
        "tmsrv_dataerr": snapshot.tmsrv_dataerr,
        "tmscript_dataerr": snapshot.tmscript_dataerr,
    }
    failures.extend(
        f"{name} is {value}"
        for name, value in communication_errors.items()
        if value != 0
    )
    # Watson is a TM5S. Its S-series Auto states are 20=standby and 21=running.
    # Legacy blue-light values, errors, safeguard, maintenance, and recovery
    # states are deliberately excluded from this target-specific physical gate.
    if require_auto_mode and snapshot.robot_light not in {20, 21}:
        failures.append(
            "Auto standby/running is not verified "
            f"(ma_mode={snapshot.ma_mode}, robot_light={snapshot.robot_light})"
        )

    if not 0 < snapshot.project_speed <= max_project_speed:
        failures.append(
            f"project_speed must be 1..{max_project_speed}, got {snapshot.project_speed}"
        )
    if snapshot.feedback_age_s < 0.0 or snapshot.feedback_age_s > max_message_age_s:
        failures.append(
            f"feedback is stale ({snapshot.feedback_age_s:.3f}s > {max_message_age_s:.3f}s)"
        )
    if snapshot.joint_state_age_s < 0.0 or snapshot.joint_state_age_s > max_message_age_s:
        failures.append(
            f"joint state is stale ({snapshot.joint_state_age_s:.3f}s > {max_message_age_s:.3f}s)"
        )
    if len(snapshot.joint_positions) != len(JOINT_NAMES):
        failures.append(f"expected {len(JOINT_NAMES)} joint positions")
    elif not all(isfinite(value) for value in snapshot.joint_positions):
        failures.append("joint positions contain a non-finite value")
    if len(snapshot.feedback_joint_positions) != len(JOINT_NAMES):
        failures.append(f"expected {len(JOINT_NAMES)} feedback joint positions")
    elif not all(isfinite(value) for value in snapshot.feedback_joint_positions):
        failures.append("feedback joint positions contain a non-finite value")
    elif len(snapshot.joint_positions) == len(JOINT_NAMES) and all(
        isfinite(value) for value in snapshot.joint_positions
    ):
        source_delta = max(
            abs(snapshot.joint_positions[index] - snapshot.feedback_joint_positions[index])
            for index in range(len(JOINT_NAMES))
        )
        if source_delta > max_position_source_delta_rad:
            failures.append(
                "joint_states and feedback joint positions disagree "
                f"({source_delta:.6f}rad > {max_position_source_delta_rad:.6f}rad)"
            )
    if len(snapshot.joint_velocities) != len(JOINT_NAMES):
        failures.append(f"expected {len(JOINT_NAMES)} joint velocities")
    elif not all(isfinite(value) for value in snapshot.joint_velocities):
        failures.append("joint velocities contain a non-finite value")
    elif require_stationary:
        max_velocity = max(abs(value) for value in snapshot.joint_velocities)
        if max_velocity > max_stationary_velocity_rad_s:
            failures.append(
                "robot is not stationary "
                f"({max_velocity:.6f}rad/s > {max_stationary_velocity_rad_s:.6f}rad/s)"
            )
    return failures


def wrist_check_targets(
    current_positions: Sequence[float],
    *,
    amplitude_deg: float = 0.9,
) -> list[tuple[str, tuple[float, ...]]]:
    """Move J6 one small step toward zero, then return to the captured state."""

    if len(current_positions) != len(JOINT_NAMES):
        raise ValueError(f"expected {len(JOINT_NAMES)} current joint positions")
    current = tuple(float(value) for value in current_positions)
    if not all(isfinite(value) for value in current):
        raise ValueError("current joint positions must be finite")
    if not 0.0 < amplitude_deg <= 1.0:
        raise ValueError("first-motion amplitude must be greater than 0 and at most 1 degree")

    target = list(current)
    step = radians(amplitude_deg)
    # Moving toward zero avoids consuming additional travel at either wrapped J6 extreme.
    target[-1] += -step if current[-1] > 0.0 else step
    return [("wrist_check", tuple(target)), ("return_to_start", current)]


def j6_profile_targets(
    current_positions: Sequence[float],
    *,
    guard_profile: str,
) -> list[tuple[str, tuple[float, ...]]]:
    """Build the exact locked 6- or 12-degree J6 out-and-return sequence."""

    profile = get_j6_guard_profile(guard_profile)
    if profile.name == FIRST_MOTION_PROFILE:
        raise ValueError("first_motion must use wrist_check_targets")
    if len(current_positions) != len(JOINT_NAMES):
        raise ValueError(f"expected {len(JOINT_NAMES)} current joint positions")
    current = tuple(float(value) for value in current_positions)
    if not all(isfinite(value) for value in current):
        raise ValueError("current joint positions must be finite")

    target = list(current)
    step = profile.requested_amplitude_rad
    target[-1] += -step if current[-1] > 0.0 else step
    stage_name = (
        "j6_qualification"
        if profile.name == J6_QUALIFICATION_PROFILE
        else "j6_showcase"
    )
    return [(stage_name, tuple(target)), ("return_to_start", current)]


def motion_envelope_failures(
    snapshot: HealthSnapshot,
    *,
    expected_start: Sequence[float],
    expected_goal: Sequence[float],
    hard_reference_start: Sequence[float] | None = None,
    hard_travel_start: Sequence[float] | None = None,
    max_non_target_excursion_rad: float = MAX_LIVE_NON_TARGET_EXCURSION_RAD,
    max_target_overshoot_rad: float = 0.003,
    max_live_velocity_rad_s: float = 0.10,
    guard_profile: str = FIRST_MOTION_PROFILE,
) -> list[str]:
    """Monitor physical feedback against the deliberately tiny J6 envelope."""

    profile = get_j6_guard_profile(guard_profile)
    start = tuple(float(value) for value in expected_start)
    goal = tuple(float(value) for value in expected_goal)
    hard_reference = (
        start
        if hard_reference_start is None
        else tuple(float(value) for value in hard_reference_start)
    )
    travel_start = (
        start
        if hard_travel_start is None
        else tuple(float(value) for value in hard_travel_start)
    )
    positions = snapshot.feedback_joint_positions
    velocities = snapshot.joint_velocities
    if any(
        len(values) != len(JOINT_NAMES)
        for values in (start, goal, hard_reference, travel_start, positions, velocities)
    ):
        return ["live motion envelope does not contain six-axis data"]
    if not all(
        isfinite(value)
        for values in (start, goal, hard_reference, travel_start, positions, velocities)
        for value in values
    ):
        return ["live motion envelope contains non-finite data"]

    limits = (
        max_non_target_excursion_rad,
        max_target_overshoot_rad,
        max_live_velocity_rad_s,
    )
    if not all(isfinite(value) and value >= 0.0 for value in limits):
        return ["live motion envelope limits are non-finite or negative"]

    failures: list[str] = []
    hard_non_target_limit = min(
        max_non_target_excursion_rad,
        MAX_LIVE_NON_TARGET_EXCURSION_RAD,
    )
    non_target_excursion = max(
        abs(positions[joint] - hard_reference[joint])
        for joint in range(len(JOINT_NAMES) - 1)
    )
    if non_target_excursion > hard_non_target_limit:
        failures.append(
            "live non-target joint excursion "
            f"{non_target_excursion:.6f}rad exceeds {hard_non_target_limit:.6f}rad"
        )
    target_low = max(
        min(start[-1], goal[-1]) - max_target_overshoot_rad,
        start[-1] - profile.hard_excursion_rad,
        hard_reference[-1] - profile.hard_excursion_rad,
        travel_start[-1] - profile.hard_excursion_rad,
    )
    target_high = min(
        max(start[-1], goal[-1]) + max_target_overshoot_rad,
        start[-1] + profile.hard_excursion_rad,
        hard_reference[-1] + profile.hard_excursion_rad,
        travel_start[-1] + profile.hard_excursion_rad,
    )
    if not target_low <= positions[-1] <= target_high:
        failures.append(
            f"live joint_6 position {positions[-1]:.6f}rad left guarded interval "
            f"[{target_low:.6f}, {target_high:.6f}]rad"
        )
    max_velocity = max(abs(value) for value in velocities)
    hard_live_velocity_limit = min(
        max_live_velocity_rad_s,
        profile.max_live_velocity_rad_s,
    )
    if max_velocity > hard_live_velocity_limit:
        failures.append(
            f"live joint velocity {max_velocity:.6f}rad/s exceeds "
            f"{hard_live_velocity_limit:.6f}rad/s"
        )
    return failures


def _real_roots_in_unit_interval(a: float, b: float, c: float) -> tuple[float, ...]:
    """Solve a*s^2+b*s+c=0 and retain real roots in [0, 1]."""

    epsilon = 1e-15
    roots: list[float] = []
    if abs(a) <= epsilon:
        if abs(b) > epsilon:
            root = -c / b
            if 0.0 <= root <= 1.0:
                roots.append(root)
        return tuple(roots)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return ()
    root_term = sqrt(max(discriminant, 0.0))
    for root in ((-b - root_term) / (2.0 * a), (-b + root_term) / (2.0 * a)):
        if 0.0 <= root <= 1.0 and not any(abs(root - seen) <= epsilon for seen in roots):
            roots.append(root)
    return tuple(roots)


def _hermite_segment_extrema(
    q0: float,
    q1: float,
    v0: float,
    v1: float,
    duration_s: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return exact position/velocity/acceleration candidates for cubic PVT."""

    h = duration_s
    a = 2.0 * q0 - 2.0 * q1 + h * (v0 + v1)
    b = -3.0 * q0 + 3.0 * q1 - h * (2.0 * v0 + v1)
    c = h * v0
    d = q0

    def position(s: float) -> float:
        return ((a * s + b) * s + c) * s + d

    def velocity(s: float) -> float:
        return (3.0 * a * s * s + 2.0 * b * s + c) / h

    def acceleration(s: float) -> float:
        return (6.0 * a * s + 2.0 * b) / (h * h)

    position_parameters = tuple(
        sorted(
            (0.0, 1.0)
            + _real_roots_in_unit_interval(3.0 * a, 2.0 * b, c)
        )
    )
    velocity_parameters = [0.0, 1.0]
    if abs(a) > 1e-15:
        acceleration_root = -b / (3.0 * a)
        if 0.0 <= acceleration_root <= 1.0:
            velocity_parameters.append(acceleration_root)
    return (
        tuple(position(s) for s in position_parameters),
        tuple(velocity(s) for s in velocity_parameters),
        (acceleration(0.0), acceleration(1.0)),
    )


def _emulate_tm_driver_pvt_filter(
    planned_samples: Sequence[TrajectorySample],
    *,
    execution_start_positions: Sequence[float] | None = None,
    execution_start_velocities: Sequence[float] | None = None,
) -> tuple[tuple[TrajectorySample, ...], int]:
    """Mirror tm_driver filtering and its omitted zero-time PVT point.

    The controller does not receive the planned zero-time point. During an
    execution recheck, seed the first physical cubic from the final live robot
    position and velocity instead.
    """

    if len(planned_samples) < 2:
        raise ValueError("trajectory must contain at least two samples")
    if abs(planned_samples[0].time_s) > 1e-12:
        raise ValueError("first trajectory sample must have time_from_start exactly 0")
    if (execution_start_positions is None) != (execution_start_velocities is None):
        raise ValueError(
            "execution start positions and velocities must be supplied together"
        )

    if execution_start_positions is None:
        first_positions = planned_samples[0].positions
        first_velocities = planned_samples[0].velocities
        first_accelerations = planned_samples[0].accelerations
    else:
        first_positions = tuple(float(value) for value in execution_start_positions)
        first_velocities = tuple(float(value) for value in execution_start_velocities)
        if len(first_positions) != len(JOINT_NAMES):
            raise ValueError("execution start must contain six positions")
        if len(first_velocities) != len(JOINT_NAMES):
            raise ValueError("execution start must contain six velocities")
        if not all(isfinite(value) for value in first_positions + first_velocities):
            raise ValueError("execution start contains a non-finite value")
        # PVT consumes endpoint positions and velocities, not accelerations.
        first_accelerations = (0.0,) * len(JOINT_NAMES)

    selected: list[tuple[TrajectorySample, float]] = []
    previous_selected_index = 0
    second_previous_selected_index = 0
    for index in range(1, len(planned_samples) - 1):
        segment_duration = (
            planned_samples[index].time_s
            - planned_samples[previous_selected_index].time_s
        )
        if segment_duration >= TM_DRIVER_MIN_SEGMENT_DURATION_S:
            second_previous_selected_index = previous_selected_index
            previous_selected_index = index
            selected.append((planned_samples[index], segment_duration))

    last_index = len(planned_samples) - 1
    last_duration = (
        planned_samples[last_index].time_s
        - planned_samples[previous_selected_index].time_s
    )
    if last_duration >= TM_DRIVER_MIN_SEGMENT_DURATION_S:
        selected.append((planned_samples[last_index], last_duration))
    else:
        if not selected:
            raise ValueError(
                "tm_driver would have no PVT point to replace for the short final segment"
            )
        replacement_duration = (
            planned_samples[last_index].time_s
            - planned_samples[second_previous_selected_index].time_s
        )
        selected[-1] = (planned_samples[last_index], replacement_duration)

    executed = [
        TrajectorySample(
            positions=first_positions,
            velocities=first_velocities,
            accelerations=first_accelerations,
            time_s=0.0,
        )
    ]
    cumulative_time = 0.0
    for endpoint, segment_duration in selected:
        if segment_duration < TM_DRIVER_MIN_SEGMENT_DURATION_S:
            raise ValueError(
                f"emulated tm_driver PVT segment is too short ({segment_duration:.6f}s)"
            )
        cumulative_time += segment_duration
        executed.append(
            TrajectorySample(
                positions=endpoint.positions,
                velocities=endpoint.velocities,
                accelerations=endpoint.accelerations,
                time_s=cumulative_time,
            )
        )
    expected_duration = planned_samples[-1].time_s
    if abs(cumulative_time - expected_duration) > 1e-9:
        raise ValueError(
            "emulated tm_driver PVT duration does not match planned total duration"
        )
    return tuple(executed), len(planned_samples) - len(executed)


def validate_trajectory_samples(
    samples: Sequence[TrajectorySample],
    *,
    expected_start: Sequence[float],
    expected_goal: Sequence[float],
    hard_reference_start: Sequence[float] | None = None,
    hard_travel_start: Sequence[float] | None = None,
    execution_start_positions: Sequence[float] | None = None,
    execution_start_velocities: Sequence[float] | None = None,
    max_start_error_rad: float = 0.001,
    max_goal_error_rad: float = 0.001,
    max_excursion_rad: float = MAX_FIRST_MOTION_EXCURSION_RAD,
    max_sample_step_rad: float = 0.03,
    max_velocity_rad_s: float = 0.10,
    max_acceleration_rad_s2: float = 0.10,
    max_non_target_excursion_rad: float = MAX_PLANNED_NON_TARGET_EXCURSION_RAD,
    max_target_overshoot_rad: float = 0.001,
    min_total_duration_s: float = 0.25,
    max_total_duration_s: float = 30.0,
    max_endpoint_velocity_rad_s: float = 0.005,
    max_target_path_excess_rad: float = 0.002,
    max_reverse_velocity_rad_s: float = 1e-6,
    max_live_start_reverse_velocity_rad_s: float = (
        MAX_LIVE_START_REVERSE_VELOCITY_RAD_S
    ),
    max_live_start_reverse_excursion_rad: float = (
        MAX_LIVE_START_REVERSE_EXCURSION_RAD
    ),
    guard_profile: str = FIRST_MOTION_PROFILE,
) -> dict[str, float | int]:
    """Fail closed on malformed or unexpectedly energetic MoveIt/PVT output."""

    profile = get_j6_guard_profile(guard_profile)
    if len(samples) < 2:
        raise ValueError("trajectory must contain at least two samples")
    start = tuple(float(value) for value in expected_start)
    goal = tuple(float(value) for value in expected_goal)
    hard_reference = (
        start
        if hard_reference_start is None
        else tuple(float(value) for value in hard_reference_start)
    )
    if (execution_start_positions is None) != (execution_start_velocities is None):
        raise ValueError(
            "execution start positions and velocities must be supplied together"
        )
    execution_start = (
        None
        if execution_start_positions is None
        else tuple(float(value) for value in execution_start_positions)
    )
    execution_velocities = (
        None
        if execution_start_velocities is None
        else tuple(float(value) for value in execution_start_velocities)
    )
    supplied_travel_start = (
        None
        if hard_travel_start is None
        else tuple(float(value) for value in hard_travel_start)
    )
    if execution_start is not None:
        if supplied_travel_start is not None:
            if len(supplied_travel_start) != len(execution_start) or any(
                abs(supplied_travel_start[index] - execution_start[index]) > 1e-12
                for index in range(len(execution_start))
            ):
                raise ValueError("hard travel start must equal the live execution start")
        travel_start = execution_start
    else:
        travel_start = start if supplied_travel_start is None else supplied_travel_start
    if any(
        len(values) != len(JOINT_NAMES)
        for values in (
            start,
            goal,
            hard_reference,
            travel_start,
            *(() if execution_start is None else (execution_start, execution_velocities)),
        )
    ):
        raise ValueError(
            "expected, hard-reference, travel, and execution states must contain six joints"
        )
    if not all(
        isfinite(value)
        for values in (
            start,
            goal,
            hard_reference,
            travel_start,
            *(() if execution_start is None else (execution_start, execution_velocities)),
        )
        for value in values
    ):
        raise ValueError(
            "expected, hard-reference, travel, and execution states must be finite"
        )
    for joint in range(len(JOINT_NAMES) - 1):
        if abs(goal[joint] - start[joint]) > 1e-9:
            raise ValueError("guarded trajectory may change only joint_6")
    requested_target_displacement = goal[-1] - start[-1]
    if abs(requested_target_displacement) <= 1e-9:
        raise ValueError("guarded joint_6 displacement must be non-zero")
    if profile.name == FIRST_MOTION_PROFILE:
        if abs(requested_target_displacement) > radians(1.0) + 1e-9:
            raise ValueError("first-motion joint_6 displacement exceeds one degree")
    elif (
        abs(abs(requested_target_displacement) - profile.requested_amplitude_rad)
        > 1e-9
    ):
        raise ValueError(
            f"{profile.name} joint_6 displacement must be exactly "
            f"{profile.requested_amplitude_deg:.1f} degrees"
        )
    target_direction = 1.0 if requested_target_displacement > 0.0 else -1.0

    nonnegative_limits = {
        "max_start_error_rad": max_start_error_rad,
        "max_goal_error_rad": max_goal_error_rad,
        "max_non_target_excursion_rad": max_non_target_excursion_rad,
        "max_target_overshoot_rad": max_target_overshoot_rad,
        "max_endpoint_velocity_rad_s": max_endpoint_velocity_rad_s,
        "max_target_path_excess_rad": max_target_path_excess_rad,
        "max_reverse_velocity_rad_s": max_reverse_velocity_rad_s,
        "max_live_start_reverse_velocity_rad_s": (
            max_live_start_reverse_velocity_rad_s
        ),
        "max_live_start_reverse_excursion_rad": (
            max_live_start_reverse_excursion_rad
        ),
    }
    positive_limits = {
        "max_excursion_rad": max_excursion_rad,
        "max_sample_step_rad": max_sample_step_rad,
        "max_velocity_rad_s": max_velocity_rad_s,
        "max_acceleration_rad_s2": max_acceleration_rad_s2,
        "min_total_duration_s": min_total_duration_s,
        "max_total_duration_s": max_total_duration_s,
    }
    for name, value in nonnegative_limits.items():
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name, value in positive_limits.items():
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if max_total_duration_s < min_total_duration_s:
        raise ValueError("max_total_duration_s must be at least min_total_duration_s")

    # An execution recheck never replaces the planning-time proof. Validate the
    # exact planned PVT first, then repeat below with the live state as the
    # controller's physical start. This keeps skipped-point and non-target
    # constraints hard even when the helper is called directly.
    if execution_start is not None:
        validate_trajectory_samples(
            samples,
            expected_start=start,
            expected_goal=goal,
            hard_reference_start=hard_reference,
            hard_travel_start=start,
            max_start_error_rad=max_start_error_rad,
            max_goal_error_rad=max_goal_error_rad,
            max_excursion_rad=max_excursion_rad,
            max_sample_step_rad=max_sample_step_rad,
            max_velocity_rad_s=max_velocity_rad_s,
            max_acceleration_rad_s2=max_acceleration_rad_s2,
            max_non_target_excursion_rad=max_non_target_excursion_rad,
            max_target_overshoot_rad=max_target_overshoot_rad,
            min_total_duration_s=min_total_duration_s,
            max_total_duration_s=max_total_duration_s,
            max_endpoint_velocity_rad_s=max_endpoint_velocity_rad_s,
            max_target_path_excess_rad=max_target_path_excess_rad,
            max_reverse_velocity_rad_s=max_reverse_velocity_rad_s,
            max_live_start_reverse_velocity_rad_s=(
                max_live_start_reverse_velocity_rad_s
            ),
            max_live_start_reverse_excursion_rad=(
                max_live_start_reverse_excursion_rad
            ),
            guard_profile=profile.name,
        )

    planned_samples = tuple(samples)
    previous_planned_time = -1.0
    for index, sample in enumerate(planned_samples):
        if len(sample.positions) != len(JOINT_NAMES):
            raise ValueError(f"trajectory sample {index} does not contain six positions")
        if len(sample.velocities) != len(JOINT_NAMES):
            raise ValueError(f"trajectory sample {index} must contain six velocities")
        if len(sample.accelerations) != len(JOINT_NAMES):
            raise ValueError(f"trajectory sample {index} must contain six accelerations")
        values = sample.positions + sample.velocities + sample.accelerations + (sample.time_s,)
        if not all(isfinite(value) for value in values):
            raise ValueError(f"trajectory sample {index} contains a non-finite value")
        if sample.time_s < 0.0 or sample.time_s <= previous_planned_time:
            raise ValueError("trajectory timestamps must be non-negative and strictly increasing")
        previous_planned_time = sample.time_s
    endpoint_velocity = max(
        max(abs(value) for value in planned_samples[0].velocities),
        max(abs(value) for value in planned_samples[-1].velocities),
    )
    if execution_velocities is not None:
        endpoint_velocity = max(
            endpoint_velocity,
            max(abs(value) for value in execution_velocities),
        )
    if endpoint_velocity > max_endpoint_velocity_rad_s:
        raise ValueError(
            f"trajectory endpoint velocity {endpoint_velocity:.6f}rad/s exceeds "
            f"{max_endpoint_velocity_rad_s:.6f}rad/s"
        )

    samples, pvt_skipped_points = _emulate_tm_driver_pvt_filter(
        planned_samples,
        execution_start_positions=execution_start,
        execution_start_velocities=execution_velocities,
    )

    previous_time = -1.0
    previous_positions: tuple[float, ...] | None = None
    previous_velocities: tuple[float, ...] | None = None
    previous_segment_velocity: tuple[float, ...] | None = None
    previous_segment_duration: float | None = None
    max_step = 0.0
    max_excursion = 0.0
    max_velocity = 0.0
    max_acceleration = 0.0
    max_derived_velocity = 0.0
    max_derived_acceleration = 0.0
    max_interpolated_velocity = 0.0
    max_interpolated_acceleration = 0.0
    max_non_target_excursion = 0.0
    max_live_start_reverse_excursion = 0.0
    target_path_length = 0.0
    hard_excursion_limit = min(max_excursion_rad, profile.hard_excursion_rad)
    hard_velocity_limit = min(
        max_velocity_rad_s,
        profile.max_planned_velocity_rad_s,
    )
    hard_acceleration_limit = min(
        max_acceleration_rad_s2,
        profile.max_planned_acceleration_rad_s2,
    )
    hard_min_duration = max(min_total_duration_s, profile.min_duration_s)
    hard_max_duration = min(max_total_duration_s, profile.max_duration_s)
    if hard_max_duration < hard_min_duration:
        raise ValueError(
            f"duration limits do not overlap the {profile.name} profile"
        )
    hard_non_target_limit = min(
        max_non_target_excursion_rad,
        MAX_PLANNED_NON_TARGET_EXCURSION_RAD,
    )
    if execution_start is not None:
        hard_non_target_limit = MAX_LIVE_NON_TARGET_EXCURSION_RAD
    hard_reverse_velocity_limit = min(
        max_reverse_velocity_rad_s,
        MAX_REVERSE_VELOCITY_RAD_S,
    )
    hard_live_start_reverse_velocity_limit = min(
        max_live_start_reverse_velocity_rad_s,
        MAX_LIVE_START_REVERSE_VELOCITY_RAD_S,
    )
    hard_live_start_reverse_excursion_limit = min(
        max_live_start_reverse_excursion_rad,
        MAX_LIVE_START_REVERSE_EXCURSION_RAD,
    )
    target_low = max(
        min(start[-1], goal[-1]) - max_target_overshoot_rad,
        start[-1] - hard_excursion_limit,
        hard_reference[-1] - hard_excursion_limit,
        travel_start[-1] - hard_excursion_limit,
    )
    target_high = min(
        max(start[-1], goal[-1]) + max_target_overshoot_rad,
        start[-1] + hard_excursion_limit,
        hard_reference[-1] + hard_excursion_limit,
        travel_start[-1] + hard_excursion_limit,
    )
    for index, sample in enumerate(samples):
        if len(sample.positions) != len(JOINT_NAMES):
            raise ValueError(f"trajectory sample {index} does not contain six positions")
        if len(sample.velocities) != len(JOINT_NAMES):
            raise ValueError(f"trajectory sample {index} must contain six velocities")
        if len(sample.accelerations) != len(JOINT_NAMES):
            raise ValueError(f"trajectory sample {index} must contain six accelerations")
        values = sample.positions + sample.velocities + sample.accelerations + (sample.time_s,)
        if not all(isfinite(value) for value in values):
            raise ValueError(f"trajectory sample {index} contains a non-finite value")
        prior_time = previous_time
        if sample.time_s < 0.0 or sample.time_s <= previous_time:
            raise ValueError("trajectory timestamps must be non-negative and strictly increasing")

        max_excursion = max(
            max_excursion,
            max(
                abs(value - hard_reference[joint])
                for joint, value in enumerate(sample.positions)
            ),
        )
        max_non_target_excursion = max(
            max_non_target_excursion,
            max(
                abs(sample.positions[joint] - hard_reference[joint])
                for joint in range(len(JOINT_NAMES) - 1)
            ),
        )
        if not (
            target_low - 1e-12
            <= sample.positions[-1]
            <= target_high + 1e-12
        ):
            raise ValueError(
                "joint_6 trajectory overshoots the guarded J6 segment "
                f"at sample {index}"
            )
        if previous_positions is not None:
            if previous_velocities is None:
                raise ValueError("previous trajectory sample has no velocities")
            segment_duration = sample.time_s - prior_time
            if segment_duration < TM_DRIVER_MIN_SEGMENT_DURATION_S:
                raise ValueError(
                    f"trajectory segment {index} is too short ({segment_duration:.6f}s)"
                )
            segment_velocity = tuple(
                (value - previous_positions[joint]) / segment_duration
                for joint, value in enumerate(sample.positions)
            )
            max_derived_velocity = max(
                max_derived_velocity,
                max(abs(value) for value in segment_velocity),
            )
            if previous_segment_velocity is not None and previous_segment_duration is not None:
                centre_duration = 0.5 * (segment_duration + previous_segment_duration)
                max_derived_acceleration = max(
                    max_derived_acceleration,
                    max(
                        abs(segment_velocity[joint] - previous_segment_velocity[joint])
                        / centre_duration
                        for joint in range(len(JOINT_NAMES))
                    ),
                )
            previous_segment_velocity = segment_velocity
            previous_segment_duration = segment_duration
            target_step = sample.positions[-1] - previous_positions[-1]
            if target_direction * target_step < -1e-9:
                raise ValueError(
                    "joint_6 trajectory reverses direction between "
                    f"samples {index - 1} and {index}"
                )
            for joint in range(len(JOINT_NAMES)):
                positions, velocities, accelerations = _hermite_segment_extrema(
                    previous_positions[joint],
                    sample.positions[joint],
                    previous_velocities[joint],
                    sample.velocities[joint],
                    segment_duration,
                )
                max_interpolated_velocity = max(
                    max_interpolated_velocity,
                    max(abs(value) for value in velocities),
                )
                max_interpolated_acceleration = max(
                    max_interpolated_acceleration,
                    max(abs(value) for value in accelerations),
                )
                if joint < len(JOINT_NAMES) - 1:
                    continuous_excursion = max(
                        abs(value - hard_reference[joint]) for value in positions
                    )
                    max_excursion = max(max_excursion, continuous_excursion)
                    max_non_target_excursion = max(
                        max_non_target_excursion,
                        continuous_excursion,
                    )
                else:
                    target_path_length += sum(
                        abs(positions[position_index] - positions[position_index - 1])
                        for position_index in range(1, len(positions))
                    )
                    if execution_start is not None and index == 1:
                        live_start_reverse_excursion = max(
                            0.0,
                            max(
                                -target_direction
                                * (value - previous_positions[-1])
                                for value in positions
                            ),
                        )
                        max_live_start_reverse_excursion = max(
                            max_live_start_reverse_excursion,
                            live_start_reverse_excursion,
                        )
                        if live_start_reverse_excursion > (
                            hard_live_start_reverse_excursion_limit + 1e-15
                        ):
                            raise ValueError(
                                "live-start cubic joint_6 trajectory reverse excursion "
                                f"{live_start_reverse_excursion:.12f}rad exceeds "
                                f"{hard_live_start_reverse_excursion_limit:.12f}rad"
                            )
                    if any(
                        not target_low - 1e-12 <= value <= target_high + 1e-12
                        for value in positions
                    ):
                        raise ValueError(
                            "cubic joint_6 trajectory overshoots the guarded segment "
                            f"between samples {index - 1} and {index}"
                        )
                    reverse_velocity_limit = hard_reverse_velocity_limit
                    if execution_start is not None and index == 1:
                        # Only the controller's omitted-zero-point segment uses
                        # the tightly capped live-feedback noise allowance.
                        reverse_velocity_limit = (
                            hard_live_start_reverse_velocity_limit
                        )
                    if min(target_direction * value for value in velocities) < (
                        -reverse_velocity_limit
                    ):
                        raise ValueError(
                            "cubic joint_6 trajectory reverses direction "
                            f"between samples {index - 1} and {index}"
                        )
            max_step = max(
                max_step,
                max(
                    abs(value - previous_positions[joint])
                    for joint, value in enumerate(sample.positions)
                ),
            )
        previous_positions = sample.positions
        previous_velocities = sample.velocities
        max_velocity = max(max_velocity, max(abs(value) for value in sample.velocities))
        max_acceleration = max(
            max_acceleration,
            max(abs(value) for value in sample.accelerations),
        )
        previous_time = sample.time_s

    start_error = max(
        abs(samples[0].positions[joint] - start[joint])
        for joint in range(len(JOINT_NAMES))
    )
    goal_error = max(
        abs(samples[-1].positions[joint] - goal[joint])
        for joint in range(len(JOINT_NAMES))
    )
    if start_error > max_start_error_rad:
        raise ValueError(
            f"trajectory start mismatch {start_error:.6f}rad exceeds {max_start_error_rad:.6f}rad"
        )
    if goal_error > max_goal_error_rad:
        raise ValueError(
            f"trajectory goal mismatch {goal_error:.6f}rad exceeds {max_goal_error_rad:.6f}rad"
        )
    if max_excursion > hard_excursion_limit + 1e-12:
        cap_name = (
            "guarded one-degree cap"
            if profile.name == FIRST_MOTION_PROFILE
            else f"guarded {profile.hard_excursion_deg:.1f}-degree profile cap"
        )
        raise ValueError(
            "trajectory excursion "
            f"{max_excursion:.6f}rad exceeds the {cap_name} "
            f"{hard_excursion_limit:.6f}rad"
        )
    if max_non_target_excursion > hard_non_target_limit:
        raise ValueError(
            "non-target joint excursion "
            f"{max_non_target_excursion:.6f}rad exceeds "
            f"{hard_non_target_limit:.6f}rad"
        )
    max_target_path_length = (
        min(
            abs(requested_target_displacement) + max_target_path_excess_rad,
            profile.hard_excursion_rad,
        )
    )
    if target_path_length > max_target_path_length + 1e-12:
        raise ValueError(
            "joint_6 trajectory path length "
            f"{target_path_length:.6f}rad exceeds {max_target_path_length:.6f}rad"
        )
    if max_step > max_sample_step_rad:
        raise ValueError(
            f"trajectory sample step {max_step:.6f}rad exceeds {max_sample_step_rad:.6f}rad"
        )
    if max_velocity > hard_velocity_limit:
        raise ValueError(
            f"trajectory velocity {max_velocity:.6f}rad/s exceeds {hard_velocity_limit:.6f}rad/s"
        )
    if max_acceleration > hard_acceleration_limit:
        raise ValueError(
            "trajectory acceleration "
            f"{max_acceleration:.6f}rad/s^2 exceeds {hard_acceleration_limit:.6f}rad/s^2"
        )
    if max_derived_velocity > hard_velocity_limit:
        raise ValueError(
            "position/time-derived trajectory velocity "
            f"{max_derived_velocity:.6f}rad/s exceeds {hard_velocity_limit:.6f}rad/s"
        )
    if max_derived_acceleration > hard_acceleration_limit:
        raise ValueError(
            "position/time-derived trajectory acceleration "
            f"{max_derived_acceleration:.6f}rad/s^2 exceeds "
            f"{hard_acceleration_limit:.6f}rad/s^2"
        )
    if max_interpolated_velocity > hard_velocity_limit:
        raise ValueError(
            "cubic-interpolated trajectory velocity "
            f"{max_interpolated_velocity:.6f}rad/s exceeds {hard_velocity_limit:.6f}rad/s"
        )
    if max_interpolated_acceleration > hard_acceleration_limit:
        raise ValueError(
            "cubic-interpolated trajectory acceleration "
            f"{max_interpolated_acceleration:.6f}rad/s^2 exceeds "
            f"{hard_acceleration_limit:.6f}rad/s^2"
        )
    total_duration = samples[-1].time_s - samples[0].time_s
    if not hard_min_duration <= total_duration <= hard_max_duration:
        raise ValueError(
            f"trajectory duration {total_duration:.6f}s is outside "
            f"[{hard_min_duration:.6f}, {hard_max_duration:.6f}]s"
        )
    return {
        "sample_count": len(planned_samples),
        "tm_pvt_sample_count": len(samples),
        "tm_pvt_skipped_points": pvt_skipped_points,
        "tm_driver_min_segment_duration_s": TM_DRIVER_MIN_SEGMENT_DURATION_S,
        "first_pvt_segment_duration_s": samples[1].time_s - samples[0].time_s,
        "duration_s": total_duration,
        "max_start_error_rad": start_error,
        "max_goal_error_rad": goal_error,
        "max_excursion_rad": max_excursion,
        "max_sample_step_rad": max_step,
        "max_velocity_rad_s": max_velocity,
        "max_acceleration_rad_s2": max_acceleration,
        "max_derived_velocity_rad_s": max_derived_velocity,
        "max_derived_acceleration_rad_s2": max_derived_acceleration,
        "max_interpolated_velocity_rad_s": max_interpolated_velocity,
        "max_interpolated_acceleration_rad_s2": max_interpolated_acceleration,
        "max_non_target_excursion_rad": max_non_target_excursion,
        "max_live_start_reverse_excursion_rad": (
            max_live_start_reverse_excursion
        ),
        "target_path_length_rad": target_path_length,
        "guard_hard_excursion_rad": hard_excursion_limit,
        "guard_max_velocity_rad_s": hard_velocity_limit,
        "guard_max_acceleration_rad_s2": hard_acceleration_limit,
        "guard_min_duration_s": hard_min_duration,
        "guard_max_duration_s": hard_max_duration,
    }
