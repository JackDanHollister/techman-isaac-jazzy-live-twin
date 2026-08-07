"""Offline-only retiming and validation for the reviewed Watson seven-pin path.

This module deliberately contains no ROS imports, network calls, controller
clients, script command construction, or execution entry point.  It converts
the hash-pinned 300 Hz cuMotion samples into the position/velocity/time points
that survive the installed Techman driver's 25 ms filter, then round-trips each
transmitted numeric field through that exact binary's fixed-five-decimal wire
serialization before proving the resulting six-axis cubic PVT envelope.

The zero-time message seed is omitted by the installed driver.  Consequently,
offline stage records prove only wire cubics between transmitted endpoints.
``validate_live_first_wire_cubic`` is the separate pure function that must be
called with a current six-axis position and velocity before the first wire
endpoint can be treated as validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Any, Mapping, Sequence


ARENA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWED_PLAN = (
    ARENA_DIR
    / "reference/seven_pin/"
    "multi_pin_verticalization_plan.json"
)

EXPECTED_PLAN_SHA256 = (
    "0c5cc66bd54510bc92e92fefc727555abfdf157e541e67755cdccaea565b2c5a"
)
EXPECTED_SOURCE_NUMERIC_SHA256 = (
    "ca8005329f64baf6e205813cece9801ce8f779862dbcd4dc4b0757e1bfcce8bf"
)
EXPECTED_STAGE_COUNT = 49
EXPECTED_SOURCE_SAMPLE_COUNT = 18102
EXPECTED_SPECIMEN_IDS = tuple(range(1, 8))
EXPECTED_STAGE_NAMES = (
    "approach_tilted_pregrasp",
    "descend_tilted_grasp",
    "lift_tilted",
    "reorient_vertical",
    "descend_vertical",
    "retreat_vertical",
    "return_ready",
)
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
READY_JOINT_POSITIONS_RAD = (0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0)

# These are intentionally immutable.  The 1.05 scale leaves at least five
# percent below each nominal acceleration limit after the installed driver's
# point selection and exact cubic-Hermite reconstruction.
GLOBAL_TIME_SCALE = 1.05
DERIVATIVE_LIMIT_FRACTION = 0.95
TM_DRIVER_MIN_SEGMENT_DURATION_S = 0.025
MAX_CONTROLLER_SEGMENT_DURATION_S = 0.060
MAX_ENDPOINT_VELOCITY_RAD_S = 0.005
ROS_DURATION_NANOSECONDS_PER_SECOND = 1_000_000_000

JOINT_POSITION_LOWER_RAD = (
    -2.0 * math.pi,
    -2.0 * math.pi,
    -2.7576202181510405,
    -2.0 * math.pi,
    -2.0 * math.pi,
    -2.0 * math.pi,
)
JOINT_POSITION_UPPER_RAD = (
    2.0 * math.pi,
    2.0 * math.pi,
    2.7576202181510405,
    2.0 * math.pi,
    2.0 * math.pi,
    2.0 * math.pi,
)
JOINT_VELOCITY_LIMITS_RAD_S = (
    3.6651914291880923,
    3.6651914291880923,
    3.6651914291880923,
    3.9269908169872414,
    3.9269908169872414,
    7.853981633974483,
)
JOINT_ACCELERATION_LIMITS_RAD_S2 = (2.0, 2.0, 2.0, 2.5, 2.5, 4.0)

ARTIFACT_KIND = "offline_non_executable_watson_six_axis_pvt_candidate"
ARTIFACT_STATUS = (
    "validated_offline_wire_internal_first_live_cubic_pending_"
    "no_ros_no_command_path"
)
ARTIFACT_DIGEST_FIELD = "artifact_payload_sha256"
ARTIFACT_WARNING = (
    "Offline retimed and fixed-five-decimal wire-numeric evidence only. "
    "Every stage's first physical cubic still requires current q/v validation. "
    "This JSON is not a ROS trajectory, PVT script, controller goal, execution "
    "authorization, or physical safety proof."
)
MAX_SOURCE_PLAN_BYTES = 32 * 1024 * 1024
MAX_DRIVER_COMPONENT_BYTES = 64 * 1024 * 1024

TECHMAN_WORKSPACE = Path(
    os.environ.get(
        "TECHMAN_WORKSPACE",
        str(Path.home() / "tm2_ws_apt"),
    )
).expanduser()
TM_DRIVER_MOVEIT_SOURCE_PATH = Path(
    TECHMAN_WORKSPACE
    / "src/tm2_ros2/tm_driver/src/"
    "tm_ros2_moveit_sct.cpp"
)
TM_DRIVER_MOVEIT_SOURCE_SHA256 = (
    "d8a06da6b95bfea4b8e41415a1186852ce6139ddf4c97def795b6100fcb5fa92"
)
TM_DRIVER_COMMAND_SOURCE_PATH = Path(
    TECHMAN_WORKSPACE / "src/tm2_ros2/tm_driver/src/tm_command.cpp"
)
TM_DRIVER_COMMAND_SOURCE_SHA256 = (
    "2015acc15413f25104928a53790dcbd8e7acfb8190084d90ada1560f48ca180d"
)
TM_DRIVER_COMMAND_HEADER_PATH = Path(
    TECHMAN_WORKSPACE
    / "src/tm2_ros2/tm_driver/include/"
    "tm_driver/tm_command.h"
)
TM_DRIVER_COMMAND_HEADER_SHA256 = (
    "4437dccf1e6f36c4d09b59f02ae5016b5671792fbae8deb9638491f834b273e2"
)
TM_DRIVER_BINARY_PATH = Path(
    TECHMAN_WORKSPACE / "build/tm_driver/tm_driver"
)
TM_DRIVER_INSTALLED_BINARY_PATH = Path(
    TECHMAN_WORKSPACE / "install/tm_driver/lib/tm_driver/tm_driver"
)
TM_DRIVER_BINARY_SHA256 = (
    "6e08f2e7e7a114104a9ec6837796f2da88bb7791c7454b02f0b25071afeb5f15"
)
TM_DRIVER_WIRE_DECIMAL_PLACES = 5
RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0

TM_DRIVER_MOVEIT_REQUIRED_MARKERS = (
    b"get_pvt_traj(traj_points, 0.025)",
    b"point.time = sec(traj_points[i].time_from_start) - "
    b"sec(traj_points[i_1].time_from_start)",
    b"point.time = sec(traj_points[i].time_from_start) - "
    b"sec(traj_points[i_2].time_from_start)",
    b"pvts.points.back() = point",
)
TM_DRIVER_COMMAND_REQUIRED_MARKERS = (
    b"ss << std::fixed << std::setprecision(precision)",
    b"for (auto &value : point.positions) { ss << deg(value) << \",\"; }",
    b"for (auto &value : point.velocities) { ss << deg(value) << \",\"; }",
    b"ss << point.time << \")\\r\\n\"",
)
TM_DRIVER_HEADER_REQUIRED_MARKERS = (
    b"static double deg(double ang) { return (180.0 / M_PI) * ang; }",
    b"static std::string set_pvt_traj(const TmPvtTraj &pvts, int precision = 5)",
)


@dataclass(frozen=True)
class PVTPoint:
    """One absolute-time joint PVT seed or endpoint."""

    source_sample_index: int
    time_s: float
    positions: tuple[float, ...]
    velocities: tuple[float, ...]


@dataclass(frozen=True)
class SerializedWirePoint:
    """One transmitted PVT endpoint after exact fixed-five text round-trip."""

    source_sample_index: int
    segment_duration_text: str
    segment_duration_s: float
    segment_duration_ticks_1e5: int
    cumulative_time_s: float
    cumulative_time_ticks_1e5: int
    position_degrees_text: tuple[str, ...]
    velocity_degrees_s_text: tuple[str, ...]
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_digest(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    copy.pop(ARTIFACT_DIGEST_FIELD, None)
    return hashlib.sha256(
        json.dumps(
            copy,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _float64_sha256(vectors: Sequence[Sequence[float]]) -> str:
    digest = hashlib.sha256()
    for vector in vectors:
        values = tuple(float(value) for value in vector)
        digest.update(struct.pack(f"<{len(values)}d", *values))
    return digest.hexdigest()


def _points_float64_sha256(points: Sequence[PVTPoint]) -> str:
    digest = hashlib.sha256()
    for point in points:
        digest.update(struct.pack("<q", point.source_sample_index))
        digest.update(
            struct.pack(
                "<13d",
                point.time_s,
                *point.positions,
                *point.velocities,
            )
        )
    return digest.hexdigest()


def _finite_vector(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != len(JOINT_NAMES):
        raise ValueError(f"{label} must contain six values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _read_regular_file(path: Path, maximum_bytes: int) -> bytes:
    path = path.expanduser()
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Reviewed plan must be a regular non-symlink file: {path}")
    if before.st_size <= 0 or before.st_size > maximum_bytes:
        raise ValueError(f"Reviewed plan has an invalid file size: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Reviewed plan changed while it was opened")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) > maximum_bytes:
        raise ValueError("Reviewed plan exceeds the maximum accepted size")
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        raise ValueError("Reviewed plan changed while it was read")
    return data


def _verified_component(
    *,
    label: str,
    path: Path,
    expected_sha256: str,
    required_markers: Sequence[bytes] = (),
    require_executable: bool = False,
    record_path: str | None = None,
) -> dict[str, Any]:
    normalized = _absolute_path_without_following_final_symlink(path)
    data = _read_regular_file(normalized, MAX_DRIVER_COMPONENT_BYTES)
    observed_sha256 = sha256_bytes(data)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Installed {label} SHA-256 mismatch: "
            f"{observed_sha256} != {expected_sha256}"
        )
    for marker in required_markers:
        if marker not in data:
            raise ValueError(
                f"Installed {label} no longer contains a pinned serializer marker"
            )
    if require_executable and not os.access(normalized, os.X_OK):
        raise ValueError(f"Installed {label} is not executable: {normalized}")
    return {
        "path": str(normalized) if record_path is None else record_path,
        "sha256": observed_sha256,
        "size_bytes": len(data),
        "regular_non_symlink": True,
        "executable": require_executable,
    }


def verify_installed_tm_driver_provenance() -> dict[str, Any]:
    """Verify the exact installed filter, serializer, defaults, and binary.

    Hash equality is the authority.  Marker checks make the pinned semantics
    human-auditable and fail closed if a constant is accidentally paired with
    the wrong component.
    """

    install_link = TM_DRIVER_INSTALLED_BINARY_PATH.expanduser()
    if not install_link.is_symlink():
        raise ValueError(
            "Installed tm_driver entry must remain the reviewed symlink: "
            f"{install_link}"
        )
    installed_target = install_link.resolve(strict=True)
    reviewed_binary = TM_DRIVER_BINARY_PATH.resolve(strict=True)
    if installed_target != reviewed_binary:
        raise ValueError(
            "Installed tm_driver symlink target changed: "
            f"{installed_target} != {reviewed_binary}"
        )

    moveit = _verified_component(
        label="tm_driver MoveIt PVT source",
        path=TM_DRIVER_MOVEIT_SOURCE_PATH,
        expected_sha256=TM_DRIVER_MOVEIT_SOURCE_SHA256,
        required_markers=TM_DRIVER_MOVEIT_REQUIRED_MARKERS,
        record_path="src/tm2_ros2/tm_driver/src/tm_ros2_moveit_sct.cpp",
    )
    command = _verified_component(
        label="tm_driver PVT serializer source",
        path=TM_DRIVER_COMMAND_SOURCE_PATH,
        expected_sha256=TM_DRIVER_COMMAND_SOURCE_SHA256,
        required_markers=TM_DRIVER_COMMAND_REQUIRED_MARKERS,
        record_path="src/tm2_ros2/tm_driver/src/tm_command.cpp",
    )
    header = _verified_component(
        label="tm_driver PVT serializer header",
        path=TM_DRIVER_COMMAND_HEADER_PATH,
        expected_sha256=TM_DRIVER_COMMAND_HEADER_SHA256,
        required_markers=TM_DRIVER_HEADER_REQUIRED_MARKERS,
        record_path="src/tm2_ros2/tm_driver/include/tm_driver/tm_command.h",
    )
    binary = _verified_component(
        label="tm_driver binary",
        path=TM_DRIVER_BINARY_PATH,
        expected_sha256=TM_DRIVER_BINARY_SHA256,
        require_executable=True,
        record_path="build/tm_driver/tm_driver",
    )
    binary["installed_symlink_path"] = (
        "install/tm_driver/lib/tm_driver/tm_driver"
    )
    binary["installed_symlink_target"] = "build/tm_driver/tm_driver"
    return {
        "verification_status": (
            "exact_installed_sources_header_binary_and_semantics_match"
        ),
        "moveit_filter_source": moveit,
        "pvt_serializer_source": command,
        "pvt_serializer_header": header,
        "tm_driver_binary": binary,
        "verified_semantics": {
            "filter_minimum_segment_seconds": (
                TM_DRIVER_MIN_SEGMENT_DURATION_S
            ),
            "filter_happens_before_wire_formatting": True,
            "zero_time_message_seed_transmitted": False,
            "short_final_replaces_previous_transmitted_endpoint": True,
            "joint_position_input_unit": "radian",
            "joint_velocity_input_unit": "radian_per_second",
            "wire_joint_position_unit": "degree",
            "wire_joint_velocity_unit": "degree_per_second",
            "conversion_expression": "(180.0 / M_PI) * value",
            "numeric_format": "std_fixed",
            "decimal_places": TM_DRIVER_WIRE_DECIMAL_PLACES,
            "decimal_separator": ".",
            "wire_time_kind": "relative_segment_seconds",
            "wire_time_accumulation_for_proof": (
                "sum_each_independently_parsed_fixed_5_segment"
            ),
            "total_time_serialized": False,
        },
    }


def _absolute_path_without_following_final_symlink(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.parent.resolve() / expanded.name


def load_reviewed_plan(path: Path = DEFAULT_REVIEWED_PLAN) -> dict[str, Any]:
    """Load only the exact reviewed source-plan bytes and validate their content."""

    normalized_path = _absolute_path_without_following_final_symlink(path)
    data = _read_regular_file(normalized_path, MAX_SOURCE_PLAN_BYTES)
    observed_hash = sha256_bytes(data)
    if observed_hash != EXPECTED_PLAN_SHA256:
        raise ValueError(
            "Reviewed seven-pin plan SHA-256 mismatch: "
            f"{observed_hash} != {EXPECTED_PLAN_SHA256}"
        )
    try:
        plan = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Reviewed seven-pin plan is not valid UTF-8 JSON") from exc
    if not isinstance(plan, dict):
        raise ValueError("Reviewed seven-pin plan must be a JSON object")
    validate_reviewed_plan(plan)
    return plan


def _source_stages(plan: dict[str, Any]):
    for specimen in plan["specimens"]:
        specimen_id = int(specimen["specimen_id"])
        for stage_index, stage in enumerate(specimen["stages"]):
            yield specimen_id, stage_index, stage


def validate_reviewed_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Recompute the reviewed plan's hashes, continuity, and acceptance gates."""

    if plan.get("format_version") != 1 or plan.get("frame_id") != "base":
        raise ValueError("Reviewed plan format or frame changed")
    if plan.get("planning_tool_frame") != "pin_grasp_tcp":
        raise ValueError("Reviewed plan tool frame changed")
    for field in ("ros_used", "watson_connected", "real_robot_commanded"):
        if plan.get(field) is not False:
            raise ValueError(f"Reviewed plan must keep {field} false")
    if float(plan.get("control_dt_seconds", 0.0)) != 1.0 / 300.0:
        raise ValueError("Reviewed plan is no longer the 300 Hz source")
    if tuple(plan.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("Reviewed plan joint order changed")
    ready = _finite_vector(plan.get("ready_joint_positions"), "ready pose")
    if ready != READY_JOINT_POSITIONS_RAD:
        raise ValueError("Reviewed plan ready pose changed")
    if tuple(plan.get("specimen_ids", ())) != EXPECTED_SPECIMEN_IDS:
        raise ValueError("Reviewed plan specimen order changed")
    specimens = plan.get("specimens")
    if not isinstance(specimens, list) or len(specimens) != len(EXPECTED_SPECIMEN_IDS):
        raise ValueError("Reviewed plan must contain exactly seven specimens")

    top_validation = plan.get("validation")
    if not isinstance(top_validation, dict):
        raise ValueError("Reviewed plan is missing validation evidence")
    required_clearance = float(plan.get("required_sampled_sphere_clearance_m", 0.0))
    if (
        top_validation.get("all_stages_accepted") is not True
        or top_validation.get("sampled_self_collision") is not False
        or top_validation.get("derivative_limits_met") is not True
        or top_validation.get("cycles_ready_to_ready") is not True
        or top_validation.get("adjacent_stage_positions_continuous") is not True
        or required_clearance != 0.004
        or float(top_validation.get("minimum_sampled_sphere_clearance_m", 0.0))
        < required_clearance
    ):
        raise ValueError("Reviewed plan top-level acceptance gates failed")

    position_rows: list[tuple[float, ...]] = []
    velocity_rows: list[tuple[float, ...]] = []
    stage_count = 0
    sample_count = 0
    source_duration_s = 0.0
    maximum_source_step = 0.0
    previous = ready
    for expected_specimen_id, specimen in zip(EXPECTED_SPECIMEN_IDS, specimens):
        if specimen.get("specimen_id") != expected_specimen_id:
            raise ValueError("Reviewed plan specimen IDs changed")
        stages = specimen.get("stages")
        if not isinstance(stages, list) or tuple(
            stage.get("name") for stage in stages
        ) != EXPECTED_STAGE_NAMES:
            raise ValueError(
                f"Specimen {expected_specimen_id} stage order changed"
            )
        for stage in stages:
            stage_count += 1
            validation = stage.get("trajectory_validation")
            if not isinstance(validation, dict):
                raise ValueError("Reviewed stage is missing trajectory validation")
            if (
                stage.get("accepted") is not True
                or stage.get("path_found") is not True
                or stage.get("goal_tolerance_met") is not True
                or validation.get("sampled_self_collision") is not False
                or validation.get("derivative_limits_met") is not True
                or float(
                    validation.get("minimum_sampled_sphere_clearance_m", 0.0)
                )
                < required_clearance
            ):
                raise ValueError("Reviewed plan contains a failed stage gate")
            if tuple(validation.get("velocity_limits_rad_s", ())) != (
                JOINT_VELOCITY_LIMITS_RAD_S
            ):
                raise ValueError("Reviewed plan velocity limits changed")
            if tuple(validation.get("acceleration_limits_rad_s2", ())) != (
                JOINT_ACCELERATION_LIMITS_RAD_S2
            ):
                raise ValueError("Reviewed plan acceleration limits changed")

            start = _finite_vector(
                stage.get("start_joint_positions"), "stage start"
            )
            end = _finite_vector(stage.get("end_joint_positions"), "stage end")
            if start != previous:
                raise ValueError("Reviewed plan lost exact stage continuity")
            samples = stage.get("control_samples")
            if not isinstance(samples, list) or len(samples) < 2:
                raise ValueError("Reviewed stage has too few control samples")
            stage_positions: list[tuple[float, ...]] = []
            stage_velocities: list[tuple[float, ...]] = []
            prior_time = -1.0
            for sample_index, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    raise ValueError("Reviewed control sample must be an object")
                time_s = float(sample.get("time_seconds", math.nan))
                if (
                    not math.isfinite(time_s)
                    or time_s < 0.0
                    or time_s <= prior_time
                ):
                    raise ValueError(
                        "Reviewed stage sample times must be strictly increasing"
                    )
                if sample_index == 0 and time_s != 0.0:
                    raise ValueError("Reviewed stage must start at time zero")
                position = _finite_vector(
                    sample.get("joint_positions"), "source sample positions"
                )
                velocity = _finite_vector(
                    sample.get("joint_velocities"), "source sample velocities"
                )
                if stage_positions:
                    maximum_source_step = max(
                        maximum_source_step,
                        max(
                            abs(position[joint] - stage_positions[-1][joint])
                            for joint in range(len(JOINT_NAMES))
                        ),
                    )
                stage_positions.append(position)
                stage_velocities.append(velocity)
                prior_time = time_s
            if stage_positions[0] != start or stage_positions[-1] != end:
                raise ValueError("Reviewed stage sample endpoints changed")
            if any(stage_velocities[0]) or any(stage_velocities[-1]):
                raise ValueError("Reviewed stage endpoint velocities must be zero")
            stage_hash = _float64_sha256(stage_positions + stage_velocities)
            if stage_hash != stage.get("control_samples_float64_sha256"):
                raise ValueError("Reviewed stage numeric hash mismatch")
            position_rows.extend(stage_positions)
            velocity_rows.extend(stage_velocities)
            sample_count += len(samples)
            source_duration_s += prior_time
            previous = end
        if previous != ready:
            raise ValueError(
                f"Specimen {expected_specimen_id} no longer returns exactly ready"
            )

    aggregate_hash = _float64_sha256(position_rows + velocity_rows)
    if (
        stage_count != EXPECTED_STAGE_COUNT
        or sample_count != EXPECTED_SOURCE_SAMPLE_COUNT
        or top_validation.get("stage_count") != EXPECTED_STAGE_COUNT
        or top_validation.get("control_sample_count")
        != EXPECTED_SOURCE_SAMPLE_COUNT
    ):
        raise ValueError("Reviewed stage or sample count changed")
    if (
        aggregate_hash != EXPECTED_SOURCE_NUMERIC_SHA256
        or top_validation.get("control_samples_float64_sha256")
        != EXPECTED_SOURCE_NUMERIC_SHA256
    ):
        raise ValueError("Reviewed aggregate numeric hash mismatch")
    if not math.isclose(
        maximum_source_step,
        float(top_validation.get("maximum_observed_control_step_rad", math.nan)),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Reviewed maximum control step evidence changed")
    return {
        "stage_count": stage_count,
        "source_sample_count": sample_count,
        "source_stage_motion_duration_seconds": source_duration_s,
        "maximum_source_control_step_rad": maximum_source_step,
        "source_numeric_sha256": aggregate_hash,
    }


def _validate_point_sequence(points: Sequence[PVTPoint]) -> None:
    if len(points) < 2:
        raise ValueError("PVT trajectory must contain at least two points")
    prior_time = -1.0
    prior_source_index = -1
    for point_index, point in enumerate(points):
        if not isinstance(point.source_sample_index, int):
            raise ValueError("PVT source sample index must be an integer")
        if point.source_sample_index <= prior_source_index:
            raise ValueError("PVT source sample indices must be strictly increasing")
        if not math.isfinite(point.time_s) or point.time_s < 0.0:
            raise ValueError("PVT time must be finite and non-negative")
        if point.time_s <= prior_time:
            raise ValueError("PVT times must be strictly increasing")
        if point_index == 0 and point.time_s != 0.0:
            raise ValueError("PVT first point must have time_from_start zero")
        if len(point.positions) != len(JOINT_NAMES) or len(point.velocities) != len(
            JOINT_NAMES
        ):
            raise ValueError("PVT point must contain six positions and velocities")
        if not all(
            math.isfinite(value)
            for value in point.positions + point.velocities
        ):
            raise ValueError("PVT point contains a non-finite value")
        prior_time = point.time_s
        prior_source_index = point.source_sample_index


def emulate_tm_driver_selection(
    points: Sequence[PVTPoint],
) -> tuple[tuple[PVTPoint, ...], int]:
    """Mirror the installed driver's zero-point omission and 25 ms selection.

    The returned sequence retains the zero-time seed for offline cubic proof.
    Every later returned point is exactly one endpoint that ``tm_driver`` would
    transmit.  ``skipped`` matches the driver's skipped source-point count.
    """

    planned = tuple(points)
    _validate_point_sequence(planned)
    transmitted_indices: list[int] = []
    previous_selected_index = 0
    second_previous_selected_index = 0
    for index in range(1, len(planned) - 1):
        segment_duration = (
            planned[index].time_s - planned[previous_selected_index].time_s
        )
        if segment_duration >= TM_DRIVER_MIN_SEGMENT_DURATION_S:
            second_previous_selected_index = previous_selected_index
            previous_selected_index = index
            transmitted_indices.append(index)

    last_index = len(planned) - 1
    last_duration = (
        planned[last_index].time_s - planned[previous_selected_index].time_s
    )
    if last_duration >= TM_DRIVER_MIN_SEGMENT_DURATION_S:
        transmitted_indices.append(last_index)
    else:
        if not transmitted_indices:
            raise ValueError(
                "tm_driver would have no prior PVT endpoint to replace for "
                "the short final segment"
            )
        replacement_duration = (
            planned[last_index].time_s
            - planned[second_previous_selected_index].time_s
        )
        if replacement_duration < TM_DRIVER_MIN_SEGMENT_DURATION_S:
            raise ValueError("tm_driver replacement PVT segment is too short")
        transmitted_indices[-1] = last_index

    selected = (planned[0],) + tuple(planned[index] for index in transmitted_indices)
    skipped = len(planned) - len(selected)
    return selected, skipped


def _cpp_fixed_5(value: float) -> str:
    """Match ``std::fixed << std::setprecision(5)`` for finite C-locale values."""

    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("tm_driver wire scalar must be finite")
    text = format(converted, f".{TM_DRIVER_WIRE_DECIMAL_PLACES}f")
    if "." not in text or len(text.rsplit(".", maxsplit=1)[1]) != (
        TM_DRIVER_WIRE_DECIMAL_PLACES
    ):
        raise AssertionError("Fixed-five wire formatter contract failed")
    return text


def _fixed_5_ticks(text: str) -> int:
    if not isinstance(text, str):
        raise ValueError("Fixed-five wire value must be text")
    sign = -1 if text.startswith("-") else 1
    unsigned = text[1:] if text[:1] in ("-", "+") else text
    whole, separator, fraction = unsigned.partition(".")
    if (
        separator != "."
        or not whole.isdigit()
        or len(fraction) != TM_DRIVER_WIRE_DECIMAL_PLACES
        or not fraction.isdigit()
    ):
        raise ValueError("Wire value is not exact fixed-five decimal text")
    return sign * (
        int(whole) * (10**TM_DRIVER_WIRE_DECIMAL_PLACES) + int(fraction)
    )


def _rad_to_wire_text_and_back(value_rad: float) -> tuple[str, float]:
    text = _cpp_fixed_5(float(value_rad) * RAD_TO_DEG)
    return text, float(text) * DEG_TO_RAD


def serialize_filtered_message_points_to_wire(
    message_points: Sequence[PVTPoint],
) -> tuple[SerializedWirePoint, ...]:
    """Round-trip the points that the reviewed installed binary would transmit.

    ``message_points`` includes the zero-time message seed followed only by
    endpoints that survived ``get_pvt_traj(..., 0.025)``.  The returned tuple
    excludes that seed because the installed driver excludes it from the PVT
    script.
    """

    filtered = tuple(message_points)
    _validate_point_sequence(filtered)
    reapplied, skipped = emulate_tm_driver_selection(filtered)
    if skipped != 0 or reapplied != filtered:
        raise ValueError(
            "Message points would still be changed by the installed 25 ms filter"
        )

    cumulative_ticks = 0
    wire_points: list[SerializedWirePoint] = []
    for point_index in range(1, len(filtered)):
        previous = filtered[point_index - 1]
        point = filtered[point_index]
        segment_text = _cpp_fixed_5(point.time_s - previous.time_s)
        segment_ticks = _fixed_5_ticks(segment_text)
        if segment_ticks <= 0:
            raise ValueError("Serialized tm_driver segment duration is not positive")
        cumulative_ticks += segment_ticks

        position_text: list[str] = []
        velocity_text: list[str] = []
        positions_rad: list[float] = []
        velocities_rad_s: list[float] = []
        for value in point.positions:
            text, round_tripped = _rad_to_wire_text_and_back(value)
            position_text.append(text)
            positions_rad.append(round_tripped)
        for value in point.velocities:
            text, round_tripped = _rad_to_wire_text_and_back(value)
            velocity_text.append(text)
            velocities_rad_s.append(round_tripped)
        wire_points.append(
            SerializedWirePoint(
                source_sample_index=point.source_sample_index,
                segment_duration_text=segment_text,
                segment_duration_s=float(segment_text),
                segment_duration_ticks_1e5=segment_ticks,
                cumulative_time_s=(
                    cumulative_ticks / (10**TM_DRIVER_WIRE_DECIMAL_PLACES)
                ),
                cumulative_time_ticks_1e5=cumulative_ticks,
                position_degrees_text=tuple(position_text),
                velocity_degrees_s_text=tuple(velocity_text),
                positions_rad=tuple(positions_rad),
                velocities_rad_s=tuple(velocities_rad_s),
            )
        )
    if not wire_points:
        raise ValueError("Installed tm_driver would transmit no PVT endpoint")
    return tuple(wire_points)


def _serialized_wire_tokens_sha256(
    wire_points: Sequence[SerializedWirePoint],
) -> str:
    digest = hashlib.sha256()
    for point in wire_points:
        digest.update(struct.pack("<q", point.source_sample_index))
        for token in (
            *point.position_degrees_text,
            *point.velocity_degrees_s_text,
            point.segment_duration_text,
        ):
            encoded = token.encode("ascii")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()


def _wire_numeric_points_sha256(
    wire_points: Sequence[SerializedWirePoint],
) -> str:
    points = tuple(
        PVTPoint(
            source_sample_index=point.source_sample_index,
            time_s=point.cumulative_time_s,
            positions=point.positions_rad,
            velocities=point.velocities_rad_s,
        )
        for point in wire_points
    )
    return _points_float64_sha256(points)


def _real_roots_in_unit_interval(
    a: float, b: float, c: float
) -> tuple[float, ...]:
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
    root_term = math.sqrt(max(discriminant, 0.0))
    for root in (
        (-b - root_term) / (2.0 * a),
        (-b + root_term) / (2.0 * a),
    ):
        if 0.0 <= root <= 1.0 and not any(
            abs(root - seen) <= epsilon for seen in roots
        ):
            roots.append(root)
    return tuple(roots)


def hermite_segment_extrema(
    q0: float,
    q1: float,
    v0: float,
    v1: float,
    duration_s: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return exact position, velocity, and acceleration extrema candidates."""

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("Hermite segment duration must be finite and positive")
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
        tuple(position(value) for value in position_parameters),
        tuple(velocity(value) for value in velocity_parameters),
        (acceleration(0.0), acceleration(1.0)),
    )


def _validate_cubic_sequence(
    points: Sequence[PVTPoint],
    *,
    proof_domain: str,
    segment_durations_s: Sequence[float] | None = None,
    require_filter_stable: bool,
    enforce_stage_endpoint_velocity: bool,
    wire_serialization_included: bool,
    live_first_seed_included: bool,
) -> dict[str, Any]:
    candidate = tuple(points)
    _validate_point_sequence(candidate)
    post_filter_skipped: int | None = None
    if require_filter_stable:
        post_filter, post_filter_skipped = emulate_tm_driver_selection(candidate)
        if post_filter_skipped != 0 or post_filter != candidate:
            raise ValueError(
                "Retimed candidate would still be changed by the tm_driver filter"
            )
    if segment_durations_s is None:
        durations = tuple(
            candidate[index].time_s - candidate[index - 1].time_s
            for index in range(1, len(candidate))
        )
    else:
        durations = tuple(float(value) for value in segment_durations_s)
        if len(durations) != len(candidate) - 1 or not all(
            math.isfinite(value) and value > 0.0 for value in durations
        ):
            raise ValueError(
                "Explicit PVT segment durations must be finite, positive, "
                "and match the point count"
            )
        if not math.isclose(
            math.fsum(durations),
            candidate[-1].time_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Explicit PVT segment durations disagree with cumulative time"
            )

    endpoint_velocity = max(
        max(abs(value) for value in candidate[0].velocities),
        max(abs(value) for value in candidate[-1].velocities),
    )
    if (
        enforce_stage_endpoint_velocity
        and endpoint_velocity > MAX_ENDPOINT_VELOCITY_RAD_S
    ):
        raise ValueError(
            f"Retimed endpoint velocity {endpoint_velocity:.9f}rad/s exceeds "
            f"{MAX_ENDPOINT_VELOCITY_RAD_S:.9f}rad/s"
        )

    position_min = [math.inf] * len(JOINT_NAMES)
    position_max = [-math.inf] * len(JOINT_NAMES)
    maximum_velocity = [0.0] * len(JOINT_NAMES)
    maximum_acceleration = [0.0] * len(JOINT_NAMES)
    maximum_step = [0.0] * len(JOINT_NAMES)
    minimum_segment_duration = math.inf
    maximum_segment_duration = 0.0
    for point_index in range(1, len(candidate)):
        previous = candidate[point_index - 1]
        point = candidate[point_index]
        duration = durations[point_index - 1]
        if duration < TM_DRIVER_MIN_SEGMENT_DURATION_S:
            raise ValueError(
                f"PVT segment {point_index} duration {duration:.9f}s is below "
                f"{TM_DRIVER_MIN_SEGMENT_DURATION_S:.9f}s"
            )
        if duration > MAX_CONTROLLER_SEGMENT_DURATION_S:
            raise ValueError(
                f"PVT segment {point_index} duration {duration:.9f}s exceeds "
                f"{MAX_CONTROLLER_SEGMENT_DURATION_S:.9f}s"
            )
        minimum_segment_duration = min(minimum_segment_duration, duration)
        maximum_segment_duration = max(maximum_segment_duration, duration)
        for joint in range(len(JOINT_NAMES)):
            positions, velocities, accelerations = hermite_segment_extrema(
                previous.positions[joint],
                point.positions[joint],
                previous.velocities[joint],
                point.velocities[joint],
                duration,
            )
            position_min[joint] = min(position_min[joint], *positions)
            position_max[joint] = max(position_max[joint], *positions)
            maximum_velocity[joint] = max(
                maximum_velocity[joint], *(abs(value) for value in velocities)
            )
            maximum_acceleration[joint] = max(
                maximum_acceleration[joint],
                *(abs(value) for value in accelerations),
            )
            maximum_step[joint] = max(
                maximum_step[joint],
                abs(point.positions[joint] - previous.positions[joint]),
            )

    for joint, joint_name in enumerate(JOINT_NAMES):
        if (
            position_min[joint] < JOINT_POSITION_LOWER_RAD[joint] - 1e-12
            or position_max[joint] > JOINT_POSITION_UPPER_RAD[joint] + 1e-12
        ):
            raise ValueError(f"{joint_name} cubic position exceeds its hard limit")
        velocity_cap = (
            JOINT_VELOCITY_LIMITS_RAD_S[joint] * DERIVATIVE_LIMIT_FRACTION
        )
        acceleration_cap = (
            JOINT_ACCELERATION_LIMITS_RAD_S2[joint]
            * DERIVATIVE_LIMIT_FRACTION
        )
        if maximum_velocity[joint] > velocity_cap + 1e-12:
            raise ValueError(
                f"{joint_name} cubic velocity {maximum_velocity[joint]:.9f}rad/s "
                f"exceeds immutable margin cap {velocity_cap:.9f}rad/s"
            )
        if maximum_acceleration[joint] > acceleration_cap + 1e-12:
            raise ValueError(
                f"{joint_name} cubic acceleration "
                f"{maximum_acceleration[joint]:.9f}rad/s^2 exceeds immutable "
                f"margin cap {acceleration_cap:.9f}rad/s^2"
            )

    return {
        "proof_domain": proof_domain,
        "wire_serialization_included": wire_serialization_included,
        "live_first_seed_included": live_first_seed_included,
        "physical_wire_proof_claimed": (
            wire_serialization_included and live_first_seed_included
        ),
        "point_count_including_zero_seed": len(candidate),
        "driver_transmitted_point_count": len(candidate) - 1,
        "post_candidate_filter_skipped_points": post_filter_skipped,
        "duration_seconds": math.fsum(durations),
        "minimum_segment_duration_seconds": minimum_segment_duration,
        "maximum_segment_duration_seconds": maximum_segment_duration,
        "position_minimum_rad": position_min,
        "position_maximum_rad": position_max,
        "maximum_endpoint_step_rad": maximum_step,
        "maximum_cubic_velocity_rad_s": maximum_velocity,
        "maximum_cubic_acceleration_rad_s2": maximum_acceleration,
        "maximum_velocity_limit_utilization": max(
            maximum_velocity[joint] / JOINT_VELOCITY_LIMITS_RAD_S[joint]
            for joint in range(len(JOINT_NAMES))
        ),
        "maximum_acceleration_limit_utilization": max(
            maximum_acceleration[joint]
            / JOINT_ACCELERATION_LIMITS_RAD_S2[joint]
            for joint in range(len(JOINT_NAMES))
        ),
        "maximum_endpoint_velocity_rad_s": endpoint_velocity,
        "retimed_points_float64_sha256": _points_float64_sha256(candidate),
    }


def validate_six_axis_pvt(points: Sequence[PVTPoint]) -> dict[str, Any]:
    """Validate message-level PVT points, explicitly before wire rounding.

    These metrics remain useful for diagnosing the retimer and filter, but
    ``physical_wire_proof_claimed`` is always false.  Physical claims must use
    the fixed-five wire points plus a separately validated live first cubic.
    """

    return _validate_cubic_sequence(
        points,
        proof_domain="message_points_pre_fixed_5_wire_diagnostic_only",
        require_filter_stable=True,
        enforce_stage_endpoint_velocity=True,
        wire_serialization_included=False,
        live_first_seed_included=False,
    )


def _validate_wire_internal_cubics(
    wire_points: Sequence[SerializedWirePoint],
) -> dict[str, Any]:
    """Validate all wire cubics after the first live-seeded segment."""

    transmitted = tuple(wire_points)
    if not transmitted:
        raise ValueError("Wire proof requires at least one transmitted endpoint")
    if len(transmitted) == 1:
        return {
            "proof_domain": (
                "fixed_5_wire_internal_cubics_excluding_first_live_seed"
            ),
            "status": "no_internal_cubic_only_first_live_cubic_pending",
            "wire_serialization_included": True,
            "live_first_seed_included": False,
            "physical_wire_proof_claimed": False,
            "wire_endpoint_count": 1,
            "internal_cubic_count": 0,
            "post_candidate_filter_skipped_points": 0,
        }

    first_ticks = transmitted[0].cumulative_time_ticks_1e5
    points = tuple(
        PVTPoint(
            source_sample_index=point.source_sample_index,
            time_s=(
                (point.cumulative_time_ticks_1e5 - first_ticks)
                / (10**TM_DRIVER_WIRE_DECIMAL_PLACES)
            ),
            positions=point.positions_rad,
            velocities=point.velocities_rad_s,
        )
        for point in transmitted
    )
    metrics = _validate_cubic_sequence(
        points,
        proof_domain="fixed_5_wire_internal_cubics_excluding_first_live_seed",
        segment_durations_s=tuple(
            point.segment_duration_s for point in transmitted[1:]
        ),
        require_filter_stable=True,
        enforce_stage_endpoint_velocity=False,
        wire_serialization_included=True,
        live_first_seed_included=False,
    )
    metrics.update(
        {
            "status": "validated_internal_wire_cubics_first_live_cubic_pending",
            "wire_endpoint_count": len(transmitted),
            "internal_cubic_count": len(transmitted) - 1,
        }
    )
    return metrics


def _scaled_points(points: Sequence[PVTPoint]) -> tuple[PVTPoint, ...]:
    scaled: list[PVTPoint] = []
    for point in points:
        scaled_time_nanoseconds = int(
            round(
                point.time_s
                * GLOBAL_TIME_SCALE
                * ROS_DURATION_NANOSECONDS_PER_SECOND
            )
        )
        scaled.append(
            PVTPoint(
                source_sample_index=point.source_sample_index,
                time_s=(
                    scaled_time_nanoseconds
                    / ROS_DURATION_NANOSECONDS_PER_SECOND
                ),
                positions=point.positions,
                velocities=tuple(
                    value / GLOBAL_TIME_SCALE for value in point.velocities
                ),
            )
        )
    return tuple(scaled)


def _unscaled_control_points(
    control_samples: Sequence[dict[str, Any]],
) -> tuple[PVTPoint, ...]:
    points: list[PVTPoint] = []
    for source_index, sample in enumerate(control_samples):
        if not isinstance(sample, dict):
            raise ValueError("Ingress control sample must be an object")
        time_value = sample.get(
            "time_seconds", sample.get("time_from_start_seconds")
        )
        positions = sample.get(
            "joint_positions", sample.get("joint_positions_rad")
        )
        velocities = sample.get(
            "joint_velocities", sample.get("joint_velocities_rad_s")
        )
        try:
            time_s = float(time_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Ingress control sample time must be a finite number"
            ) from exc
        points.append(
            PVTPoint(
                source_sample_index=source_index,
                time_s=time_s,
                positions=_finite_vector(positions, "source positions"),
                velocities=_finite_vector(velocities, "source velocities"),
            )
        )
    return tuple(points)


def _source_points(stage: dict[str, Any]) -> tuple[PVTPoint, ...]:
    return _scaled_points(_unscaled_control_points(stage["control_samples"]))


def _point_record(
    point: PVTPoint, previous: PVTPoint | None
) -> dict[str, Any]:
    time_nanoseconds = int(
        round(point.time_s * ROS_DURATION_NANOSECONDS_PER_SECOND)
    )
    if (
        time_nanoseconds / ROS_DURATION_NANOSECONDS_PER_SECOND
        != point.time_s
    ):
        raise ValueError("Retimed point is not representable as a ROS duration")
    return {
        "source_sample_index": point.source_sample_index,
        "time_from_start_seconds": point.time_s,
        "time_from_start_nanoseconds": time_nanoseconds,
        "segment_duration_seconds": (
            0.0 if previous is None else point.time_s - previous.time_s
        ),
        "joint_positions_rad": list(point.positions),
        "joint_velocities_rad_s": list(point.velocities),
    }


def _point_from_record(record: dict[str, Any]) -> PVTPoint:
    if not isinstance(record, dict):
        raise ValueError("Retimed point record must be an object")
    source_sample_index = record.get("source_sample_index")
    if not isinstance(source_sample_index, int):
        raise ValueError("Retimed point source index must be an integer")
    time_nanoseconds = record.get("time_from_start_nanoseconds")
    if not isinstance(time_nanoseconds, int) or time_nanoseconds < 0:
        raise ValueError(
            "Retimed point time_from_start_nanoseconds must be a non-negative integer"
        )
    time_s = float(record.get("time_from_start_seconds", math.nan))
    if (
        time_nanoseconds / ROS_DURATION_NANOSECONDS_PER_SECOND
        != time_s
    ):
        raise ValueError("Retimed seconds and nanoseconds disagree")
    return PVTPoint(
        source_sample_index=source_sample_index,
        time_s=time_s,
        positions=_finite_vector(
            record.get("joint_positions_rad"), "retimed positions"
        ),
        velocities=_finite_vector(
            record.get("joint_velocities_rad_s"), "retimed velocities"
        ),
    )


def _wire_point_record(point: SerializedWirePoint) -> dict[str, Any]:
    return {
        "source_sample_index": point.source_sample_index,
        "wire_segment_duration_seconds_fixed_5": point.segment_duration_text,
        "wire_segment_duration_seconds": point.segment_duration_s,
        "wire_segment_duration_ticks_1e5": point.segment_duration_ticks_1e5,
        "cumulative_wire_time_seconds": point.cumulative_time_s,
        "cumulative_wire_time_ticks_1e5": point.cumulative_time_ticks_1e5,
        "joint_positions_degrees_fixed_5": list(point.position_degrees_text),
        "joint_velocities_degrees_per_second_fixed_5": list(
            point.velocity_degrees_s_text
        ),
        "joint_positions_rad_after_wire_roundtrip": list(point.positions_rad),
        "joint_velocities_rad_s_after_wire_roundtrip": list(
            point.velocities_rad_s
        ),
    }


def _fixed_5_vector(
    value: Any,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != len(JOINT_NAMES):
        raise ValueError(f"{label} must contain six fixed-five text values")
    result = tuple(value)
    for item in result:
        _fixed_5_ticks(item)
    return result


def _wire_point_from_record(record: Mapping[str, Any]) -> SerializedWirePoint:
    if not isinstance(record, Mapping):
        raise ValueError("Serialized wire point record must be an object")
    source_sample_index = record.get("source_sample_index")
    if not isinstance(source_sample_index, int) or source_sample_index < 1:
        raise ValueError("Serialized wire point source index is invalid")

    segment_text = record.get("wire_segment_duration_seconds_fixed_5")
    segment_ticks = _fixed_5_ticks(segment_text)
    if segment_ticks <= 0:
        raise ValueError("Serialized wire point duration must be positive")
    if record.get("wire_segment_duration_ticks_1e5") != segment_ticks:
        raise ValueError("Serialized wire point duration ticks changed")
    segment_duration_s = float(segment_text)
    if record.get("wire_segment_duration_seconds") != segment_duration_s:
        raise ValueError("Serialized wire point duration numeric value changed")

    cumulative_ticks = record.get("cumulative_wire_time_ticks_1e5")
    if not isinstance(cumulative_ticks, int) or cumulative_ticks < segment_ticks:
        raise ValueError("Serialized wire point cumulative ticks are invalid")
    cumulative_time_s = cumulative_ticks / (
        10**TM_DRIVER_WIRE_DECIMAL_PLACES
    )
    if record.get("cumulative_wire_time_seconds") != cumulative_time_s:
        raise ValueError("Serialized wire point cumulative time changed")

    position_text = _fixed_5_vector(
        record.get("joint_positions_degrees_fixed_5"),
        "serialized wire positions",
    )
    velocity_text = _fixed_5_vector(
        record.get("joint_velocities_degrees_per_second_fixed_5"),
        "serialized wire velocities",
    )
    positions_rad = tuple(float(text) * DEG_TO_RAD for text in position_text)
    velocities_rad_s = tuple(float(text) * DEG_TO_RAD for text in velocity_text)
    if tuple(
        _finite_vector(
            record.get("joint_positions_rad_after_wire_roundtrip"),
            "serialized wire round-trip positions",
        )
    ) != positions_rad:
        raise ValueError("Serialized wire round-trip positions changed")
    if tuple(
        _finite_vector(
            record.get("joint_velocities_rad_s_after_wire_roundtrip"),
            "serialized wire round-trip velocities",
        )
    ) != velocities_rad_s:
        raise ValueError("Serialized wire round-trip velocities changed")
    return SerializedWirePoint(
        source_sample_index=source_sample_index,
        segment_duration_text=segment_text,
        segment_duration_s=segment_duration_s,
        segment_duration_ticks_1e5=segment_ticks,
        cumulative_time_s=cumulative_time_s,
        cumulative_time_ticks_1e5=cumulative_ticks,
        position_degrees_text=position_text,
        velocity_degrees_s_text=velocity_text,
        positions_rad=positions_rad,
        velocities_rad_s=velocities_rad_s,
    )


def validate_live_first_wire_cubic(
    current_joint_positions_rad: Sequence[float],
    current_joint_velocities_rad_s: Sequence[float],
    first_serialized_wire_point: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the physical first cubic from arbitrary current q/v.

    This is a pure calculation.  It does not read a robot, create a ROS
    message, construct a PVT script, authorize execution, or open a network.
    The caller must supply a previously re-derived first wire endpoint record
    plus the current position and velocity captured by its separate live guard.
    """

    current_positions = _finite_vector(
        current_joint_positions_rad, "live first-cubic positions"
    )
    current_velocities = _finite_vector(
        current_joint_velocities_rad_s, "live first-cubic velocities"
    )
    endpoint = _wire_point_from_record(first_serialized_wire_point)
    if endpoint.cumulative_time_ticks_1e5 != endpoint.segment_duration_ticks_1e5:
        raise ValueError(
            "Live first-cubic validator requires the first serialized wire point"
        )
    duration = endpoint.segment_duration_s
    points = (
        PVTPoint(
            source_sample_index=0,
            time_s=0.0,
            positions=current_positions,
            velocities=current_velocities,
        ),
        PVTPoint(
            source_sample_index=1,
            time_s=duration,
            positions=endpoint.positions_rad,
            velocities=endpoint.velocities_rad_s,
        ),
    )
    metrics = _validate_cubic_sequence(
        points,
        proof_domain="fixed_5_wire_first_cubic_with_supplied_live_q_v",
        segment_durations_s=(duration,),
        require_filter_stable=False,
        enforce_stage_endpoint_velocity=False,
        wire_serialization_included=True,
        live_first_seed_included=True,
    )
    metrics.update(
        {
            "status": "validated_live_first_wire_cubic",
            "first_wire_source_sample_index": endpoint.source_sample_index,
            "wire_segment_duration_seconds_fixed_5": (
                endpoint.segment_duration_text
            ),
            "live_start_joint_positions_rad": list(current_positions),
            "live_start_joint_velocities_rad_s": list(current_velocities),
            "first_wire_joint_positions_rad": list(endpoint.positions_rad),
            "first_wire_joint_velocities_rad_s": list(
                endpoint.velocities_rad_s
            ),
        }
    )
    return metrics


def _build_stage_record(
    specimen_id: int, stage_index: int, stage: dict[str, Any]
) -> dict[str, Any]:
    scaled_source = _source_points(stage)
    message_points, skipped = emulate_tm_driver_selection(scaled_source)
    message_validation = validate_six_axis_pvt(message_points)
    wire_points = serialize_filtered_message_points_to_wire(message_points)
    wire_validation = _validate_wire_internal_cubics(wire_points)
    point_records = [
        _point_record(
            point,
            None if point_index == 0 else message_points[point_index - 1],
        )
        for point_index, point in enumerate(message_points)
    ]
    wire_records = [_wire_point_record(point) for point in wire_points]
    return {
        "specimen_id": specimen_id,
        "stage_index": stage_index,
        "stage_name": stage["name"],
        "source_control_samples_float64_sha256": stage[
            "control_samples_float64_sha256"
        ],
        "source_sample_count": len(scaled_source),
        "source_duration_seconds": float(
            stage["control_samples"][-1]["time_seconds"]
        ),
        "global_time_scale": GLOBAL_TIME_SCALE,
        "message_retimed_duration_seconds": message_points[-1].time_s,
        "wire_serialized_duration_seconds": wire_points[-1].cumulative_time_s,
        "start_joint_positions_rad": list(message_points[0].positions),
        "end_joint_positions_rad": list(message_points[-1].positions),
        "source_filter_skipped_points": skipped,
        "controller_points_scope": (
            "message_level_pre_fixed_5_wire_serialization_including_"
            "nontransmitted_zero_seed"
        ),
        "controller_points": point_records,
        "message_points_float64_sha256": _points_float64_sha256(message_points),
        "message_point_diagnostic_validation": message_validation,
        "message_point_physical_wire_proof_claimed": False,
        "serialized_wire_points_scope": (
            "transmitted_numeric_fields_after_deg_fixed_5_text_parse_"
            "back_to_rad_with_cumulative_rounded_segment_times"
        ),
        "serialized_wire_points": wire_records,
        "serialized_wire_tokens_sha256": _serialized_wire_tokens_sha256(
            wire_points
        ),
        "serialized_wire_numeric_points_float64_sha256": (
            _wire_numeric_points_sha256(wire_points)
        ),
        "serialized_wire_internal_validation": wire_validation,
        "first_wire_cubic": {
            "status": "requires_current_joint_position_and_velocity_validation",
            "validated_offline": False,
            "public_validator": "validate_live_first_wire_cubic",
            "first_serialized_wire_point_index": 0,
        },
        "full_physical_wire_stage_proof_claimed": False,
    }


def _immutable_limits_record() -> dict[str, Any]:
    return {
        "joint_names": list(JOINT_NAMES),
        "joint_position_lower_rad": list(JOINT_POSITION_LOWER_RAD),
        "joint_position_upper_rad": list(JOINT_POSITION_UPPER_RAD),
        "joint_velocity_limits_rad_s": list(JOINT_VELOCITY_LIMITS_RAD_S),
        "joint_acceleration_limits_rad_s2": list(
            JOINT_ACCELERATION_LIMITS_RAD_S2
        ),
        "derivative_limit_fraction": DERIVATIVE_LIMIT_FRACTION,
        "tm_driver_min_segment_duration_seconds": (
            TM_DRIVER_MIN_SEGMENT_DURATION_S
        ),
        "maximum_controller_segment_duration_seconds": (
            MAX_CONTROLLER_SEGMENT_DURATION_S
        ),
        "maximum_endpoint_velocity_rad_s": MAX_ENDPOINT_VELOCITY_RAD_S,
        "global_time_scale": GLOBAL_TIME_SCALE,
        "ros_duration_quantum_nanoseconds": 1,
        "tm_driver_wire_decimal_places": TM_DRIVER_WIRE_DECIMAL_PLACES,
        "caller_overrides_accepted": False,
    }


def _retiming_record() -> dict[str, Any]:
    return {
        "method": (
            "scale_source_time_and_velocity_quantize_to_ros_nanoseconds_then_"
            "apply_exact_installed_tm_driver_25ms_selection_then_round_each_"
            "transmitted_q_v_rad_to_deg_and_relative_dt_to_fixed_5_text_parse_"
            "back_and_accumulate_wire_segment_times"
        ),
        "global_time_scale": GLOBAL_TIME_SCALE,
        "velocity_scale": 1.0 / GLOBAL_TIME_SCALE,
        "ros_duration_quantum_nanoseconds": 1,
        "tm_driver_zero_time_seed_transmitted": False,
        "tm_driver_filter_reapplied_to_candidate": True,
        "wire_joint_position_unit": "degree",
        "wire_joint_velocity_unit": "degree_per_second",
        "wire_segment_time_kind": "relative_seconds",
        "wire_numeric_format": "std_fixed",
        "wire_decimal_places": TM_DRIVER_WIRE_DECIMAL_PLACES,
        "wire_values_parsed_back_for_cubic_proof": True,
        "wire_relative_times_accumulated_for_cubic_proof": True,
        "driver_filter_source": (
            "tm_ros2_moveit_sct.cpp get_pvt_traj(points, 0.025)"
        ),
        "driver_serializer_source": (
            "tm_command.cpp TmCommand::set_pvt_traj default precision 5"
        ),
    }


def _safety_record() -> dict[str, Any]:
    return {
        "offline_only": True,
        "ros_graph_created": False,
        "network_connection_opened": False,
        "watson_contacted": False,
        "controller_message_created": False,
        "pvt_script_created": False,
        "command_path_created": False,
        "execution_authorized": False,
        "arm_token_created": False,
        "gripper_command_created": False,
        "motion_commanded": False,
    }


def _proof_scope_record() -> dict[str, Any]:
    return {
        "message_point_diagnostics_validated": True,
        "fixed_5_wire_internal_cubics_validated": True,
        "live_first_cubics_validated": False,
        "all_stage_first_cubics_require_live_q_v": True,
        "full_physical_wire_stage_proof_claimed": False,
        "ros_message_created": False,
        "pvt_script_created": False,
        "execution_authorized": False,
    }


def _derive_retimed_ingress_candidate(
    control_samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    unscaled = _unscaled_control_points(control_samples)
    _validate_point_sequence(unscaled)
    scaled = _scaled_points(unscaled)
    message_points, skipped = emulate_tm_driver_selection(scaled)
    message_validation = validate_six_axis_pvt(message_points)
    wire_points = serialize_filtered_message_points_to_wire(message_points)
    wire_validation = _validate_wire_internal_cubics(wire_points)
    records = [
        _point_record(
            point,
            None if point_index == 0 else message_points[point_index - 1],
        )
        for point_index, point in enumerate(message_points)
    ]
    return {
        "format_version": 2,
        "candidate_kind": (
            "offline_non_executable_ingress_six_axis_fixed_5_wire_candidate"
        ),
        "installed_tm_driver_provenance": (
            verify_installed_tm_driver_provenance()
        ),
        "retiming": _retiming_record(),
        "immutable_six_axis_limits": _immutable_limits_record(),
        "source_sample_count": len(unscaled),
        "source_points_float64_sha256": _points_float64_sha256(unscaled),
        "source_filter_skipped_points": skipped,
        "controller_points_scope": (
            "message_level_pre_fixed_5_wire_serialization_including_"
            "nontransmitted_zero_seed"
        ),
        "controller_points": records,
        "message_points_float64_sha256": _points_float64_sha256(
            message_points
        ),
        "message_point_diagnostic_validation": message_validation,
        "message_point_physical_wire_proof_claimed": False,
        "serialized_wire_points_scope": (
            "transmitted_numeric_fields_after_deg_fixed_5_text_parse_"
            "back_to_rad_with_cumulative_rounded_segment_times"
        ),
        "serialized_wire_points": [
            _wire_point_record(point) for point in wire_points
        ],
        "serialized_wire_tokens_sha256": _serialized_wire_tokens_sha256(
            wire_points
        ),
        "serialized_wire_numeric_points_float64_sha256": (
            _wire_numeric_points_sha256(wire_points)
        ),
        "serialized_wire_internal_validation": wire_validation,
        "first_wire_cubic": {
            "status": "requires_current_joint_position_and_velocity_validation",
            "validated_offline": False,
            "public_validator": "validate_live_first_wire_cubic",
            "first_serialized_wire_point_index": 0,
        },
        "full_physical_wire_trajectory_proof_claimed": False,
        "safety": _safety_record(),
        "warning": (
            "Offline ingress retiming and fixed-five wire-numeric proof only; "
            "the live-start q/v seeded first cubic, collision state, tool load, "
            "and execution authorization remain outside this candidate."
        ),
    }


def retime_ingress_control_samples(
    control_samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Retime arbitrary offline ingress samples under the immutable wire proof.

    This returns data only.  It does not build a ROS message or check a live
    robot start; any later physical integration must separately call
    ``validate_live_first_wire_cubic`` with the current q/v.
    """

    candidate = _derive_retimed_ingress_candidate(control_samples)
    validate_retimed_ingress_candidate(candidate, control_samples)
    return candidate


def validate_retimed_ingress_candidate(
    candidate: dict[str, Any],
    control_samples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Re-derive an arbitrary ingress candidate and reject any changed value."""

    if not isinstance(candidate, dict):
        raise ValueError("Retimed ingress candidate must be an object")
    expected = _derive_retimed_ingress_candidate(control_samples)
    if candidate != expected:
        raise ValueError(
            "Retimed ingress candidate does not exactly derive from its "
            "source samples, installed provenance, immutable filter, and "
            "fixed-five wire serializer"
        )
    return expected["serialized_wire_internal_validation"]


def _aggregate_metrics(
    source_metrics: dict[str, Any], stages: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    message_validations = [
        stage["message_point_diagnostic_validation"] for stage in stages
    ]
    wire_validations = [
        stage["serialized_wire_internal_validation"] for stage in stages
    ]
    if not all(
        validation.get("status")
        == "validated_internal_wire_cubics_first_live_cubic_pending"
        for validation in wire_validations
    ):
        raise ValueError(
            "Every reviewed stage must contain validated internal wire cubics"
        )
    message_point_count = sum(
        len(stage["controller_points"]) for stage in stages
    )
    wire_endpoint_count = sum(
        len(stage["serialized_wire_points"]) for stage in stages
    )
    source_skipped = sum(
        stage["source_filter_skipped_points"] for stage in stages
    )
    message_duration = math.fsum(
        stage["message_retimed_duration_seconds"] for stage in stages
    )
    wire_duration = math.fsum(
        stage["wire_serialized_duration_seconds"] for stage in stages
    )
    wire_position_min = [
        min(validation["position_minimum_rad"][joint] for validation in wire_validations)
        for joint in range(len(JOINT_NAMES))
    ]
    wire_position_max = [
        max(validation["position_maximum_rad"][joint] for validation in wire_validations)
        for joint in range(len(JOINT_NAMES))
    ]
    wire_maximum_step = [
        max(
            validation["maximum_endpoint_step_rad"][joint]
            for validation in wire_validations
        )
        for joint in range(len(JOINT_NAMES))
    ]
    wire_maximum_velocity = [
        max(
            validation["maximum_cubic_velocity_rad_s"][joint]
            for validation in wire_validations
        )
        for joint in range(len(JOINT_NAMES))
    ]
    wire_maximum_acceleration = [
        max(
            validation["maximum_cubic_acceleration_rad_s2"][joint]
            for validation in wire_validations
        )
        for joint in range(len(JOINT_NAMES))
    ]
    message_points: list[PVTPoint] = []
    wire_points: list[PVTPoint] = []
    wire_text_digest = hashlib.sha256()
    for stage in stages:
        for record in stage["controller_points"]:
            message_points.append(_point_from_record(record))
        for record in stage["serialized_wire_points"]:
            point = _wire_point_from_record(record)
            wire_points.append(
                PVTPoint(
                    source_sample_index=point.source_sample_index,
                    time_s=point.cumulative_time_s,
                    positions=point.positions_rad,
                    velocities=point.velocities_rad_s,
                )
            )
        wire_text_digest.update(
            bytes.fromhex(stage["serialized_wire_tokens_sha256"])
        )
    return {
        **source_metrics,
        "message_point_count_including_stage_zero_seeds": message_point_count,
        "driver_transmitted_wire_endpoint_count": wire_endpoint_count,
        "source_filter_skipped_points": source_skipped,
        "post_message_candidate_filter_skipped_points": 0,
        "message_retimed_stage_motion_duration_seconds": message_duration,
        "wire_serialized_stage_motion_duration_seconds": wire_duration,
        "wire_minus_message_duration_seconds": wire_duration - message_duration,
        "message_minimum_segment_duration_seconds": min(
            validation["minimum_segment_duration_seconds"]
            for validation in message_validations
        ),
        "message_maximum_segment_duration_seconds": max(
            validation["maximum_segment_duration_seconds"]
            for validation in message_validations
        ),
        "wire_internal_minimum_segment_duration_seconds": min(
            validation["minimum_segment_duration_seconds"]
            for validation in wire_validations
        ),
        "wire_internal_maximum_segment_duration_seconds": max(
            validation["maximum_segment_duration_seconds"]
            for validation in wire_validations
        ),
        "wire_internal_position_minimum_rad": wire_position_min,
        "wire_internal_position_maximum_rad": wire_position_max,
        "wire_internal_maximum_endpoint_step_rad": wire_maximum_step,
        "wire_internal_maximum_cubic_velocity_rad_s": wire_maximum_velocity,
        "wire_internal_maximum_cubic_acceleration_rad_s2": (
            wire_maximum_acceleration
        ),
        "wire_internal_maximum_velocity_limit_utilization": max(
            wire_maximum_velocity[joint] / JOINT_VELOCITY_LIMITS_RAD_S[joint]
            for joint in range(len(JOINT_NAMES))
        ),
        "wire_internal_maximum_acceleration_limit_utilization": max(
            wire_maximum_acceleration[joint]
            / JOINT_ACCELERATION_LIMITS_RAD_S2[joint]
            for joint in range(len(JOINT_NAMES))
        ),
        "wire_internal_cubic_count": sum(
            validation["internal_cubic_count"]
            for validation in wire_validations
        ),
        "wire_internal_stage_count_validated": len(wire_validations),
        "wire_first_cubic_count_pending_live_validation": len(stages),
        "message_point_physical_wire_proof_claimed": False,
        "full_physical_wire_stage_proof_claimed": False,
        "message_points_float64_sha256": _points_float64_sha256(
            message_points
        ),
        "serialized_wire_numeric_points_float64_sha256": (
            _points_float64_sha256(wire_points)
        ),
        "serialized_wire_stage_token_hashes_sha256": (
            wire_text_digest.hexdigest()
        ),
    }


def build_retimed_artifact(
    plan_path: Path = DEFAULT_REVIEWED_PLAN,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build and self-validate a private, non-executable PVT candidate."""

    resolved_plan_path = _absolute_path_without_following_final_symlink(plan_path)
    plan = load_reviewed_plan(resolved_plan_path)
    installed_provenance = verify_installed_tm_driver_provenance()
    source_metrics = validate_reviewed_plan(plan)
    stage_records = [
        _build_stage_record(specimen_id, stage_index, stage)
        for specimen_id, stage_index, stage in _source_stages(plan)
    ]
    metrics = _aggregate_metrics(source_metrics, stage_records)
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("Artifact timestamp must include a timezone")
    artifact: dict[str, Any] = {
        "format_version": 2,
        "artifact_kind": ARTIFACT_KIND,
        "status": ARTIFACT_STATUS,
        "timestamp_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "source_plan": str(resolved_plan_path),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "source_numeric_sample_sha256": EXPECTED_SOURCE_NUMERIC_SHA256,
        "joint_names": list(JOINT_NAMES),
        "ready_joint_positions_rad": list(READY_JOINT_POSITIONS_RAD),
        "installed_tm_driver_provenance": installed_provenance,
        "retiming": _retiming_record(),
        "immutable_six_axis_limits": _immutable_limits_record(),
        "proof_scope": _proof_scope_record(),
        "metrics": metrics,
        "stages": stage_records,
        "safety": _safety_record(),
        "warning": ARTIFACT_WARNING,
    }
    artifact[ARTIFACT_DIGEST_FIELD] = canonical_digest(artifact)
    validate_retimed_artifact(artifact, plan)
    return artifact


def validate_retimed_artifact(
    artifact: dict[str, Any], source_plan: dict[str, Any]
) -> dict[str, Any]:
    """Re-derive every candidate point and reject any widened or changed field."""

    source_metrics = validate_reviewed_plan(source_plan)
    if artifact.get("format_version") != 2:
        raise ValueError(
            "Retimed artifact format v2 with fixed-five wire proof is required; "
            "message-only/unrounded v1 evidence is rejected"
        )
    if artifact.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Retimed artifact kind changed")
    if artifact.get("status") != ARTIFACT_STATUS:
        raise ValueError("Retimed artifact status changed")
    if artifact.get("source_plan_sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("Retimed artifact source-plan hash changed")
    if (
        artifact.get("source_numeric_sample_sha256")
        != EXPECTED_SOURCE_NUMERIC_SHA256
    ):
        raise ValueError("Retimed artifact source numeric hash changed")
    if tuple(artifact.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("Retimed artifact joint order changed")
    if tuple(artifact.get("ready_joint_positions_rad", ())) != (
        READY_JOINT_POSITIONS_RAD
    ):
        raise ValueError("Retimed artifact ready pose changed")
    installed_provenance = verify_installed_tm_driver_provenance()
    if artifact.get("installed_tm_driver_provenance") != installed_provenance:
        raise ValueError("Installed tm_driver provenance changed")
    if artifact.get("immutable_six_axis_limits") != _immutable_limits_record():
        raise ValueError("Immutable six-axis limits were changed")
    if artifact.get("retiming") != _retiming_record():
        raise ValueError("Immutable retiming policy was changed")
    safety = artifact.get("safety")
    if safety != _safety_record():
        raise ValueError("Retimed artifact safety scope was widened")
    if artifact.get("proof_scope") != _proof_scope_record():
        raise ValueError("Retimed artifact proof scope changed")
    try:
        parsed_timestamp = datetime.fromisoformat(
            str(artifact.get("timestamp_utc"))
        )
    except ValueError as exc:
        raise ValueError("Retimed artifact timestamp is invalid") from exc
    if parsed_timestamp.tzinfo is None:
        raise ValueError("Retimed artifact timestamp must include a timezone")
    if artifact.get("warning") != ARTIFACT_WARNING:
        raise ValueError("Retimed artifact warning changed")

    stored_stages = artifact.get("stages")
    if not isinstance(stored_stages, list) or len(stored_stages) != (
        EXPECTED_STAGE_COUNT
    ):
        raise ValueError("Retimed artifact stage count changed")
    expected_stages = [
        _build_stage_record(specimen_id, stage_index, stage)
        for specimen_id, stage_index, stage in _source_stages(source_plan)
    ]
    for stage_number, (stored, expected) in enumerate(
        zip(stored_stages, expected_stages)
    ):
        if stored != expected:
            raise ValueError(
                f"Retimed stage {stage_number} does not exactly derive from "
                "the reviewed source and immutable filter"
            )
        points = tuple(
            _point_from_record(record) for record in stored["controller_points"]
        )
        for point_index, record in enumerate(stored["controller_points"]):
            expected_segment = (
                0.0
                if point_index == 0
                else points[point_index].time_s - points[point_index - 1].time_s
            )
            if record.get("segment_duration_seconds") != expected_segment:
                raise ValueError("Retimed point segment duration changed")
        message_validation = validate_six_axis_pvt(points)
        if message_validation.get("physical_wire_proof_claimed") is not False:
            raise ValueError("Unrounded message proof was promoted to wire proof")
        wire_points = tuple(
            _wire_point_from_record(record)
            for record in stored["serialized_wire_points"]
        )
        prior_cumulative_ticks = 0
        for wire_point in wire_points:
            if (
                wire_point.cumulative_time_ticks_1e5
                != prior_cumulative_ticks
                + wire_point.segment_duration_ticks_1e5
            ):
                raise ValueError(
                    "Serialized wire cumulative segment timing changed"
                )
            prior_cumulative_ticks = wire_point.cumulative_time_ticks_1e5
        wire_validation = _validate_wire_internal_cubics(wire_points)
        if wire_validation.get("physical_wire_proof_claimed") is not False:
            raise ValueError(
                "Internal wire proof improperly claimed the live first cubic"
            )

    expected_metrics = _aggregate_metrics(source_metrics, expected_stages)
    if artifact.get("metrics") != expected_metrics:
        raise ValueError("Retimed aggregate metrics changed")
    if artifact.get(ARTIFACT_DIGEST_FIELD) != canonical_digest(artifact):
        raise ValueError("Retimed artifact payload digest mismatch")
    return expected_metrics


def write_private_artifact(path: Path, artifact: dict[str, Any]) -> Path:
    """Create one exclusive mode-0600 artifact and durably sync it."""

    if artifact.get(ARTIFACT_DIGEST_FIELD) != canonical_digest(artifact):
        raise ValueError("Refusing to write artifact with an invalid digest")
    destination = path.expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    destination = destination.parent.resolve() / destination.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(artifact, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
            ):
                raise RuntimeError("Private retimed artifact file checks failed")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    final = destination.lstat()
    if (
        destination.is_symlink()
        or not stat.S_ISREG(final.st_mode)
        or stat.S_IMODE(final.st_mode) != 0o600
        or final.st_uid != os.geteuid()
        or final.st_nlink != 1
    ):
        destination.unlink(missing_ok=True)
        raise RuntimeError("Private retimed artifact final checks failed")
    directory_descriptor = os.open(
        destination.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return destination
