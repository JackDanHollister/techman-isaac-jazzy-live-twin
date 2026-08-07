"""Strict validation for the Watson offline commissioning template.

This module has no controller, ROS, motion, or write path.  A successful
validation means only that the deliberately blocked template is intact; it
never means that a tool configuration is ready or has been applied.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


OFFLINE_MANIFEST_SCHEMA_VERSION = 1
OBSERVED_WATSON_GUARD_SCHEMA_VERSION = 8
ARTIFACT_KIND = "offline_controller_commissioning_template"
BLOCKED_STATUS = "blocked_not_ready_not_applied"

REQUIRED_PHYSICAL_ITEMS = (
    "quick_changer_revision",
    "adapter_k_presence",
    "quick_changer_keyed_yaw",
    "calibrated_pinch_tcp",
    "workpiece_payload",
    "physical_dry_tool_principal_moi",
)

PROMOTION_BLOCKERS = (
    "approved v2-labelled working CAD has not been registered to the confirmed physical clock orientation",
    "application pinch TCP has not been physically calibrated",
    "workpiece mass, centre of gravity, and moments of inertia are unknown",
    "physical dry-tool principal moments of inertia are unknown",
    "the schema-8 Watson guard must still read and approve the live controller records",
)


_EXPECTED_MANIFEST: dict[str, Any] = {
    "offline_manifest_schema_version": OFFLINE_MANIFEST_SCHEMA_VERSION,
    "artifact_kind": ARTIFACT_KIND,
    "status": BLOCKED_STATUS,
    "safety": {
        "offline_only": True,
        "controller_contacted": False,
        "controller_write_authorized": False,
        "controller_write_performed": False,
        "controller_application_authorized": False,
        "motion_commanded": False,
        "applied_to_controller": False,
    },
    "guard_relationship": {
        "observed_watson_report_schema_version": (
            OBSERVED_WATSON_GUARD_SCHEMA_VERSION
        ),
        "accepted_as_live_controller_tool_audit": False,
        "can_bypass_live_ask_item_query": False,
        "promotion_passed": False,
    },
    "candidate_assembly_scope": {
        "robot": "Watson TM5S",
        "quick_changer": "OnRobot single standard robot-side Quick Changer",
        "gripper": "OnRobot 2FG7",
        "included": [
            "single_standard_quick_changer_robot_side",
            "integrated_2fg7",
        ],
        "excluded_from_candidate": ["adapter_k", "accessory", "workpiece"],
        "status": (
            "direct_stack_physically_confirmed_remaining_tcp_payload_and_"
            "cad_registration_gates_apply"
        ),
    },
    "official_source_facts": {
        "quick_changer_datasheet": {
            "title": "OnRobot Quick Changers Datasheet v2.0",
            "url": (
                "https://onrobot.com/storage/datasheets/quick-changers/"
                "datasheet_quick_changers_v2.0_en.pdf"
            ),
            "sha256": (
                "67948af8a8cb3e21e99a87b4348f360b72f842f4fd5d68d71defe57631daeb15"
            ),
            "technical_data_page": 3,
            "mechanical_dimensions_page": 7,
            "single_quick_changer_mass_kg": 0.06,
            "flange_to_tool_interface_z_m": 0.0136,
            "physical_body_height_m": 0.0161,
            "repeatability_m": 0.00002,
        },
        "installed_quick_changer_identity": {
            "item_number": "109498",
            "version": "QC-R v3",
            "ip_classification": "IP67",
            "label_observed_date": "2026-07-20",
            "label_evidence": "user_read_installed_label",
            "official_product_page": (
                "https://b2b.onrobot.com/quick-changer-robot-side/"
            ),
            "official_current_datasheet": (
                "https://onrobot.com/storage/datasheets/quick-changers/"
                "datasheet_quick_changers_v2.0_en.pdf"
            ),
            "official_current_datasheet_sha256": (
                "67948af8a8cb3e21e99a87b4348f360b72f842f4fd5d68d71defe57631daeb15"
            ),
        },
        "quick_changer_cog_calculator": {
            "title": "OnRobot TCP_COG_Calculator v1.4",
            "url": (
                "https://onrobot.dforigo.com/api/renderpoint/feeds/web/artifacts/"
                "NXD4QOYN1U6IC4HLd9mlGQ/download/TCP_COG_Calculator_v1.4.xlsx"
            ),
            "sha256": (
                "d0e7e757f2f726b816004fdbb85a6ef59748fb4df7f45993d14336f0fdb0563d"
            ),
            "cells": "Data!P6:AA6",
            "single_quick_changer_cog_xyz_m": [0.0, 0.0, 0.004],
        },
        "techman_2fg7_manual": {
            "title": "OnRobot User Manual for Techman Robots v6.6.0",
            "filename": (
                "User_Manual_For_Techman_Robots_Quick_Changer_2FG7_"
                "v6.6.0_EN.pdf"
            ),
            "sha256": (
                "c51f23b57611c17a9da2351b0b5eacd0f5a154124d5930a96a7c1a68cbf772cd"
            ),
            "printed_page": 78,
            "scope": "single_device_without_mount_adapter_or_workpiece",
            "nominal_tcp_xyz_from_device_origin_m": [0.0, 0.0, 0.125],
            "mass_kg": 1.14,
            "cog_xyz_from_device_origin_m": [0.0, 0.0, 0.052],
        },
    },
    "safe_precomputations": {
        "scope": (
            "nominal_direct_dry_stack_single_qc_plus_2fg7_without_adapter_"
            "accessory_or_workpiece"
        ),
        "nominal_flange_to_2fg7_vendor_tcp_xyz_m": [0.0, 0.0, 0.1386],
        "nominal_dry_mass_kg": 1.2,
        "nominal_dry_cog_xyz_from_robot_flange_m": [0.0, 0.0, 0.06252],
        "derivation": {
            "tcp_z_m": "0.0136 + 0.125",
            "mass_kg": "0.06 + 1.14",
            "cog_z_m": (
                "(0.06 * 0.004 + 1.14 * (0.0136 + 0.052)) / 1.20"
            ),
        },
        "status": (
            "vendor_nominal_reference_only_not_a_calibrated_pinch_tcp_"
            "or_controller_approved_payload"
        ),
    },
    "application_grasp_baseline": {
        "scope": "simulation_and_planning_reference_only_not_a_controller_entry",
        "pin_axis_from_gripper_toward_specimen": [0.0, 0.0, 1.0],
        "lateral_alignment": "centred_between_inward_fingers",
        "clear_pin_length_before_specimen_m": 0.01,
        "grasp_point_on_clear_section": "midpoint",
        "pinch_xyz_from_2fg7_device_origin_m": [0.0, 0.0, 0.15725],
        "pinch_xyz_from_robot_flange_m": [0.0, 0.0, 0.17085],
        "pinch_to_specimen_m": 0.005,
        "specimen_geometry": "variable_not_fixed",
        "remaining_pin_length_after_specimen_m": None,
        "cable_geometry_included": False,
        "status": (
            "user_selected_10mm_cad_relative_baseline_not_physically_calibrated"
        ),
    },
    "required_physical_measurements": {
        "quick_changer_revision": {
            "item_number": "109498",
            "value": "QC-R v3",
            "ip_classification": "IP67",
            "observed_date": "2026-07-20",
            "evidence": "user_read_installed_label",
            "status": "physically_confirmed",
        },
        "adapter_k_presence": {
            "value": False,
            "observed_date": "2026-07-20",
            "evidence": "user_physical_inspection",
            "status": "physically_confirmed_absent",
        },
        "quick_changer_keyed_yaw": {
            "value_rad": None,
            "clock_orientation": {
                "viewpoint": (
                    "flange_facing_down_viewed_from_robot_wrist_toward_"
                    "tool_and_floor"
                ),
                "twelve_oclock_reference": "tm_eih_camera",
                "quick_release_control": "12_oclock_facing_tm_eih_camera",
                "cable_wrap": "3_oclock",
                "cable_end_socket": "9_oclock",
            },
            "observed_date": "2026-07-20",
            "status": (
                "physical_clock_features_confirmed_numeric_approved_v2_working_cad_"
                "registration_pending"
            ),
        },
        "calibrated_pinch_tcp": {
            "xyz_from_robot_flange_m": None,
            "rpy_from_robot_flange_rad": None,
            "status": "unknown_requires_physical_tcp_calibration",
        },
        "workpiece_payload": {
            "mass_kg": None,
            "cog_xyz_from_tcp_m": None,
            "principal_moi_kg_m2": None,
            "status": "unknown_requires_actual_workpiece_measurement",
        },
        "physical_dry_tool_principal_moi": {
            "principal_moi_kg_m2": None,
            "status": "unknown_simulation_proxy_not_accepted",
        },
    },
    "controller_entry_placeholders": {
        "tcp_name": None,
        "tcp_value": None,
        "mass_kg": None,
        "principal_moi": None,
        "mass_centre_frame": None,
        "status": "intentionally_blank_not_a_controller_entry_sheet",
    },
    "promotion_blockers": list(PROMOTION_BLOCKERS),
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number: {value}")


def parse_offline_commissioning_manifest(raw: str) -> dict[str, Any]:
    """Parse strict JSON for the offline template without accepting extensions."""

    try:
        manifest = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"offline commissioning template is not strict JSON: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("offline commissioning template root must be an object")
    return manifest


def load_offline_commissioning_manifest(path: Path) -> dict[str, Any]:
    """Read an offline template.  This function never writes or contacts a robot."""

    return parse_offline_commissioning_manifest(path.read_text(encoding="utf-8"))


def _compare_exact(
    actual: Any,
    expected: Any,
    path: str,
    failures: list[str],
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            failures.append(f"{path} must be an object")
            return
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            failures.append(f"{path} is missing keys {missing}")
        if extra:
            failures.append(f"{path} contains forbidden extra keys {extra}")
        for key in sorted(expected.keys() & actual.keys()):
            _compare_exact(actual[key], expected[key], f"{path}.{key}", failures)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            failures.append(f"{path} must be an array")
            return
        if len(actual) != len(expected):
            failures.append(
                f"{path} must contain {len(expected)} entries; found {len(actual)}"
            )
            return
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare_exact(actual_item, expected_item, f"{path}[{index}]", failures)
        return
    if isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            failures.append(f"{path} must be the reviewed value {expected!r}")
        return
    if type(actual) is not type(expected) or actual != expected:
        failures.append(f"{path} must be {expected!r}; found {actual!r}")


def offline_commissioning_manifest_failures(manifest: Any) -> list[str]:
    """Return every way an artifact differs from the reviewed blocked template."""

    failures: list[str] = []
    _compare_exact(manifest, _EXPECTED_MANIFEST, "$", failures)
    return failures


def validate_offline_commissioning_manifest(manifest: Any) -> dict[str, Any]:
    """Validate template integrity and return an explicitly non-promoting result."""

    failures = offline_commissioning_manifest_failures(manifest)
    if failures:
        raise ValueError(
            "invalid offline commissioning template: " + "; ".join(failures)
        )
    return {
        "offline_template_integrity_passed": True,
        "status": BLOCKED_STATUS,
        "commissioning_ready": False,
        "controller_application_authorized": False,
        "applied_to_controller": False,
        "motion_commanded": False,
        "promotion_passed": False,
        "safe_precomputations": dict(manifest["safe_precomputations"]),
        "required_physical_items": list(REQUIRED_PHYSICAL_ITEMS),
        "promotion_blockers": list(PROMOTION_BLOCKERS),
    }
