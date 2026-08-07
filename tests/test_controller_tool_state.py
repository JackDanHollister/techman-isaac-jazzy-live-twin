from __future__ import annotations

from types import SimpleNamespace
import unittest

from pin_axis_3d_sim.controller_tool_state import (
    controller_tool_failures,
    matches_qc_2fg7_vendor_profile,
    parse_controller_tool_items,
    parse_tmflow_item,
    query_controller_tool_items,
)


RAW_UNCONFIGURED = {
    "TCP_Name": 'TCP_Name="RobotEndFlange"',
    "TCP_Value": "TCP_Value={0,0,0,0,0,0}",
    "TCP_Mass": "TCP_Mass=0",
    "TCP_MOI": "TCP_MOI={0,0,0}",
    "TCP_MCF": "TCP_MCF={0,0,0,0,0,0}",
    "Base_Name": 'Base_Name="RobotBase"',
    "Base_Value": "Base_Value={0,0,0,0,0,0}",
}


class ControllerToolParsingTests(unittest.TestCase):
    def test_parses_exact_live_unconfigured_responses(self) -> None:
        parsed = parse_controller_tool_items(RAW_UNCONFIGURED)
        self.assertEqual(parsed["active_tcp_name"], "RobotEndFlange")
        self.assertEqual(parsed["tcp_value"], [0.0] * 6)
        self.assertEqual(parsed["mass_kg"], 0.0)
        failures = controller_tool_failures(parsed)
        self.assertTrue(any("bare RobotEndFlange" in failure for failure in failures))
        self.assertTrue(any("mass is zero" in failure for failure in failures))

    def test_commissioned_nonzero_record_passes_structural_gate(self) -> None:
        parsed = parse_controller_tool_items(
            {
                **RAW_UNCONFIGURED,
                "TCP_Name": 'TCP_Name="WatsonQC2FG7"',
                "TCP_Value": "TCP_Value={0,0,138.6,0,0,0}",
                "TCP_Mass": "TCP_Mass=1.2",
                "TCP_MOI": "TCP_MOI={0.002,0.003,0.001}",
                "TCP_MCF": "TCP_MCF={0,0,62.52,0,0,0}",
            }
        )
        self.assertEqual(controller_tool_failures(parsed), [])

    def test_exact_vendor_profile_allows_unpublished_zero_inertia(self) -> None:
        parsed = parse_controller_tool_items(
            {
                **RAW_UNCONFIGURED,
                "TCP_Name": 'TCP_Name="QC_2FG7_VENDOR"',
                "TCP_Value": "TCP_Value={0,0,138.6,0,0,0}",
                "TCP_Mass": "TCP_Mass=1.2",
                "TCP_MOI": "TCP_MOI={0,0,0}",
                "TCP_MCF": "TCP_MCF={0,0,62.52,0,0,0}",
            }
        )
        self.assertTrue(matches_qc_2fg7_vendor_profile(parsed))
        self.assertEqual(controller_tool_failures(parsed), [])

    def test_zero_inertia_exception_rejects_nearby_or_renamed_profiles(self) -> None:
        base = {
            **RAW_UNCONFIGURED,
            "TCP_Name": 'TCP_Name="QC_2FG7_VENDOR"',
            "TCP_Value": "TCP_Value={0,0,138.6,0,0,0}",
            "TCP_Mass": "TCP_Mass=1.2",
            "TCP_MOI": "TCP_MOI={0,0,0}",
            "TCP_MCF": "TCP_MCF={0,0,62.52,0,0,0}",
        }
        for replacement in (
            {"TCP_Name": 'TCP_Name="QC_2FG7_OTHER"'},
            {"TCP_Value": "TCP_Value={0,0,138.7,0,0,0}"},
            {"TCP_Mass": "TCP_Mass=1.1"},
            {"TCP_MCF": "TCP_MCF={0,0,63.52,0,0,0}"},
        ):
            parsed = parse_controller_tool_items({**base, **replacement})
            self.assertFalse(matches_qc_2fg7_vendor_profile(parsed))
            self.assertTrue(
                any("principal moments" in failure for failure in controller_tool_failures(parsed))
            )

    def test_rejects_trailing_or_malformed_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "flat brace"):
            parse_tmflow_item("TCP_Value", "TCP_Value={0,0,0,0,0,0}junk")
        with self.assertRaisesRegex(ValueError, "6 values"):
            parse_tmflow_item("TCP_Value", "TCP_Value={0,0,0}")
        with self.assertRaisesRegex(ValueError, "quoted string"):
            parse_tmflow_item("TCP_Name", "TCP_Name=RobotEndFlange")


class FakeAskItem:
    class Request:
        pass


class FakeFuture:
    def __init__(self, response):
        self.response = response

    def done(self):
        return True

    def result(self):
        return self.response


class FakeClient:
    def __init__(self):
        self.requests = []

    def wait_for_service(self, timeout_sec):
        return timeout_sec > 0.0

    def call_async(self, request):
        self.requests.append(request)
        return FakeFuture(
            SimpleNamespace(ok=True, id=request.id, value=RAW_UNCONFIGURED[request.item])
        )


class ControllerToolQueryTests(unittest.TestCase):
    def test_query_uses_unique_alphanumeric_read_ids_only(self) -> None:
        client = FakeClient()
        result = query_controller_tool_items(
            node=object(),
            rclpy=SimpleNamespace(
                spin_until_future_complete=lambda node, future, timeout_sec: None
            ),
            ask_item_type=FakeAskItem,
            client=client,
            timeout_s=1.0,
        )
        ids = [request.id for request in client.requests]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(identifier.isalnum() for identifier in ids))
        self.assertEqual(result["write_items_called"], [])
        self.assertFalse(result["motion_commanded"])
        self.assertFalse(result["promotion_passed"])


if __name__ == "__main__":
    unittest.main()
