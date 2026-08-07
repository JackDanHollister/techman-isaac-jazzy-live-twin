# Safety model

## Authority

Isaac Sim is a visualisation and orchestration process. It is not a safety
controller. The physical E-stop, Techman safety system, risk assessment,
guarding, and trained operator remain authoritative.

## Mode boundaries

- **Preview**: no ROS graph, network, robot process, or device transport.
- **Live twin**: read-only ROS joint-state subscription.
- **Dry run**: live state inspection and message construction, zero action goals
  and zero gripper transport.
- **Execute**: physical motion and gripper actuation after explicit arming.

Execution authorisation is never inferred from running a preview or dry run.

## Fail-closed behavior

The physical runner rejects:

- stale or inconsistent joint feedback;
- unexpected action or topic owners;
- controller/TMflow errors;
- wrong Listen node, speed, tool, base, or start pose;
- changed plan, ingress, driver, serializer, or tool hashes;
- out-of-envelope samples or first-wire cubics;
- unverified gripper state;
- missing stationary or cancellation proof.

## Pause and Stop

Techman has no physical pause semantic in this integration. Isaac Pause maps to
guarded cancellation. Stop requests are not considered complete until the
runner verifies the resulting arm and gripper state.

## Current task boundary

The validated physical demonstration is empty-cell motion. The virtual pins and
tray are not registered physical obstacles. Real picking requires application
TCP calibration, workcell registration, perception, collision validation, and
a new task risk assessment.

## Reports

Reports are useful evidence, not safety authority. A green report does not prove
the physical area was clear, and a software failure must not delay use of the
physical E-stop when the robot state is uncertain.
