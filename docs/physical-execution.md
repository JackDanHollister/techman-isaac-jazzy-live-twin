# Physical execution

## Experimental Watson profile

The physical path is a reference implementation for one reviewed TM5S-900
system, internally named Watson. It is not a generic launch command for an
uncommissioned Techman robot.

The reviewed motion is an **empty-cell air replay**. It contains no registered
physical pin tray, specimen, obstacle, or collision scene.

## Prepare local artifacts

Git stores the sanitised reference artifacts read-only. The runner requires
private, owner-only working copies:

```bash
mkdir -p local
cp config/watson-site.env.example local/watson-site.env
chmod 600 local/watson-site.env
# Edit local/watson-site.env for the reviewed cell, including its robot MAC.

./scripts/stage_reference_execution.sh
./scripts/run_watson_multi_pin_air_replay.sh --offline-validate
```

The site profile stays outside Git. Staging copies no credentials and contacts
no device.

## Dry run

Dry run:

- connects to the existing ROS graph;
- checks fresh feedback and graph ownership;
- reads the controller tool profile;
- builds all exact `RobotTrajectory` messages;
- creates no action goal;
- creates no 2FG7 transport.

```bash
./scripts/run_isaac_watson_hil.sh --mode dry-run
```

## Execute gates

Execute requires all of the following:

- a visible Isaac window;
- panel **ARM ONE-SHOT**;
- Isaac toolbar **Play**;
- exact arm and gripper confirmation tokens;
- explicit cell-clear confirmation;
- isolated, reviewed network routing;
- exact `/watson` graph ownership;
- Techman Listen node `Listen1`;
- project speed exactly `50`;
- no robot/controller error and fresh stationary feedback;
- named controller tool `QC_2FG7_VENDOR`;
- the reviewed start pose within `0.001 rad`;
- the hash-pinned ingress and 49-stage path;
- fresh first-wire cubic validation at every stage;
- verified stationary/action-idle state after every stage;
- ordered 2FG7 completion or recovery STOP.

The command is intentionally discoverable in `--help`, but is omitted here to
discourage copy-and-run use on an unreviewed cell.

## Stop behavior

Isaac Pause, Stop, panel Stop, window close, SIGINT, SIGTERM, and SIGHUP all
request runner-first cancellation. The wrapper waits for arm cancellation and
2FG7 recovery before removing its owned ROS stack.

If stationary or stop proof cannot be obtained after a physical command, the
run exits on an E-stop-class status. The software report cannot replace direct
observation of the robot and vendor safety indicators.

## Before adapting

Create a new site profile and revalidate:

- robot model and joint limits;
- controller and `tm_driver` versions;
- network interface and routes;
- ROS namespace and graph owners;
- tool TCP, payload, centre of gravity, and inertia;
- start pose and ingress;
- workspace collision model;
- velocity/acceleration and wire-format behavior;
- gripper device, direction, force, width, and stop semantics.

Do not loosen the Watson-specific checks merely to make another robot accept the
existing replay.
