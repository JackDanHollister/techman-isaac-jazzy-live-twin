# Architecture

## Runtime boundaries

The system deliberately separates presentation from physical command authority.

### Isaac process

The Isaac process:

- subscribes to `/watson/joint_states`;
- validates and name-maps the six Techman joints;
- teleports the paused Isaac articulation to measured positions;
- updates the two visual 2FG7 joints from ordered runner events;
- attaches and releases synthetic specimens for presentation;
- observes Isaac Play, Pause, and Stop;
- starts or signals the external guarded wrapper.

Its ROS graph is audited at runtime. It must not contain robot-command
publishers, service clients, or action clients.

### Guarded wrapper

The external wrapper is the only physical authority. In execute mode it owns the
Techman ROS/MoveIt launch, validates network routing and graph ownership,
selects and reads back the named TCP, checks fresh robot state around every
stage, submits the reviewed trajectories, controls the 2FG7, and performs
ordered stop recovery before tearing down its ROS stack.

### Event protocol

The runner prints versioned JSON events with consecutive sequence numbers:

- run started;
- stage started/completed/failed;
- gripper started/completed;
- one final run completed/failed/stopped event.

Events are presentation state only. They do not grant permission to move the
robot.

## State flow

```text
Techman feedback -> ROS 2 Jazzy -> validator -> Isaac arm articulation
2FG7 command completion -> event stream -> Isaac finger/specimen presentation
Isaac Play -> HIL coordinator -> guarded wrapper -> MoveIt/tm_driver/Compute Box
```

## Timeline semantics

- **Play** consumes the one-shot launch after manual arming.
- **Pause** requests guarded cancellation; physical pause is not supported.
- **Stop** requests the same guarded cancellation.
- Closing the window requests cancellation and waits for runner recovery before
  wrapper teardown.

The timeline is immediately paused after the trigger. Isaac physics is not used
as a clock for the physical trajectory.

## Twin scope

The current twin is kinematic:

- arm pose comes from measured joints;
- finger and specimen state comes from ordered events;
- scene objects use manually configured poses;
- no force, contact, friction, payload mass, or camera registration claim is
  made.
