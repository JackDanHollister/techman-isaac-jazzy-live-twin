# Read-only live twin

## Purpose

The live twin displays the physical Techman arm in Isaac without creating a
robot command path.

## Start

Create `local/watson-site.env` first (see Installation — the preflight reads
it). Bring up the Techman ROS graph separately with MoveIt trajectory
execution disabled, exporting `ROS_DOMAIN_ID=219` and
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` so the viewer's pinned isolated
domain can see it, then run:

```bash
./scripts/run_isaac_live_twin.sh
```

The wrapper first creates a fresh read-only health report. It then launches
Isaac with the isolated bundled Jazzy runtime.

## Data handling

The mirror accepts exactly six named joints:

```text
joint_1 joint_2 joint_3 joint_4 joint_5 joint_6
```

Message order does not matter. The validator rejects:

- missing or duplicate names;
- incomplete position/velocity arrays;
- non-finite values;
- positions outside imported articulation limits;
- malformed timestamps.

The status panel reports:

- `WAITING`: no valid message has arrived;
- `LIVE`: feedback is fresh and being applied;
- `STALE`: the last valid pose is frozen without extrapolation.

## Articulation

The bundled reference USD has eight DOFs: six arm joints and two inward 2FG7
finger joints. The read-only twin maps only the six arm joints and leaves the
visual fingers open.

Joint positions are applied while physics remains paused. After each update,
the viewer synchronises the PhysX kinematic articulation into USD and compares
the rendered `link_6` pose with the PhysX link transform.

## ROS graph boundary

The viewer creates one best-effort joint-state subscription. It audits its own
publishers, services, clients, and actions, and fails if a command-capable path
appears.

The live viewer does not:

- publish trajectories;
- create MoveIt or controller action clients;
- call Techman services;
- step dynamics;
- command the 2FG7.
