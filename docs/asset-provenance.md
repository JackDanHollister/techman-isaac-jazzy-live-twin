# Asset provenance

## Reference bundle

`reference/seven_pin/` contains a sanitised, hash-pinned snapshot needed to run
the offline demonstration from a clean clone:

- the seven-specimen cuMotion plan;
- the articulated Techman/QC/2FG7 USD package;
- the source URDF and import validation report;
- tool and staging metadata;
- cuMotion URDF/XRDF provenance;
- the reviewed retimed and ingress execution fixtures.

Workstation paths and hostname strings were replaced with repository-relative
provenance. Numeric trajectories, limits, geometry, stage order, and validation
hashes remain checked by the test suite.

## Techman model

Robot geometry derives from Techman Robot's `tm_description` package in
`tm2_ros2`, distributed under BSD-3-Clause. A licence copy is retained under
`third_party/techman/`.

## OnRobot model

The articulated gripper uses the vendored MIT-licensed 2FG7 description under
`vendor/onrobot_2fg7/`, with inward finger geometry created for the physical
orientation used by the demonstration.

The upstream model is MIT-licensed and states that its mesh source was OnRobot
CAD. The upstream licence and README are retained with the vendored files. This
project does not relicense those meshes; anyone redistributing a modified asset
bundle should repeat the provenance and licence review.

## Excluded material

The repository does not include:

- official OnRobot STEP downloads;
- TMflow/OnRobot Component archives;
- NVIDIA software or environments;
- Zivid scans or point clouds;
- raw robot reports and runtime logs;
- controller exports;
- credentials or authentication material.

## Rebuilding

The scripts under `scripts/` can rebuild URDF, cuMotion, and Isaac assets when
the required external Techman, OnRobot, Isaac, and cuMotion installations are
available. Rebuilt assets are written below ignored `generated/` and `outputs/`
directories unless explicitly curated.
