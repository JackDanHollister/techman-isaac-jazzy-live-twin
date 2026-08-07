# Changelog

All notable changes to this standalone reference implementation are recorded
here.

## [Unreleased]

### Changed

- Prepared the repository for public review with a clearer project overview,
  engineering highlights, validation evidence, and documentation routes.
- Replaced site-specific network defaults with RFC 5737 documentation
  addresses and a generic interface example; real cell values remain local.
- Clarified third-party asset provenance and licence boundaries.
- Added history-aware secret scanning in CI with narrowly scoped false-positive
  exclusions for integrity hashes and a human-readable motion confirmation.
- Added contributor guidelines covering runtime-boundary changes, validation,
  and commit-metadata privacy.

### Added

- Standalone repository packaging.
- Sanitised, hash-pinned seven-pin preview and execution reference bundle.
- Offline preview, read-only live twin, live dry-run, and guarded execute modes.
- Isaac Play/Pause/Stop one-shot HIL integration.
- Measured six-joint Techman mirroring into the articulated eight-DOF
  Techman/QC/2FG7 presentation model.
- Ordered gripper/specimen event protocol.
- Repository installation, architecture, safety, execution, and provenance
  documentation.

### Fixed

- Guarded the HIL display loop when Isaac toolbar Stop invalidates the physics
  simulation view during teardown.
- Made the offline test suite portable when a site-local, hash-pinned
  `tm_driver` workspace is absent while retaining fail-closed verification for
  every present lab workspace.
- Removed the wrapper's undeclared ripgrep dependency from ROS graph readiness
  probes so the lifecycle suite runs on a stock GitHub Ubuntu runner.

## [0.1.0] - 2026-07-23

- Initial reference release based on the validated Watson TM5S-900
  seven-pin air-replay prototype.
