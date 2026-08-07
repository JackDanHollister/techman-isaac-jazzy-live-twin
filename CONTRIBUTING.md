# Contributing

Engineering rules for changes to this repository.

- Read `README.md` and the relevant document under `docs/` before changing a
  runtime boundary.
- Keep preview as the default mode.
- Do not contact a robot, ROS graph, or OnRobot Compute Box during ordinary
  tests.
- Physical execution requires explicit user authorisation for that run.
- Keep Isaac free of robot command publishers, service clients, and action
  clients; the guarded wrapper remains the sole physical authority.
- Preserve fail-closed validation, exact hashes, and stop-recovery ordering.
- Run `./scripts/stage_reference_execution.sh`, the offline validation, and
  `pytest -q` after changing a reference artifact or physical runner.
- Update `README.md` and `CHANGELOG.md` for significant changes.
- Do not commit generated outputs, private reports, raw scans, credentials, or
  local environments.
- Use a GitHub noreply identity for public-facing commit metadata; do not
  expose private email addresses.
