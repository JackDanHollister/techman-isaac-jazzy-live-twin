#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RVIZ_CONFIG="${DEMO_DIR}/config/watson_display_only.rviz"
TECHMAN_WORKSPACE="${TECHMAN_WORKSPACE:-$HOME/tm2_ws_apt}"

export ROS_DOMAIN_ID=219
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

set +u
source /opt/ros/jazzy/setup.bash
source "${TECHMAN_WORKSPACE}/install/setup.bash"
set -u

exec ros2 run rviz2 rviz2 -d "${RVIZ_CONFIG}" --ros-args \
  -r __ns:=/watson \
  -r /display_planned_path:=/watson/display_planned_path \
  -r /robot_description:=/watson/robot_description \
  -r tf:=/tf \
  -r tf_static:=/tf_static
