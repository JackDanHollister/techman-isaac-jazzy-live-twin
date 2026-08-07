#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SITE_ENV="${TECHMAN_SITE_ENV:-$ARENA_DIR/local/watson-site.env}"
if [[ -f "$SITE_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SITE_ENV"
  set +a
fi

ROBOT_INTERFACE="${TECHMAN_ROBOT_INTERFACE:-enp1s0}"
ROBOT_SOURCE_IP="${TECHMAN_ROBOT_SOURCE_IP:-192.0.2.100}"
ROBOT_IP="${TECHMAN_ROBOT_IP:-192.0.2.23}"

export ROS_DOMAIN_ID=219
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

source_setup() {
  set +u
  source "$1"
  set -u
}

source_setup /opt/ros/jazzy/setup.bash
if [ -f "$HOME/tm2_ws_apt/install/setup.bash" ]; then
  source_setup "$HOME/tm2_ws_apt/install/setup.bash"
elif [ -f "$HOME/tm2_ws/install/setup.bash" ]; then
  source_setup "$HOME/tm2_ws/install/setup.bash"
else
  echo "ERROR: no built Techman ROS 2 workspace was found" >&2
  exit 2
fi

for arg in "$@"; do
  if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
    exec /usr/bin/python3 "$SCRIPT_DIR/run_watson_guarded_demo.py" "$@"
  fi
done

if [ ! -r "/sys/class/net/$ROBOT_INTERFACE/carrier" ] \
  || [ "$(cat "/sys/class/net/$ROBOT_INTERFACE/carrier")" != "1" ]; then
  echo "ERROR: $ROBOT_INTERFACE has no physical Ethernet carrier; check the robot cable/link." >&2
  exit 2
fi

ROUTE="$(ip route get "$ROBOT_IP" 2>/dev/null || true)"
if [[ "$ROUTE" != *"dev $ROBOT_INTERFACE"* || "$ROUTE" != *"src $ROBOT_SOURCE_IP"* ]]; then
  echo "ERROR: Watson is not routed over robot-net/$ROBOT_INTERFACE." >&2
  echo "Observed route: ${ROUTE:-none}" >&2
  exit 2
fi

exec /usr/bin/python3 "$SCRIPT_DIR/run_watson_guarded_demo.py" "$@"
