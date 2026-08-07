#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="$SCRIPT_DIR/run_watson_multi_pin_air_replay.py"

use_existing_stack=0
offline_only=0
help_only=0
forward_args=()
for argument in "$@"; do
  case "$argument" in
    --use-existing-stack)
      use_existing_stack=1
      ;;
    --offline-validate)
      offline_only=1
      forward_args+=("$argument")
      ;;
    --help|-h)
      help_only=1
      forward_args+=("$argument")
      ;;
    --mode|--mode=*|--namespace|--namespace=*|\
    --retimed-artifact|--retimed-artifact=*|\
    --ingress-artifact|--ingress-artifact=*|\
    --state-timeout|--state-timeout=*|\
    --service-timeout|--service-timeout=*|\
    --execution-timeout|--execution-timeout=*|\
    --arm-token|--arm-token=*|\
    --gripper-token|--gripper-token=*|\
    --confirm-cell-clear|--resume-at-reviewed-ready|--hil-events|\
    --report|--report=*)
      forward_args+=("$argument")
      ;;
    --*)
      echo "ERROR: unknown or abbreviated option: $argument" >&2
      exit 2
      ;;
    *)
      forward_args+=("$argument")
      ;;
  esac
done

# These routes deliberately happen before ROS sourcing, NIC inspection, graph
# discovery, process launch, or any transport construction.
if [ "$help_only" -eq 1 ]; then
  echo "Wrapper option: --use-existing-stack (check/dry-run only; execute owns its stack)"
  exec /usr/bin/python3 "$RUNNER" "${forward_args[@]}"
fi
if [ "$offline_only" -eq 1 ]; then
  exec /usr/bin/python3 "$RUNNER" "${forward_args[@]}"
fi

execute_requested=0
for ((argument_index=0; argument_index<${#forward_args[@]}; argument_index++)); do
  argument="${forward_args[$argument_index]}"
  if [ "$argument" = "--mode=execute" ]; then
    execute_requested=1
  elif [ "$argument" = "--mode" ] \
    && [ $((argument_index + 1)) -lt ${#forward_args[@]} ] \
    && [ "${forward_args[$((argument_index + 1))]}" = "execute" ]; then
    execute_requested=1
  fi
done
if [ "$use_existing_stack" -eq 1 ] && [ "$execute_requested" -eq 1 ]; then
  echo "ERROR: execute mode must use the wrapper-owned, provenance-gated Watson stack." >&2
  exit 2
fi

ROBOT_INTERFACE="enp1s0"
ROBOT_SOURCE_IP="192.0.2.100"
ROBOT_IP="192.0.2.23"
COMPUTE_BOX_IP="192.0.2.1"

SITE_ENV="${TECHMAN_SITE_ENV:-$ARENA_DIR/local/watson-site.env}"
if [[ -f "$SITE_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SITE_ENV"
  set +a
fi

ROBOT_INTERFACE="${TECHMAN_ROBOT_INTERFACE:-$ROBOT_INTERFACE}"
ROBOT_SOURCE_IP="${TECHMAN_ROBOT_SOURCE_IP:-$ROBOT_SOURCE_IP}"
ROBOT_IP="${TECHMAN_ROBOT_IP:-$ROBOT_IP}"
COMPUTE_BOX_IP="${ONROBOT_COMPUTE_BOX_IP:-$COMPUTE_BOX_IP}"

carrier_file="/sys/class/net/$ROBOT_INTERFACE/carrier"
if [ ! -r "$carrier_file" ] || [ "$(<"$carrier_file")" != "1" ]; then
  echo "ERROR: $ROBOT_INTERFACE does not have carrier." >&2
  exit 2
fi
for target in "$ROBOT_IP" "$COMPUTE_BOX_IP"; do
  route="$(ip -4 route get "$target" 2>/dev/null || true)"
  if [[ " $route " != *" dev $ROBOT_INTERFACE "* ]] \
    || [[ " $route " != *" src $ROBOT_SOURCE_IP "* ]]; then
    echo "ERROR: route to $target must use dev $ROBOT_INTERFACE src $ROBOT_SOURCE_IP." >&2
    echo "Observed: ${route:-no route}" >&2
    exit 2
  fi
done

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
else
  echo "ERROR: native Techman Jazzy workspace is missing" >&2
  exit 2
fi

launch_pid=""
launch_pgid=""
runner_pid=""
runtime_log=""
RUNNER_GUARDED_GRACE_ATTEMPTS=360
RUNNER_GUARDED_GRACE_DELAY_SECONDS=0.25
RUNNER_FINAL_TERM_ATTEMPTS=40
RUNNER_FINAL_TERM_DELAY_SECONDS=0.25

pid_is_alive() {
  local pid="$1"
  local stat_line=""
  local stat_tail=""
  local state=""
  if [ -n "$pid" ] && [ -r "/proc/$pid/stat" ]; then
    stat_line="$(<"/proc/$pid/stat")" 2>/dev/null || stat_line=""
    if [ -n "$stat_line" ]; then
      stat_tail="${stat_line##*) }"
      state="${stat_tail%% *}"
    fi
    # kill -0 can continue to succeed for an unreaped child. Treat a zombie
    # as exited so the owning shell immediately calls wait below.
    if [ "$state" = "Z" ] || [ "$state" = "X" ]; then
      return 1
    fi
  fi
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

process_group_is_alive() {
  local pgid="$1"
  if [ -n "$launch_pid" ] \
    && [ "$pgid" = "$launch_pgid" ] \
    && ! pid_is_alive "$launch_pid"; then
    # Reap an exited launch leader before probing the negative PGID. Otherwise
    # a group containing only that direct-child zombie can look permanently
    # alive to kill -0 and trigger a false forced-cleanup path.
    wait "$launch_pid" 2>/dev/null || true
    launch_pid=""
  fi
  [ -n "$pgid" ] && kill -0 -- "-$pgid" 2>/dev/null
}

wait_for_pid_exit() {
  local pid="$1"
  local attempts="$2"
  local delay_seconds="$3"
  local attempt_index
  for ((attempt_index=0; attempt_index<attempts; attempt_index++)); do
    if ! pid_is_alive "$pid"; then
      wait "$pid" 2>/dev/null || true
      return 0
    fi
    sleep "$delay_seconds"
  done
  return 1
}

wait_for_process_group_exit() {
  local pgid="$1"
  local attempts="$2"
  local delay_seconds="$3"
  local attempt_index
  for ((attempt_index=0; attempt_index<attempts; attempt_index++)); do
    if ! process_group_is_alive "$pgid"; then
      return 0
    fi
    sleep "$delay_seconds"
  done
  return 1
}

stop_runner() {
  local first_signal="${1:-TERM}"
  local pid="$runner_pid"
  if [ -z "$pid" ]; then
    return 0
  fi

  if pid_is_alive "$pid"; then
    kill "-$first_signal" "$pid" 2>/dev/null || true
    # The Python runner uses this interval to cancel any accepted MoveIt goal,
    # stop/recheck the gripper, prove the arm stationary, and write its report.
    # Every forwarded signal, including SIGTERM, gets the same full 90-second
    # guarded-recovery interval before escalation is considered.
    if ! wait_for_pid_exit \
      "$pid" \
      "$RUNNER_GUARDED_GRACE_ATTEMPTS" \
      "$RUNNER_GUARDED_GRACE_DELAY_SECONDS"; then
      if [ "$first_signal" != "TERM" ]; then
        echo "WARNING: guarded runner did not exit after SIG$first_signal; sending SIGTERM" >&2
        kill -TERM "$pid" 2>/dev/null || true
      else
        echo "WARNING: guarded runner is still active after the full SIGTERM recovery interval" >&2
      fi
      if ! wait_for_pid_exit \
        "$pid" \
        "$RUNNER_FINAL_TERM_ATTEMPTS" \
        "$RUNNER_FINAL_TERM_DELAY_SECONDS"; then
        echo "WARNING: guarded runner did not exit after SIGTERM; forcing cleanup" >&2
        kill -KILL "$pid" 2>/dev/null || true
        wait_for_pid_exit "$pid" 20 0.10 || true
      fi
    fi
  else
    wait "$pid" 2>/dev/null || true
  fi
  runner_pid=""
}

stop_owned_stack() {
  local pid="$launch_pid"
  local pgid="$launch_pgid"
  if [ -z "$pgid" ]; then
    if [ -n "$pid" ]; then
      wait "$pid" 2>/dev/null || true
    fi
    launch_pid=""
    launch_pgid=""
    return 0
  fi

  # Probe and signal the complete setsid-created group. The ros2 launch leader
  # can exit while driver/MoveIt descendants remain alive in the same group.
  if process_group_is_alive "$pgid"; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    if ! wait_for_process_group_exit "$pgid" 40 0.25; then
      echo "WARNING: owned Watson bring-up group did not exit after SIGTERM; forcing cleanup" >&2
      kill -KILL -- "-$pgid" 2>/dev/null || true
      wait_for_process_group_exit "$pgid" 20 0.10 || true
    fi
  fi
  if [ -n "$pid" ] && ! pid_is_alive "$pid"; then
    wait "$pid" 2>/dev/null || true
  fi
  launch_pid=""
  launch_pgid=""
}

cleanup_on_exit() {
  local status=$?
  trap - EXIT INT TERM HUP
  stop_runner TERM
  stop_owned_stack
  exit "$status"
}

handle_signal() {
  local signal_name="$1"
  local signal_status="$2"
  # A second signal must not interrupt the ordered runner-then-stack teardown.
  trap '' INT TERM HUP
  trap - EXIT
  stop_runner "$signal_name"
  stop_owned_stack
  exit "$signal_status"
}

trap cleanup_on_exit EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP

if [ "$use_existing_stack" -eq 0 ]; then
  if pgrep -x tm_driver >/dev/null 2>&1 || pgrep -x move_group >/dev/null 2>&1; then
    echo "ERROR: a tm_driver or move_group process already exists." >&2
    echo "Stop the conflicting stack or rerun with --use-existing-stack after auditing it." >&2
    exit 2
  fi

  existing_nodes="$(timeout 3 ros2 node list --no-daemon 2>/dev/null || true)"
  if grep -Eq '^/watson/' <<<"$existing_nodes"; then
    echo "ERROR: an existing /watson ROS graph is already visible." >&2
    echo "Use --use-existing-stack only if that graph was deliberately launched." >&2
    exit 2
  fi

  runtime_dir="$ARENA_DIR/outputs/watson_guarded_demo/runtime_logs"
  umask 077
  mkdir -p "$runtime_dir"
  chmod 700 "$runtime_dir"
  runtime_log="$(mktemp "$runtime_dir/seven_pin_bringup_XXXXXXXX.log")"
  chmod 600 "$runtime_log"

  setsid ros2 launch tm5s_moveit_config watson_bringup.launch.py \
    namespace:=watson \
    robot_ip:="$ROBOT_IP" \
    allow_trajectory_execution:=true >"$runtime_log" 2>&1 &
  launch_pid=$!
  launch_pgid="$launch_pid"

  ready=0
  for _ in $(seq 1 80); do
    if ! pid_is_alive "$launch_pid"; then
      echo "ERROR: owned Watson bring-up exited before readiness." >&2
      tail -80 "$runtime_log" >&2 || true
      exit 2
    fi
    nodes="$(timeout 3 ros2 node list --no-daemon 2>/dev/null || true)"
    if grep -Fqx '/watson/tm_driver_node' <<<"$nodes" \
      && grep -Fqx '/watson/move_group' <<<"$nodes" \
      && grep -Fqx '/watson/robot_state_publisher' <<<"$nodes"; then
      ready=1
      break
    fi
    sleep 0.5
  done
  if [ "$ready" -ne 1 ]; then
    echo "ERROR: owned Watson bring-up did not expose the exact required nodes." >&2
    tail -80 "$runtime_log" >&2 || true
    exit 2
  fi
  echo "Owned Watson bring-up ready. Private log: $runtime_log"
fi

runner_status=0
/usr/bin/env --default-signal=INT,TERM,HUP \
  /usr/bin/python3 "$RUNNER" "${forward_args[@]}" &
runner_pid=$!
if wait "$runner_pid"; then
  runner_status=0
else
  runner_status=$?
fi
runner_pid=""
exit "$runner_status"
