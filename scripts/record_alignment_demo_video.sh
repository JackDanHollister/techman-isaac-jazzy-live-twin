#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SEED="${1:-$(date +%s)}"
RECORDINGS_DIR="$ARENA_DIR/outputs/recordings"
mkdir -p "$RECORDINGS_DIR"

VIDEO_OUT="${VIDEO_OUT:-$RECORDINGS_DIR/pin_alignment_${SEED}.mp4}"
VIDEO_DIR="$(dirname "$VIDEO_OUT")"
mkdir -p "$VIDEO_DIR"
VIDEO_DIR="$(cd "$VIDEO_DIR" && pwd)"
VIDEO_OUT="$VIDEO_DIR/$(basename "$VIDEO_OUT")"

RECORDER_LOG="${RECORDER_LOG:-${VIDEO_OUT%.*}.recorder.log}"
FPS="${FPS:-30}"
RECORD_BACKEND="${RECORD_BACKEND:-auto}"
PORTAL_SOURCE="${PORTAL_SOURCE:-monitor}"
PORTAL_START_TIMEOUT="${PORTAL_START_TIMEOUT:-120}"
DISPLAY_NAME="${DISPLAY:-:0}"
RECORD_OFFSET="${RECORD_OFFSET:-0,0}"
RECORD_SIZE="${RECORD_SIZE:-}"
QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

if [ -z "${ROS_DOMAIN_ID:-}" ]; then
  export ROS_DOMAIN_ID="$((120 + RANDOM % 100))"
else
  export ROS_DOMAIN_ID
fi

is_video_size() {
  [[ "$1" =~ ^[0-9]+x[0-9]+$ ]]
}

show_recorder_log() {
  if [ -f "$RECORDER_LOG" ]; then
    echo ""
    echo "Recorder log tail:"
    tail -n 80 "$RECORDER_LOG" || true
  else
    echo ""
    echo "Recorder log was not created: $RECORDER_LOG"
  fi
}

verify_video() {
  local video_path="$1"
  if [ ! -s "$video_path" ]; then
    echo "Recording failed: no non-empty video was created at: $video_path"
    show_recorder_log
    exit 1
  fi

  if command -v ffprobe >/dev/null 2>&1; then
    if ! ffprobe -v error "$video_path" >/dev/null 2>&1; then
      echo "Recording failed: ffprobe could not read the video at: $video_path"
      show_recorder_log
      exit 1
    fi
  fi
}

detect_record_size() {
  if [ -n "$RECORD_SIZE" ]; then
    return 0
  fi

  if command -v xrandr >/dev/null 2>&1; then
    RECORD_SIZE="$(xrandr 2>/dev/null | awk '
      /^[[:space:]]*[0-9]+x[0-9]+[[:space:]].*\*/ {print $1; exit}
      /^Screen .* current [0-9]+ x [0-9]+,/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "current") {
            width = $(i + 1)
            height = $(i + 3)
            gsub(",", "", height)
            print width "x" height
            exit
          }
        }
      }
    ')"
    if ! is_video_size "$RECORD_SIZE"; then
      RECORD_SIZE=""
    fi
  fi

  if [ -z "$RECORD_SIZE" ] && command -v xdpyinfo >/dev/null 2>&1; then
    RECORD_SIZE="$(xdpyinfo 2>/dev/null | awk '/dimensions:/ {print $2; exit}')"
    if ! is_video_size "$RECORD_SIZE"; then
      RECORD_SIZE=""
    fi
  fi
}

gnome_screencast_supported() {
  command -v gdbus >/dev/null 2>&1 || return 1
  gdbus call \
    --session \
    --dest org.gnome.Shell.Screencast \
    --object-path /org/gnome/Shell/Screencast \
    --method org.freedesktop.DBus.Properties.Get \
    org.gnome.Shell.Screencast \
    ScreencastSupported 2>/dev/null | grep -q true
}

portal_recorder_supported() {
  command -v gst-launch-1.0 >/dev/null 2>&1 || return 1
  command -v gdbus >/dev/null 2>&1 || return 1
  /usr/bin/python3 - <<'PY' >/dev/null 2>&1
import dbus  # noqa: F401
import gi  # noqa: F401
PY
}

choose_backend() {
  case "$RECORD_BACKEND" in
    auto)
      if [ "${XDG_SESSION_TYPE:-}" = "wayland" ] && portal_recorder_supported; then
        RECORD_BACKEND="portal"
      else
        RECORD_BACKEND="x11"
      fi
      ;;
    portal|gnome|x11)
      ;;
    *)
      echo "Unknown RECORD_BACKEND='$RECORD_BACKEND'. Use auto, portal, gnome, or x11."
      exit 2
      ;;
  esac
}

run_demo() {
  AUTO_PLAY="${AUTO_PLAY:-1}" \
  KEEP_OPEN="${KEEP_OPEN:-0}" \
  USE_RVIZ="${USE_RVIZ:-1}" \
  QT_QPA_PLATFORM="$QT_QPA_PLATFORM" \
  ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  "$SCRIPT_DIR/play_random_pin_alignment_demo.sh" "$SEED"
}

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required to record/convert the RViz demo."
  echo "Install it, then rerun this script:"
  echo "  sudo apt update && sudo apt install -y ffmpeg"
  exit 2
fi

choose_backend

rm -f "$VIDEO_OUT" "$RECORDER_LOG"

echo "Recording RViz demo to: $VIDEO_OUT"
echo "Recorder backend: $RECORD_BACKEND"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "Recorder log: $RECORDER_LOG"

if [ "$RECORD_BACKEND" = "gnome" ]; then
  RAW_VIDEO="${RAW_VIDEO:-${VIDEO_OUT%.*}.gnome.webm}"
  rm -f "$RAW_VIDEO"

  echo "GNOME raw capture: $RAW_VIDEO"
  echo ""

  {
    echo "Starting GNOME Shell screencast at $(date --iso-8601=seconds)"
    gdbus call \
      --session \
      --dest org.gnome.Shell.Screencast \
      --object-path /org/gnome/Shell/Screencast \
      --method org.gnome.Shell.Screencast.Screencast \
      "$RAW_VIDEO" \
      "{'draw-cursor': <true>, 'framerate': <$FPS>}"
  } >"$RECORDER_LOG" 2>&1

  if ! head -n 20 "$RECORDER_LOG" | grep -q "^(true,"; then
    echo "Recording failed: GNOME Shell screencast did not start."
    show_recorder_log
    exit 1
  fi

  cleanup_gnome() {
    gdbus call \
      --session \
      --dest org.gnome.Shell.Screencast \
      --object-path /org/gnome/Shell/Screencast \
      --method org.gnome.Shell.Screencast.StopScreencast >>"$RECORDER_LOG" 2>&1 || true
  }
  trap cleanup_gnome EXIT

  sleep 1
  run_demo
  cleanup_gnome
  trap - EXIT
  sleep 1

  verify_video "$RAW_VIDEO"

  case "$VIDEO_OUT" in
    *.webm)
      cp "$RAW_VIDEO" "$VIDEO_OUT"
      ;;
    *)
      {
        echo ""
        echo "Converting GNOME WebM capture to MP4 at $(date --iso-8601=seconds)"
        ffmpeg \
          -y \
          -i "$RAW_VIDEO" \
          -vf "fps=${FPS}" \
          -r "$FPS" \
          -c:v libx264 \
          -preset veryfast \
          -crf 22 \
          -pix_fmt yuv420p \
          "$VIDEO_OUT"
      } >>"$RECORDER_LOG" 2>&1
      ;;
  esac

  verify_video "$VIDEO_OUT"
elif [ "$RECORD_BACKEND" = "portal" ]; then
  RAW_VIDEO="${RAW_VIDEO:-${VIDEO_OUT%.*}.portal.webm}"
  rm -f "$RAW_VIDEO"

  echo "Portal raw capture: $RAW_VIDEO"
  echo "Portal source: $PORTAL_SOURCE"
  echo "A desktop sharing prompt may appear; choose the monitor/window that contains RViz."
  echo ""

  /usr/bin/python3 "$SCRIPT_DIR/record_pipewire_portal.py" \
    "$RAW_VIDEO" \
    --fps "$FPS" \
    --source "$PORTAL_SOURCE" \
    --timeout-s "$PORTAL_START_TIMEOUT" \
    >"$RECORDER_LOG" 2>&1 &
  RECORDER_PID="$!"
  RECORDER_STOPPED=0

  portal_recorder_running() {
    if [ -z "${RECORDER_PID:-}" ]; then
      return 1
    fi
    local stat
    stat="$(ps -p "$RECORDER_PID" -o stat= 2>/dev/null || true)"
    [ -n "$stat" ] && [[ "$stat" != Z* ]]
  }

  stop_portal_recorder() {
    if [ "$RECORDER_STOPPED" -eq 1 ]; then
      return 0
    fi
    RECORDER_STOPPED=1

    if portal_recorder_running; then
      kill -INT "$RECORDER_PID" >/dev/null 2>&1 || true
    fi
    if [ -n "${RECORDER_PID:-}" ]; then
      wait "$RECORDER_PID" >/dev/null 2>&1 || true
    fi
  }

  cleanup_portal() {
    stop_portal_recorder || true
  }
  trap cleanup_portal EXIT

  STARTED=0
  for _ in $(seq 1 "$((PORTAL_START_TIMEOUT * 2))"); do
    if grep -q "Recording WebM:" "$RECORDER_LOG" 2>/dev/null; then
      STARTED=1
      break
    fi
    if ! portal_recorder_running; then
      echo "Recording failed: PipeWire portal recorder stopped before the demo started."
      show_recorder_log
      exit 1
    fi
    sleep 0.5
  done

  if [ "$STARTED" -ne 1 ]; then
    echo "Recording failed: PipeWire portal recorder did not start within ${PORTAL_START_TIMEOUT}s."
    show_recorder_log
    exit 1
  fi

  run_demo
  stop_portal_recorder
  trap - EXIT

  verify_video "$RAW_VIDEO"

  case "$VIDEO_OUT" in
    *.webm)
      cp "$RAW_VIDEO" "$VIDEO_OUT"
      ;;
    *)
      {
        echo ""
        echo "Converting PipeWire WebM capture to MP4 at $(date --iso-8601=seconds)"
        ffmpeg \
          -y \
          -i "$RAW_VIDEO" \
          -vf "fps=${FPS}" \
          -r "$FPS" \
          -c:v libx264 \
          -preset veryfast \
          -crf 22 \
          -pix_fmt yuv420p \
          "$VIDEO_OUT"
      } >>"$RECORDER_LOG" 2>&1
      ;;
  esac

  verify_video "$VIDEO_OUT"
else
  detect_record_size
  if ! is_video_size "$RECORD_SIZE"; then
    echo "Could not detect a valid screen size. Set RECORD_SIZE, for example:"
    echo "  RECORD_SIZE=1920x1080 $0"
    exit 2
  fi

  OFFSET_X="${RECORD_OFFSET%,*}"
  OFFSET_Y="${RECORD_OFFSET#*,}"
  if [[ "$DISPLAY_NAME" == *.* ]]; then
    FFMPEG_DISPLAY="$DISPLAY_NAME"
  else
    FFMPEG_DISPLAY="${DISPLAY_NAME}.0"
  fi
  FFMPEG_INPUT="${FFMPEG_DISPLAY}+${OFFSET_X},${OFFSET_Y}"

  echo "Capture: display=$DISPLAY_NAME size=$RECORD_SIZE offset=$RECORD_OFFSET fps=$FPS"
  echo ""

  ffmpeg \
    -y \
    -video_size "$RECORD_SIZE" \
    -framerate "$FPS" \
    -f x11grab \
    -i "$FFMPEG_INPUT" \
    -c:v libx264 \
    -preset veryfast \
    -crf 22 \
    -pix_fmt yuv420p \
    "$VIDEO_OUT" \
    >"$RECORDER_LOG" 2>&1 &
  FFMPEG_PID="$!"
  FFMPEG_STOPPED=0
  FFMPEG_STATUS=0

  ffmpeg_running() {
    if [ -z "${FFMPEG_PID:-}" ]; then
      return 1
    fi

    local stat
    stat="$(ps -p "$FFMPEG_PID" -o stat= 2>/dev/null || true)"
    [ -n "$stat" ] && [[ "$stat" != Z* ]]
  }

  stop_ffmpeg() {
    if [ "$FFMPEG_STOPPED" -eq 1 ]; then
      return 0
    fi
    FFMPEG_STOPPED=1

    if ffmpeg_running; then
      kill -INT "$FFMPEG_PID" >/dev/null 2>&1 || true
    fi
    if [ -n "${FFMPEG_PID:-}" ]; then
      wait "$FFMPEG_PID" >/dev/null 2>&1 || FFMPEG_STATUS=$?
    fi
  }

  cleanup_x11() {
    stop_ffmpeg || true
  }
  trap cleanup_x11 EXIT

  sleep 2

  if ! ffmpeg_running; then
    wait "$FFMPEG_PID" >/dev/null 2>&1 || FFMPEG_STATUS=$?
    echo "Recording failed: ffmpeg stopped before the demo started (exit $FFMPEG_STATUS)."
    show_recorder_log
    exit 1
  fi

  run_demo
  stop_ffmpeg
  trap - EXIT

  verify_video "$VIDEO_OUT"
fi

VIDEO_SIZE="$(du -h "$VIDEO_OUT" | awk '{print $1}')"
echo ""
echo "Wrote video: $VIDEO_OUT ($VIDEO_SIZE)"
echo "Recorder log: $RECORDER_LOG"
