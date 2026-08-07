#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$REPO_DIR/reference/seven_pin/execution"
DESTINATION_DIR="$REPO_DIR/local/execution"

RETIMED_NAME="retimed_seven_pin_air_replay.json"
INGRESS_NAME="tool_aware_ready_ingress.json"
RETIMED_SHA256="8f24ba8c8cf6f814ba12f33e8202cf214b4fd89cd7d9017d11f75d075c5400fb"
INGRESS_SHA256="5c13f72b209781417448f48098c222077a5065809a05b7c39e46d898e713b018"

verify_source() {
  local path="$1"
  local expected="$2"
  local observed=""
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "Missing regular reference artifact: $path" >&2
    exit 2
  fi
  observed="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "Reference artifact hash mismatch: $path" >&2
    echo "Observed: $observed" >&2
    echo "Expected: $expected" >&2
    exit 2
  fi
}

verify_source "$SOURCE_DIR/$RETIMED_NAME" "$RETIMED_SHA256"
verify_source "$SOURCE_DIR/$INGRESS_NAME" "$INGRESS_SHA256"

umask 077
mkdir -p "$DESTINATION_DIR"
chmod 700 "$REPO_DIR/local" "$DESTINATION_DIR"
install -m 600 "$SOURCE_DIR/$RETIMED_NAME" "$DESTINATION_DIR/$RETIMED_NAME"
install -m 600 "$SOURCE_DIR/$INGRESS_NAME" "$DESTINATION_DIR/$INGRESS_NAME"

verify_source "$DESTINATION_DIR/$RETIMED_NAME" "$RETIMED_SHA256"
verify_source "$DESTINATION_DIR/$INGRESS_NAME" "$INGRESS_SHA256"

for path in \
  "$DESTINATION_DIR/$RETIMED_NAME" \
  "$DESTINATION_DIR/$INGRESS_NAME"; do
  mode="$(stat -c '%a' "$path")"
  if [[ "$mode" != "600" ]]; then
    echo "Staged artifact is not mode 0600: $path ($mode)" >&2
    exit 2
  fi
done

echo "Staged private execution fixtures in: $DESTINATION_DIR"
