#!/usr/bin/env bash
# validate.sh — Verify theory research session outputs are complete and well-formed.
# This session is a theory/literature survey (no code runs); validation checks document
# completeness and structural integrity.
set -euo pipefail

SESSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERRORS=0

check_file() {
    local path="$1"
    local min_lines="$2"
    local label="$3"
    if [[ ! -f "$path" ]]; then
        echo "FAIL: $label not found at $path"
        ERRORS=$((ERRORS + 1))
        return
    fi
    local lines
    lines=$(wc -l < "$path")
    if [[ "$lines" -lt "$min_lines" ]]; then
        echo "FAIL: $label too short ($lines lines, expected >= $min_lines)"
        ERRORS=$((ERRORS + 1))
    else
        echo "PASS: $label ($lines lines)"
    fi
}

check_contains() {
    local path="$1"
    local pattern="$2"
    local label="$3"
    if grep -qF "$pattern" "$path" 2>/dev/null; then
        echo "PASS: $label contains '$pattern'"
    else
        echo "FAIL: $label missing '$pattern'"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "=== Validating session: $SESSION_DIR ==="
echo ""

# Structural checks
check_file "$SESSION_DIR/notes.md"              100  "notes.md"
check_file "$SESSION_DIR/implementation-plan.md"  30  "implementation-plan.md"
check_file "$SESSION_DIR/validate.sh"              10  "validate.sh"

echo ""
echo "=== Provenance tag checks ==="
check_contains "$SESSION_DIR/notes.md" '[measured'     "notes.md"
check_contains "$SESSION_DIR/notes.md" '[assumed'       "notes.md"
check_contains "$SESSION_DIR/notes.md" '[source-code'   "notes.md"

echo ""
echo "=== Content coverage checks ==="
check_contains "$SESSION_DIR/notes.md" "YIN"          "notes.md covers YIN"
check_contains "$SESSION_DIR/notes.md" "CREPE"        "notes.md covers CREPE"
check_contains "$SESSION_DIR/notes.md" "RMVPE"        "notes.md covers RMVPE"
check_contains "$SESSION_DIR/notes.md" "PESTO"        "notes.md covers PESTO"
check_contains "$SESSION_DIR/notes.md" "FCPE"         "notes.md covers FCPE"
check_contains "$SESSION_DIR/notes.md" "Note Segmen"  "notes.md covers note segmentation"
check_contains "$SESSION_DIR/notes.md" "Voicing"      "notes.md covers voicing detection"
check_contains "$SESSION_DIR/notes.md" "MIDI"         "notes.md covers MIDI assembly"
check_contains "$SESSION_DIR/implementation-plan.md" "RMVPE"         "plan covers RMVPE priority"
check_contains "$SESSION_DIR/implementation-plan.md" "PitchToMIDI"   "plan covers shared utility"
check_contains "$SESSION_DIR/implementation-plan.md" "HMM"           "plan covers HMM upgrade path"

echo ""
if [[ "$ERRORS" -eq 0 ]]; then
    echo "=== All checks passed ==="
    exit 0
else
    echo "=== $ERRORS check(s) failed ==="
    exit 1
fi
