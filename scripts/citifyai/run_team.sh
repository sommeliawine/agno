#!/bin/bash
# This script handles the execution of the Agno team runner script
# It ensures that paths with spaces are properly handled

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Activate the virtual environment
source "$PROJECT_ROOT/agno_venv/bin/activate"

# Run the Python script with the provided arguments
python "$SCRIPT_DIR/run_team.py" "$@"
