#!/bin/bash
# run_server.sh
#
# Starts the House Price model serving API.
# Usage:
#   ./run_server.sh

cd "$(dirname "$0")"

echo "Starting House Price serving API on http://localhost:8000 ..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload