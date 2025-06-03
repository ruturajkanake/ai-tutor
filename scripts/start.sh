#!/usr/bin/env bash
PORT=${1:-8081}
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --log-level info
