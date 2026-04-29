#!/bin/bash
set -e

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example — set your HF_TOKEN before running."
    exit 1
fi

uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
