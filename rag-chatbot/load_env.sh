#!/bin/bash
# Source environment variables from .env file
# Usage: source load_env.sh

if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | sed 's/#.*//g' | xargs)
    echo "✅ Environment variables loaded"
else
    echo "❌ No .env file found in current directory"
fi