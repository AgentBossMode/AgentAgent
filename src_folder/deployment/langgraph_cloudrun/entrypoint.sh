#!/bin/sh

# Check for environment variable and set command accordingly
COMMAND = "langgraph dev --port $PORT"

# Execute the command
exec $COMMAND