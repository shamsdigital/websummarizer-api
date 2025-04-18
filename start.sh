#!/bin/bash
# websummarizer-api/start.sh
uvicorn main:app --host 0.0.0.0 --port $PORT
