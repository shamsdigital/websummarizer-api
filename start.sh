#!/bin/bash
# websummarizer-api/start.sh
gunicorn -b 0.0.0.0:$PORT main:app -w 4 -k uvicorn.workers.UvicornWorker
