#!/usr/bin/env bash
# Usage: ./prog_full.sh [status|start|stop]
PIDFILE="data/.full.pid"

case "${1:-status}" in
    status)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            PID=$(cat "$PIDFILE")
            ELAPSED=$(ps -o etimes= -p "$PID" 2>/dev/null | tr -d ' ')
            echo "RUNNING (pid $PID, ${ELAPSED}s)"
        else
            rm -f "$PIDFILE"
            echo "STOPPED"
        fi
        ;;
    start)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Already running (pid $(cat "$PIDFILE"))"
            exit 1
        fi
        setsid bash run_full.sh > data/full_nohup.log 2>&1 &
        sleep 0.5
        echo "Started (pid $(cat "$PIDFILE"))"
        ;;
    stop)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            PID=$(cat "$PIDFILE")
            kill -- -"$PID" 2>/dev/null
            rm -f "$PIDFILE"
            echo "Stopped (pid $PID)"
        else
            rm -f "$PIDFILE"
            echo "Not running"
        fi
        ;;
    *)
        echo "Usage: $0 [status|start|stop]"
        exit 1
        ;;
esac
