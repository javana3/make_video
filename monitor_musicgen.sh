#!/bin/bash
# musicgen 下载/推理 后台进度监控 — 每 5 min 写一次到 musicgen_progress.log
LOG=/tmp/musicgen_progress.log
CACHE=/c/Users/dfgfd/.cache/huggingface/hub/models--facebook--musicgen-small
TARGET_GB=2.20
PREV_BYTES=0
PREV_TIME=$(date +%s)

echo "[$(date '+%H:%M:%S')] monitor started — target ${TARGET_GB} GB" > $LOG

while true; do
  if ps -ef | grep musicgen_oneshot | grep -v grep > /dev/null 2>&1; then
    SIZE=$(du -sb $CACHE 2>/dev/null | awk '{print $1}')
    SIZE=${SIZE:-0}
    NOW=$(date +%s)
    DT=$((NOW-PREV_TIME))
    DBYTES=$((SIZE-PREV_BYTES))
    LINE=$(awk -v sz=$SIZE -v db=$DBYTES -v dt=$DT -v tgt=$TARGET_GB 'BEGIN{
      mb=sz/1048576; gb=sz/1073741824; pct=gb*100/tgt;
      sp=(dt>0)?db/dt/1048576:0;
      remain=tgt-gb;
      eta=(sp>0)?remain*1024/sp:0;
      em=int(eta/60); es=int(eta)%60;
      printf "size=%.1f MB / %.2f GB · %.1f%% · 速度=%.2f MB/s · ETA=%dm%02ds", mb, gb, pct, sp, em, es
    }')
    LATEST=$(tail -1 /tmp/musicgen.log 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tr -d '\r' | head -c 200)
    echo "[$(date '+%H:%M:%S')] $LINE  log: $LATEST" >> $LOG
    PREV_BYTES=$SIZE
    PREV_TIME=$NOW
  else
    if [ -f workspace/football-match-simulator/runs/49aecf4a/bgm/bgm_final.wav ]; then
      WAVSZ=$(stat -c %s workspace/football-match-simulator/runs/49aecf4a/bgm/bgm_final.wav 2>/dev/null)
      echo "[$(date '+%H:%M:%S')] DONE — bgm_final.wav = ${WAVSZ} bytes" >> $LOG
      exit 0
    else
      echo "[$(date '+%H:%M:%S')] process gone, no bgm_final.wav — likely error, see /tmp/musicgen.log" >> $LOG
      exit 1
    fi
  fi
  sleep 300
done
