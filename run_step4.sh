#!/bin/bash
# Waits for label generation to finish, then runs steps 3 and 4 in order. Log: new_output/step4.log
# Budget (docs/TRAINING_PLAN.md): labels <= 20 h, then dataset + prior ~1 h, feature cache ~3 h, student 12 h, eval ~1 h.
cd /mnt/hdd/WildPRIMA
export HF_HUB_OFFLINE=1
PY=~/anaconda3/envs/prima/bin/python
LOG=new_output/step4.log
stamp() { echo "[$(date '+%F %T')] $*" >> $LOG; }

stamp "waiting for make_labels.py to finish"
while pgrep -f "make_labels.py" > /dev/null; do sleep 60; done
stamp "labels done: $(tail -n +2 new_output/labels/index.csv | wc -l) windows in index.csv"

set -e
stamp "3a collect"; $PY pose_prior.py collect >> $LOG 2>&1
stamp "3b pose prior"; $PY pose_prior.py train >> $LOG 2>&1
stamp "4 feature cache"; $PY student.py cache >> $LOG 2>&1
stamp "4a student training (12 h)"; $PY student.py train --hours 12 --batch 32 >> $LOG 2>&1
stamp "4c evaluation"; $PY student.py eval >> $LOG 2>&1
stamp "all done"
