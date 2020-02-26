#!/bin/bash

set -x

trap 'kill $(jobs -p)' EXIT

mkdir -p logs
timestamp=$(date '+%m%d-%H%M%S')

python -u ./train_tempo_classifier.py -r nn-bad-679.pt > "logs/${timestamp}-bad.out" &
python -u ./train_tempo_classifier.py -r nn-ssa-131.pt > "logs/${timestamp}-ssa.out" &

#networks=(ssa bad)
#for n in "${networks[@]}"; do
#  python -u ./train_tempo_classifier.py -n $n > "logs/${timestamp}-${n}.out" &
#done

wait
