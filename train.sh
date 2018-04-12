#!/bin/bash
# Derived from ./experiments/scripts/faster_rcnn_end2end.sh

set -x
set -e

# BEGIN Modifiable parameters
CAFFEMODEL="/kaf-frcnn-repo/vgg16_faster_rcnn_iter_150000.caffemodel"
TRAIN_IMDB="cococustom_8888_latest"
GPU_ID=0
ITERS=1

NET="VGG16"
PT_DIR="coco"
# EXTRA_ARGS="--set TRAIN.PROPOSAL_METHOD selective_search"
# END - Modifiable parameters

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/parq_refine_net_w_ss.${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights ${CAFFEMODEL} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
