#!/bin/bash

DEEPLEARNING_WORKERS_COUNT=2
DEEPLEARNING_WORKER_GPU_COUNT=8
NUM_PARALLEL=$( expr "$DEEPLEARNING_WORKERS_COUNT" '*' "$DEEPLEARNING_WORKER_GPU_COUNT")

DATA_DIR="/efs/data"
FILE_SYS="efs"
LOG_DIR=/efs

SRC_DIR=/tensorpack
DATE=`date '+%Y-%m-%d-%H-%M-%S'`
RUN_ID=mask-rcnn-coco-$NUM_PARALLEL-$FILE_SYS-$DATE

EVAL_PERIOD=$( expr "8" '/' "$DEEPLEARNING_WORKERS_COUNT")
BATCH_NORM=FreezeBN

echo "Training started:" `date '+%Y-%m-%d-%H-%M-%S'`

mpirun -np $NUM_PARALLEL \
--hostfile /kubeflow/openmpi/assets/hostfile \
-bind-to none -map-by slot \
--mca plm_rsh_no_tree_spawn 1 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 \
--mca hwloc_base_binding_policy none --mca rmaps_base_mapping_policy slot \
--mca orte_keep_fqdn_hostnames t \
--output-filename /efs/${RUN_ID} \
--allow-run-as-root --display-map --tag-output --timestamp-output \
bash -c "HOROVOD_CYCLE_TIME=0.5 \
HOROVOD_FUSION_THRESHOLD=67108864 \
NCCL_SOCKET_IFNAME=eth0 \
NCCL_MIN_NRINGS=8 \
NCCL_DEBUG=INFO \
python3 $SRC_DIR/examples/FasterRCNN/train.py \
--logdir $LOG_DIR/$RUN_ID/train_log/maskrcnn \
--config MODE_MASK=True \
MODE_FPN=True \
DATA.BASEDIR=$DATA_DIR \
DATA.TRAIN='[\"train2017\"]' \
DATA.VAL=val2017 \
TRAIN.EVAL_PERIOD=$EVAL_PERIOD \
TRAIN.STEPS_PER_EPOCH=1875 \
TRAIN.LR_SCHEDULE='[120000, 160000, 180000]' \
BACKBONE.WEIGHTS=$DATA_DIR/pretrained-models/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=$BATCH_NORM \
TRAINER=horovod"

echo "Training finished:" `date '+%Y-%m-%d-%H-%M-%S'`
