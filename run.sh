#!/bin/bash

DEEPLEARNING_WORKERS_COUNT=2
DEEPLEARNING_WORKER_GPU_COUNT=8
NUM_PARALLEL=$( expr "$DEEPLEARNING_WORKERS_COUNT" '*' "$DEEPLEARNING_WORKER_GPU_COUNT")

# Customize S3 bucket
S3_BUCKET=

# Customize S3 bucket prefix for coco 2014 data tar
# Data tar must contain annotations, train data, validation data, and pretrained models
DATA_TAR_PREFIX=coco2014-data.tar

DATA_DIR=""
FILE_SYS=""
if [ -d "/ebs/" ]
then
	if [ ! -d "/ebs/data" ]
	then
		echo "Copying data tar from $S3_BUCKET to /ebs"
		mpirun -np $DEEPLEARNING_WORKERS_COUNT  --hostfile /kubeflow/openmpi/assets/hostfile \
		-bind-to none --map-by ppr:1:node --allow-run-as-root \
		aws s3 cp s3://$S3_BUCKET/$DATA_TAR_PREFIX /ebs/
		wait $!

		echo "Extracting data tar to /ebs"
		mpirun -np $DEEPLEARNING_WORKERS_COUNT --hostfile /kubeflow/openmpi/assets/hostfile \
		-bind-to none --map-by ppr:1:node --allow-run-as-root \
		tar -xf /ebs/$DATA_TAR_PREFIX -C /ebs 
		wait $!
	fi
	DATA_DIR="/ebs/data"
	FILE_SYS="ebs"
elif [ -d "/efs" ]
then
	if [ ! -d "/efs/coco/data" ]
	then
		echo "Copying data tar from $S3_BUCKET to /efs"
		aws s3 cp s3://$S3_BUCKET/$DATA_TAR_PREFIX /efs/coco

		echo "Extracting data tar to /efs"
		tar -xf /efs/coco/$DATA_TAR_PREFIX -C /efs/coco
	fi
	DATA_DIR="/efs/coco/data"
	FILE_SYS="efs"
fi

SRC_DIR=/tensorpack
DATE=`date '+%Y-%m-%d-%H-%M-%S'`
RUN_ID=mask-rcnn-coco-$NUM_PARALLEL-$FILE_SYS-$DATE

echo "Training started:" `date '+%Y-%m-%d-%H-%M-%S'`

mpirun -np $NUM_PARALLEL \
--hostfile /kubeflow/openmpi/assets/hostfile \
-bind-to none -map-by slot \
--mca plm_rsh_no_tree_spawn 1 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 \
--mca hwloc_base_binding_policy none --mca rmaps_base_mapping_policy slot \
--mca orte_keep_fqdn_hostnames t \
--output-filename /efs/${RUN_ID}-logs \
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
DATA.TRAIN='["train2014"]' \
DATA.VAL=val2014 \
TRAIN.EVAL_PERIOD=25 \
TRAIN.STEPS_PER_EPOCH=500 \
TRAIN.LR_SCHEDULE='[120000, 160000, 180000]' \
BACKBONE.WEIGHTS=$DATA_DIR/pretrained-models/ImageNet-R50-AlignPadding.npz \
BACKBONE.NORM=$BATCH_NORM \
TRAINER=horovod"

echo "Training finished:" `date '+%Y-%m-%d-%H-%M-%S'`
