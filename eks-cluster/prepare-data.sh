#!/bin/bash

# Uncomment one of the options below
# For EFS uncomment below
DATA_DIR=/efs
# For FSX uncomment below
#DATA_DIR=/fsx
# For EBS uncomment below
#DATA_DIR=$HOME


# Customize Stage DIR
# Stage directory must be on EBS volume with 100 GB available space
STAGE_DIR=$HOME/stage

if [ -e $STAGE_DIR ]
then
echo "$STAGE_DIR already exists"
exit 1
fi

if [ -e $DATA_DIR/data ]
then
echo "$DATA_DIR/data already exists"
exit 1
fi

mkdir -p $STAGE_DIR/data 
mkdir -p $DATA_DIR/data

wget -O $STAGE_DIR/data/train2017.zip http://images.cocodataset.org/zips/train2017.zip
unzip $STAGE_DIR/data/train2017.zip  -d $DATA_DIR/data
rm $STAGE_DIR/data/train2017.zip

wget -O $STAGE_DIR/data/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip $STAGE_DIR/data/val2017.zip -d $DATA_DIR/data
rm $STAGE_DIR/data/val2017.zip

wget -O $STAGE_DIR/data/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip $STAGE_DIR/data/annotations_trainval2017.zip -d $DATA_DIR/data
rm $STAGE_DIR/data/annotations_trainval2017.zip

mkdir $DATA_DIR/data/pretrained-models
wget -O $DATA_DIR/data/pretrained-models/ImageNet-R50-AlignPadding.npz http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz 

if [ -f ./run.sh ] 
then
	cp run.sh $DATA_DIR/
fi
