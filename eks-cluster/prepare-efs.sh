#!/bin/bash

EFS_DIR=/efs

# Customize Stage DIR
# Stage directory must be on EBS volume with 100 GB available space
STAGE_DIR=$HOME/stage

if [ -e $STAGE_DIR ]
then
echo "$STAGE_DIR already exists"
exit 1
fi

if [ -e $EFS_DIR/data ]
then
echo "$EFS_DIR/data already exists"
exit 1
fi

mkdir -p $STAGE_DIR/data 
mkdir -p $EFS_DIR/data

wget -O $STAGE_DIR/data/train2017.zip http://images.cocodataset.org/zips/train2017.zip
unzip $STAGE_DIR/data/train2017.zip  -d $EFS_DIR/data
rm $STAGE_DIR/data/train2017.zip

wget -O $STAGE_DIR/data/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip $STAGE_DIR/data/val2017.zip -d $EFS_DIR/data
rm $STAGE_DIR/data/val2017.zip

wget -O $STAGE_DIR/data/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip $STAGE_DIR/data/annotations_trainval2017.zip -d $EFS_DIR/data
rm $STAGE_DIR/data/annotations_trainval2017.zip

mkdir $EFS_DIR/data/pretrained-models
wget -O $EFS_DIR/data/pretrained-models/ImageNet-R50-AlignPadding.npz http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz 

cp run.sh $EFS_DIR/
