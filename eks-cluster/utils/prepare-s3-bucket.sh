#!/bin/bash

[[ $# -ne 1 ]] && echo "usage:  $0 s3-bucket" && exit 1

S3_BUCKET=$1

S3_PREFIX=ml-platform/data/coco2017

# Stage directory must be on a volume with atleast 100 GB available space
STAGE_DIR=$HOME/stage/data/coco2017

if [ -e $STAGE_DIR ]
then
echo "$STAGE_DIR already exists"
exit 1
fi

mkdir -p $STAGE_DIR

wget -O $STAGE_DIR/train2017.zip http://images.cocodataset.org/zips/train2017.zip
unzip $STAGE_DIR/train2017.zip  -d $STAGE_DIR
rm $STAGE_DIR/train2017.zip

wget -O $STAGE_DIR/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip $STAGE_DIR/val2017.zip -d $STAGE_DIR
rm $STAGE_DIR/val2017.zip

wget -O $STAGE_DIR/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip $STAGE_DIR/annotations_trainval2017.zip -d $STAGE_DIR
rm $STAGE_DIR/annotations_trainval2017.zip

mkdir $STAGE_DIR/pretrained-models
wget -O $STAGE_DIR/pretrained-models/ImageNet-R50-AlignPadding.npz http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz

aws s3 cp --recursive $STAGE_DIR s3://$S3_BUCKET/$S3_PREFIX
rm -rf $STAGE_DIR
