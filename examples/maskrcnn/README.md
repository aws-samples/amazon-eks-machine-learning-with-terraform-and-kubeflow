# Training Mask R-CNN model on COCO 2017 dataset

This example illustrates how to use [mpijob-horovod-tensorflow-gpu](../../charts/machine-learning/training/mpijob-horovod-tensorflow-gpu/Chart.yaml) Helm chart to train Mask R-CNN model on [COCO 2017 dataset](https://cocodataset.org/#download).

Before proceeding, complete the [Prerequisites](../../README.md#prerequisites) and [Getting started](../../README.md#getting-started). 

See [What is in the YAML file](../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Download COCO 2017 dataset to FSx for Lustre file-system

To download COCO 2017 dataset to FSx for Lustre file-system folder `/fsx/data/coco2017`, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug coco-2017 \
        ./charts/machine-learning/data-prep/coco-data/  -n kubeflow-user-example-com

You can tail the logs using following command:

    kubectl logs -f coco-data-coco-2017 -n kubeflow-user-example-com

Uninstall the Helm chart on completion:

    helm uninstall coco-2017  -n kubeflow-user-example-com
  
### Mask-RCNN model training

#### Train TensorPack Mask/Faster-RCNN model

To train [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn-tensorpack \
        ./charts/machine-learning/training/mpijob-horovod-tensorflow-gpu/ \
        -f examples/maskrcnn/train-maskrcnn-tensorpack.yaml -n kubeflow-user-example-com


To monitor logs:

    kubectl logs -f mpijob-maskrcnn-tensorpack-launcher-xxxxx -n kubeflow-user-example-com

Uninstall the Helm chart on completion:

    helm uninstall maskrcnn-tensorpack -n kubeflow-user-example-com

#### Train AWS Mask-RCNN model

To train [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow)

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn-aws \
        ./charts/machine-learning/training/mpijob-horovod-tensorflow-gpu/ \
        -f examples/maskrcnn/train-maskrcnn-aws.yaml -n kubeflow-user-example-com

To monitor logs:

    kubectl logs -f mpijob-maskrcnn-aws-launcher-xxxxx  -n kubeflow-user-example-com

Uninstall the Helm chart on completion:

    helm uninstall maskrcnn-aws -n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash

This will put you in a pod attached to the  EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

### Data

Downloaded data is available in `/fsx/data/coco2017` folder.

### Logs

Training `logs` are available in `/efs/home/maskrcnn-tensorpack/logs` and `/efs/home/maskrcnn-aws/logs` folders for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) and [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow) models, respectively. 