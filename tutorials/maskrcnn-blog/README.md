# Distributed training of Mask R-CNN model

This tutorial is a companion to the [Mask R-CNN distributed training blog](https://aws.amazon.com/blogs/opensource/distributed-tensorflow-training-using-kubeflow-on-amazon-eks/) on how to to do distributed training of [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN), and [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow) models using [Kubeflow MPI Operator](https://github.com/kubeflow/mpi-operator) on [Amazon Elastic Kubernetes Service (EKS)](https://docs.aws.amazon.com/whitepapers/latest/overview-deployment-options/amazon-elastic-kubernetes-service.html). 

Before proceeding, complete the [Prerequisites](../../README.md#prerequisites) and [Getting started](../../README.md#getting-started).

## Step by step tutorial

This tutorial has following steps:

  1. Upload [COCO 2017 training dataset](https://cocodataset.org/#download) to your [Amazon S3](https://aws.amazon.com/s3/) bucket
  2. Use [Helm charts](https://helm.sh/docs/developing_charts/) to launch training jobs in the EKS cluster 
  3. Use [Jupyter](https://jupyter.org/) notebook to test the trained model
  
### Upload COCO 2017 dataset to Amazon S3 bucket

To download COCO 2017 dataset to your build environment instance, and upload it to your Amazon S3 bucket, replace S3_BUCKET with your bucket name and run following command:

    ./eks-cluster/utils/prepare-s3-bucket.sh S3_BUCKET

**Note:** 
In the script above, by default, data is uploaded under a top-level S3 folder named `ml-platform`. This folder is used in the `import_path` terraform variable in the section [Use Terraform to create infrastructure](#use-terraform-to-create-infrastructure). It is **not recommended** that you change this top-level folder name. if you must change it, do a project wide search for `ml-platform` and replace it with your folder name in the various YAML files.

### Install Helm charts for model training

#### Install Mask-RCNN charts
 
You have two Helm charts available for training Mask-RCNN models. Both these Helm charts use the same Kubernetes namespace, which, by default, is set to `kubeflow`.

To train [TensorPack Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model, install the [maskrcnn](../../charts/machine-learning/training/maskrcnn/Chart.yaml) chart by executing following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn ./charts/machine-learning/training/maskrcnn/ 

By default, the training job uses `p3dn.24xlarge` instance type. If you want to use `p3.16xlarge` instance type instead, use following command:

    helm install --debug maskrcnn ./charts/machine-learning/training/maskrcnn/ \
        --set maskrcnn.gpu_instance_type=p3.16xlarge  --set maskrcnn.tf_device_min_sys_mem_mb=2560

To train [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow) optimized model, install the [maskrcnn-optimized](../../charts/machine-learning/training/maskrcnn-optimized/Chart.yaml) chart by executing following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn-optimized ./charts/machine-learning/training/maskrcnn-optimized/

 By default, the training job uses `p3dn.24xlarge` instance type, with a per GPU batch size of 4. If you want to use `p3.16xlarge` instance type instead, use following command:

    helm install --debug maskrcnn-optimized ./charts/machine-learning/training/maskrcnn-optimized/ \
        --set maskrcnn.gpu_instance_type=p3.16xlarge  --set maskrcnn.tf_device_min_sys_mem_mb=2560 \
        --set maskrcnn.batch_size_per_gpu=2 

### Monitor training

Note, this solution uses [EKS autoscaling](https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html) to automatically scale-up (from zero nodes) and scale-down (to zero nodes) the size of the [EKS managed nodegroup](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html) used for training. So, if currently your training node group has zero nodes, it may take several minutes (or longer, if GPU capacity is transiently unavailable) for the GPU nodes to be `Ready` and for the training pods to reach `Running` state. During this time, the `maskrcnn-launcher-xxxxx` pod may crash and restart automatically several times, and that is nominal behavior. Once the `maskrcnn-launcher-xxxxx` is in `Running` state, replace `xxxxx` with your launcher pod suffix below and execute:

    kubectl logs -f maskrcnn-launcher-xxxxx -n kubeflow

This will show the live training log from the launcher pod. 

### Training logs

Model checkpoints and all training logs are also available on the `shared_fs` file-system  set in `values.yaml`, i.e. `fsx` (default), or `efs`.  For `fsx` (default), access your training logs as follows:

    kubectl apply -f eks-cluster/utils/attach-pvc-fsx.yaml -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc-fsx -- /bin/bash
    cd /fsx
    ls -ltr maskrcnn-*

Type `exit` to exit from the `attach-pvc-fsx` container. 

For `efs`,  access your training logs as follows:

    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash
    cd /efs
    ls -ltr maskrcnn-*

Type `exit` to exit from the `attach-pvc` container. 

### Uninstall Helm charts after training
When training is complete, you may uninstall an installed chart by executing `helm uninstall chart-name`, for example `helm uninstall maskrcnn`. The logs and trained models will be preserved on the shared file system used for training. 

### Test trained model

#### Generate password hash

To password protect [TensorBoard](https://www.tensorflow.org/tensorboard), generate the password hash for your password using the command below:

    htpasswd -c .htpasswd tensorboard
   
Copy the generated password for `tensorboard` from `.htpasswd` file and save it to use in steps below. Finally, clean the generated password hash file:

    rm .htpasswd
    
#### Test TensorPack Mask-RCNN model

To test [TensorPack Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model, customize  [values.yaml](charts/machine-learning/testing/maskrcnn-jupyter/values.yaml), as described below:

1. Use [AWS check ip](http://checkip.amazonaws.com/) to get the public IP of your web browser client. Use this public IP to set `global.source_cidr` as a  `/32` CIDR. This will restrict Internet access to [Jupyter](https://jupyter.org/) notebook and [TensorBoard](https://www.tensorflow.org/tensorboard) services to your public IP.
2. Set `global.log_dir` to the **relative path** of your training log directory, for example, `maskrcnn-XXXX-XX-XX-XX-XX-XX`.
3. Set the generated password for `tensorboard`  as a quoted MD5 hash as shown in the example below:

    `htpasswd: "your-generated-password-hash"`

To install the `maskrcnn-jupyter` chart, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn-jupyter ./charts/machine-learning/testing/maskrcnn-jupyter/

Execute `kubectl logs -f maskrcnn-jupyter-xxxxx -n kubeflow -c jupyter` to display Jupyter log. At the beginning of the Jupyter log, note the **security token** required to access Jupyter service in a browser. 

Execute `kubectl get service maskrcnn-jupyter -n kubeflow` to get the service DNS address. To test the trained model using a Jupyter notebook, access the service in a browser on port 443 using the service DNS and the security token.  Your URL to access the Jupyter service should look similar to the example below:

  https://xxxxxxxxxxxxxxxxxxxxxxxxx.elb.xx-xxxx-x.amazonaws.com?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  
Because the service endpoint in this tutorial uses a **self-signed certificate**, accessing Jupyter service in a browser will display a browser warning. If you deem it appropriate, proceed to access the service. Open the notebook, and run it it to test the trained model. Note, there may not be any trained model checkpoint available at a given time, while training is in progress.

To access TensorBoard via web, use the service DNS address noted above. Your URL to access the TensorBoard service should look similar to the example below:

  https://xxxxxxxxxxxxxxxxxxxxxxxxx.elb.xx-xxxx-x.amazonaws.com:6443/
  
Accessing TensorBoard service in a browser will display a browser warning, because the service endpoint uses a **self-signed certificate**. If you deem it appropriate, proceed to access the service. When prompted for authentication, use the default username `tensorboard`, and your password.

#### Test AWS Mask-RCNN model 

To test [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow)  model, customize  [values.yaml](charts/machine-learning/testing/maskrcnn-optimized-jupyter/values.yaml) file, following the three steps shown for [TensorPack Mask-RCNN model](#test-tensorpack-mask-rcnn-model). Note, the `log_dir` will be different, for example, `maskrcnn-optimized-XXXX-XX-XX-XX-XX-XX`.

To install the `maskrcnn-optimized-jupyter` chart, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn-optimized-jupyter ./charts/machine-learning/testing/maskrcnn-optimized-jupyter/

Execute `kubectl logs -f maskrcnn-optimized-jupyter-xxxxx -n kubeflow -c jupyter` to display Jupyter log. At the beginning of the Jupyter log, note the **security token** required to access Jupyter service in a browser. 

Execute `kubectl get service maskrcnn-optimized-jupyter -n kubeflow` to get the service DNS address. The rest of the steps are the same as for [TensorPack Mask-RCNN model](#test-tensorpack-mask-rcnn-model).

### Uninstall Helm charts after testing
When testing is complete, you may uninstall an installed chart by executing `helm uninstall chart-name`, for example `helm uninstall maskrcnn-jupyter`, or `helm uninstall maskrcnn-optimized-jupyter`.

### (Optional) Stage Data on EFS
The COCO 2017 training data used in the tutorial is **automatically imported** from the `S3_BUCKET` to the FSx for Lustre file-system. However, if you want to use the EFS file-system as the source for your training data, you need to customize `S3_BUCKET` variable in [stage-data.yaml](eks-cluster/utils/stage-data.yaml), and run following command:

    kubectl apply -f eks-cluster/utils/stage-data.yaml -n kubeflow

Execute `kubectl get pods -n kubeflow` to check the status of the staging Pod. Once the status of the Pod is marked `Completed`, data is successfully staged on EFS.