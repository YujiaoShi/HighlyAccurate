FROM nvcr.io/nvidia/pytorch:21.12-py3

USER root

# For GKT
RUN pip install hydra-core==1.1.1
RUN pip install opencv-python==4.5.5.64
RUN pip install torch==1.11.0
RUN pip install torchvision==0.12.0
RUN pip install omegaconf==2.1.1
RUN pip install tqdm==4.63.1
RUN pip install pyquaternion==0.9.9
RUN pip install imgaug==0.4.0
RUN pip install torchmetrics==0.6.2
RUN pip install pytorch-lightning==1.6.0
RUN pip install fvcore==0.1.5.post20210825
RUN pip install einops==0.3.2
RUN pip install efficientnet-pytorch==0.7.1
RUN pip install wandb==0.12.11

# For NVIDIA A100 GPU
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# scipy, tensorboard
# RUN pip install scipy
# RUN pip install tensorboard
# RUN pip install -U scikit-learn
# RUN pip install pykitti
# RUN pip install opencv-python
# RUN pip install wandb -qqq

# # build essential, cmake, vim, git
# RUN apt-get update && apt-get -y install cmake
# RUN apt-get update && apt-get install -y vim
# RUN apt-get install -y build-essential
# RUN apt-get update && apt-get install -y git-all

# # for epn
# RUN pip install numpy
# RUN pip install torch
# RUN pip install torchvision
# RUN pip install scikit-image
# RUN pip install scikit-learn
# RUN pip install open3d
# RUN pip install tqdm
# RUN pip install trimesh==3.2.0
# RUN pip install imageio
# RUN pip install plyfile
# RUN pip install parse
# RUN pip install colour