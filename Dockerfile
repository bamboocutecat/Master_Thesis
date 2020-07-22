# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

## install conda env
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version


RUN conda create --name workenv python=3.7 -y
ENV PATH /opt/conda/envs/workenv/bin:$PATH
# 開啟root shell
SHELL [ "/bin/bash", "--login", "-c" ]    


RUN source activate workenv\
&& conda install -y pylint\
&& conda install -y tensorflow\
&& conda install -y rope\
&& conda install -y autopep8

RUN apt-get update -y
RUN apt-get install git -y

ADD siam /siam/
WORKDIR /siam


## env set up
################################################################

## nvidia gpu docker setting
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
# sudo systemctl restart docker
# docker run --gpus all nvidia/cuda:10.0-base nvidia-smi

##docker environment setting
# sudo groupadd docker
# sudo usermod -aG docker $USER

## GUI settings
# xhost +local:root

## docker run cmd settings
# docker run --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --device=/dev/video0:/dev/video0 -it visual_tracking /bin/bash
################################################################