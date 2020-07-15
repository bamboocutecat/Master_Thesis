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
&& conda install -y rope

RUN apt-get update -y
RUN apt-get install git-all -y

ADD siam /siam/
WORKDIR /siam

