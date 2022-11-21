#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM python:3.9-slim

RUN apt-get update && \
    apt-get -y upgrade &&\
    apt-get install -y make automake gcc g++ subversion python3-dev cmake \
    bash \
    patchelf gfortran libc-dev && \
    pip3 install --upgrade pip \
    setuptools \
    wheel \
    numpy \
    pandas \
    swifter \
    tqdm \
    lightgbm \
    gensim \
    torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    pyarrow \
    recpack && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /offlinerl
RUN mkdir /offlinerl/data
WORKDIR /offlinerl

COPY . /offlinerl/
RUN mkdir /offlinerl/pickles
