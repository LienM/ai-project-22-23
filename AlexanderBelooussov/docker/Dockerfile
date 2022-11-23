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
    torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    pyarrow \
    pandas-downcast \
    recpack && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /projectai
RUN mkdir /projectai/data
WORKDIR /projectai

COPY data /projectai/data
COPY lecture4 /projectai/

ENV DATA_DIR=/projectai/data/

CMD ["python3", "main.py", "--frac", "1", "--cv", "--n", "12", "--n_train_weeks", "12", "--verbose", "--ratio", "30"]
