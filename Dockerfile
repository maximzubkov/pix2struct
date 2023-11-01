FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential \
                       ca-certificates \
                       wget \
                       curl \
                       unzip \
                       ssh \
                       git \
                       vim \
                       jq

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV TOKENIZERS_PARALLELISM="true"
ENV WANDB_PROJECT=pix2struct
ENV WANDB_WATCH=false

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.11-full
RUN apt-get clean
RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python -m pip install --upgrade pip

WORKDIR /pix2struct
COPY ./setup.py .
RUN pip install -e ."[dev]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY ./pix2struct /pix2struct
RUN git submodule init && git submodule update

RUN pip cache purge
