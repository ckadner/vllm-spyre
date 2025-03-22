#syntax=docker/dockerfile:1.7-labs

## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=9.5ARG BASE_UBI_IMAGE_TAG=9.4
ARG PYTHON_VERSION=3.12


## Base Layer ##################################################################
FROM registry.access.redhat.com/ubi9/ubi-minimal:${BASE_UBI_IMAGE_TAG} AS base

ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}
WORKDIR /workspace

# Install some basic utilities
RUN --mount=type=cache,target=/root/.cache/microdnf:rw \
    microdnf update -y && \
    microdnf install -y --setopt=cachedir=/root/.cache/microdnf \
        python${PYTHON_VERSION}-devel \
        python${PYTHON_VERSION}-pip \
        python${PYTHON_VERSION}-wheel \
        git \
        vim \
        gcc \
        g++ \
        kmod \
    && microdnf clean all

RUN ln -sf $(which python${PYTHON_VERSION}) /usr/bin/python && \
    ln -sf $(which pip${PYTHON_VERSION}) /usr/bin/pip


## vLLM Base ###################################################################
FROM base as vllm-base

# Download and install vllm
ENV PIP_CACHE_DIR=/root/.cache/pip
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 https://github.com/vllm-project/vllm.git \
    && cd vllm \
    && git fetch origin pull/14242/head:spyre-workarounds \
    && git checkout spyre-workarounds \
    && python -m pip install --upgrade pip \
    && pip3 install torch=="2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu \
    && python use_existing_torch.py \
    && pip install -r requirements-build.txt \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.7.3 VLLM_TARGET_DEVICE=empty pip install --verbose . --no-build-isolation


## Spyre Base ##################################################################
FROM vllm-base as spyre-base

# Install vllm Spyre plugin
RUN mkdir /workspace/vllm-spyre
COPY --exclude=tests . /workspace/vllm-spyre
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/vllm-spyre && \
    pip install -v -e .
ENV VLLM_PLUGINS=spyre


## Spyre Tests #################################################################
FROM spyre-base as spyre-tests

# set environment variables to run tests
ENV MASTER_ADDR=localhost \
    MASTER_PORT=12355 \
    DISTRIBUTED_STRATEGY_IGNORE_MODULES=WordEmbedding

# Install test dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        sentence-transformers \
        pytest \
        pytest-timeout \
        pytest-forked

# Download models
RUN mkdir -p /models \
    && python -c "from transformers import pipeline; pipeline('text-generation', model='JackFram/llama-160m')" \
    && export VARIANT=$(ls /root/.cache/huggingface/hub/models--JackFram--llama-160m/snapshots/) \
    && ln -s /root/.cache/huggingface/hub/models--JackFram--llama-160m/snapshots/${VARIANT} /models/llama-194m \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-roberta-large-v1')" \
    && export VARIANT=$(ls /root/.cache/huggingface/hub/models--sentence-transformers--all-roberta-large-v1/snapshots/) \
    && ln -s /root/.cache/huggingface/hub/models--sentence-transformers--all-roberta-large-v1/snapshots/${VARIANT} /models/all-roberta-large-v1

COPY tests /workspace/vllm-spyre/tests/


## Spyre Release ###############################################################
FROM spyre-base as spyre-release

CMD ["/bin/bash"]
