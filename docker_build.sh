#!/bin/bash

DOCKERFILE=Dockerfile
IMAGE_NAME=rch_seg_image

distro="focal"

echo "Dockerfile: $DOCKERFILE"

# export BUILDKIT_PROGRESS=plain
export DOCKER_BUILDKIT=1
docker build \
    -f $DOCKERFILE \
    --ssh default \
    -t $IMAGE_NAME:$distro .