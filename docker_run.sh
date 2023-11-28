#!/bin/bash

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

echo "Running Docker Container"
CONTAINER_NAME=rch_seg
IMAGE_NAME=rch_seg_image
# Get distro of the built image
distro="focal"

echo "Running in $distro"

# Check if there is an already running container with the same distro
full_container_name="${CONTAINER_NAME}_${distro}"
running_container="$(docker container ls --all | grep $full_container_name)"
if [ -z "$running_container" ]; then
  echo "Running $full_container_name for the first time!"
else
  echo "Found an open $full_container_name container. Starting and attaching!"
  eval "docker start $full_container_name"
  eval "docker attach $full_container_name"
  exit 0
fi

# Check if using GPU
gpu_enabled="--gpus all"

docker run \
  $run_args \
  -it \
  --network host \
  --privileged \
  $gpu_enabled \
  --volume=$XSOCK:$XSOCK:rw \
  --volume=$XAUTH:$XAUTH:rw \
  --volume=$(pwd):/rch-seg \
  --env="XAUTHORITY=${XAUTH}" \
  --env DISPLAY=$DISPLAY \
  --env TERM=xterm-256color \
  --name $full_container_name \
  $IMAGE_NAME:$distro \
  /bin/bash