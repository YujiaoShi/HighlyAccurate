container_name=$1

xhost +local:
docker run -it --net=host --shm-size 8G --gpus all  \
  -u $(id -u) \
  -e DISPLAY=$DISPLAY \
  -e QT_GRAPHICSSYSTEM=native \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e XAUTHORITY \
  -e USER=$USER \
  --workdir=/home/$USER/ \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "/etc/passwd:/etc/passwd:rw" \
  -e "TERM=xterm-256color" \
  -v "/home/goroyeh:/home/$USER" \
  -v "/mnt/workspace:/mnt/workspace" \
  --device=/dev/dri:/dev/dri \
  --name=${container_name} \
  --security-opt seccomp=unconfined \
  satellite_slam_gkt/pytorch_env:latest