#!/bin/bash
CACHE=$1
OUTPUT_DIR=$2

cleanup() {
    echo "Interrupted. Cleaning up..."
    kill $pid2 $pid1
    pkill -f 'deploy.py'
    pkill -f 'ros2 bag record'
    exit 1
}
ros2 bag record -e '^/digit|^/allegroHand|^/franka/joint_states|^/(top|left|right)_camera/color/image_raw/compressed$|tf|^/object_marker$' -s mcap -o $2 &
pid2=$!
echo "Press ctrl+c to stop recording"
echo "Recording bag started"

python deploy.py checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage2_nn/model_last.ckpt +keyboard_interactive=false +pose_topic="object_marker" &
pid1=$!

trap cleanup EXIT

sleep 120s
