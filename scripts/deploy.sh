#!/bin/bash
CACHE=$1
python deploy.py checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage2_nn/model_last.ckpt +keyboard_interactive=true
