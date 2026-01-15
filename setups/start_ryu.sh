#!/bin/bash

echo "Starting the Ryu Controller (Physical Twin)..."

ryu-manager --ofp-tcp-listen-port 6633 \
    src/controllers/physical_twin_controller.py \
    ryu.app.ofctl_rest \
    --observe-links