#!/bin/bash

# Script per avviare il controller Ryu sulla VM Physical Twin
# --ofp-tcp-listen-port 6633: Porta per la connessione degli switch OpenFlow
# --observe-links: Abilita la scoperta automatica della topologia

echo "🚀 Avvio del Controller Ryu (Physical Twin)..."

ryu-manager --ofp-tcp-listen-port 6633 \
    src/controllers/physical_twin_controller.py \
    ryu.app.ofctl_rest \
    --observe-links