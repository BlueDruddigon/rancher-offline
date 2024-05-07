#!/bin/bash

run() {
    chmod u=rwx,g=rx,o=rx $*
    echo "=> Running: $*"
    $* || {
        echo "Failed in : $*"
        exit 1
    }
}

run ./download-rancher.sh
run ./copy-target-scripts.sh

