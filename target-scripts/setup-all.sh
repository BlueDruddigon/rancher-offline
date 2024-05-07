#!/bin/bash

run() {
    echo "=> Running: $*"
    $* || {
        echo "Failed in : $*"
        exit 1
    }
}

# load and push all images to registry
run ./load-push-all-images.sh
run ./start-rancher.sh