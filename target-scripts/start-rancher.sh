#!/bin/bash

source ./config.sh

BASEDIR="."
if [ ! -d images ] && [ -d ../outputs ]; then
    BASEDIR="../outputs"  # for tests
fi
BASEDIR=$(cd $BASEDIR; pwd)

RANCHER_IMAGE=$(cat ${BASEDIR}/images/images.list)

echo "===> Start Rancher"
sudo /usr/local/bin/nerdctl run -d \
    --network host \
    --restart always \
    --name rancher-server \
    -p 6860:80 -p 6868:443 \
    --privileged \
    ${RANCHER_IMAGE} || exit 1
