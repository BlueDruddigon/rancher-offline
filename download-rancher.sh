#!/bin/bash

source scripts/images.sh

# download images
images=$(cat images.list)
for i in $images; do
    get_image $i
done

# copy list to outputs
cp ./images.list outputs/images