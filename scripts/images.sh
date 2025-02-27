source ./config.sh

IMAGES_DIR=outputs/images
if [ ! -d $IMAGES_DIR ]; then
    mkdir -p $IMAGES_DIR
fi

get_image() {
    image=$1

    tarname="$(echo ${image} | sed s@"/"@"_"@g | sed s/":"/"-"/g)".tar
    zipname="$(echo ${image} | sed s@"/"@"_"@g | sed s/":"/"-"/g)".tar.gz

    if [ ! -e $IMAGES_DIR/$zipname ]; then
        echo "==> Pull $image"
        $sudo $docker pull $image || exit 1

        echo "==> Save $image"
        $sudo $docker save -o $IMAGES_DIR/$tarname $image
        $sudo chown $(whoami) $IMAGES_DIR/$tarname
        chmod 0644 $IMAGES_DIR/$tarname
        gzip -v $IMAGES_DIR/$tarname
    else
        echo "==> Skip $image"
    fi
}