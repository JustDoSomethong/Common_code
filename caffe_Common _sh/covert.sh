for name in ./examples/caffe_example/images/*.bmp; do     
    convert -resize 256x256\! $name $name 
done
