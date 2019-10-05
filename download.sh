#!/bin/sh

URL_BASE="https://raw.githubusercontent.com/Abdul-Mukit/ycb_video_data_share/master/data/0000/00000"
URL_C="-color.png"
URL_D="-depth.png"

max=9
for i in `seq 1 $max`
do
    wget "$URL_BASE$i$URL_C"
    wget "$URL_BASE$i$URL_D"
done