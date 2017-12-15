#!/bin/sh

if [ ! -d data ]; then
	mkdir data
fi

cd data

wget https://s3-us-west-2.amazonaws.com/traffic-light-data/bosch_train_rgb.zip.001
wget https://s3-us-west-2.amazonaws.com/traffic-light-data/bosch_train_rgb.zip.002
wget https://s3-us-west-2.amazonaws.com/traffic-light-data/bosch_train_rgb.zip.003
wget https://s3-us-west-2.amazonaws.com/traffic-light-data/bosch_train_rgb.zip.004

cat bosch_train_rgb.zip.* > bosch_train_rgb.zip

unzip bosch_train_rgb.zip
