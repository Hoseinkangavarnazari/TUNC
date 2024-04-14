#!/bin/bash

sudo systemctl stop NetworkManager && \
sudo systemctl disable NetworkManager && \

sudo ip link set wlp4s0 down && \
sudo iwconfig wlp4s0 mode ad-hoc && \
sudo iwconfig wlp4s0 essid 'xing-ad-hoc' && \
sudo iwconfig wlp4s0 key 1234ABCD56 && \

sudo ip link set wlp4s0 up && \
sudo ip addr add 192.168.2.1/24 dev wlp4s0
