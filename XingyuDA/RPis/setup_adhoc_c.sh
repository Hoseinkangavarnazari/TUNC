#!/bin/bash

sudo systemctl stop NetworkManager && \
sudo systemctl disable NetworkManager && \

sudo ip link set wlan0 down && \
sudo iwconfig wlan0 mode ad-hoc && \
sudo iwconfig wlan0 essid 'xing-ad-hoc' && \
sudo iwconfig wlan0 key 1234ABCD56 && \

sudo ip link set wlan0 up && \
sudo ip addr add 192.168.2.3/24 dev wlan0
