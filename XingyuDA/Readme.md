# Install Ansible

```sh
# on control node:
$ sudo apt install ansible

# need to ssh to each managed node via password once

# on remote nodes:
$ sudo apt install openssh-server
$ sudo systemctl start sshd
```

# Ad-Hoc Connection

This `setup_adhoc_a.sh` switches the network mode of node A to ad-hoc mode. For other nodes, just change the IP address:

```sh
$ sudo systemctl stop NetworkManager
$ sudo systemctl disable NetworkManager

# wifi interface: wlan0, essid: xing-ad-hoc
$ sudo ip link set wlan0 down
$ sudo iwconfig wlan0 mode ad-hoc
$ sudo iwconfig wlan0 essid 'xing-ad-hoc'
sudo iwconfig wlan0 key 1234ABCD56 && \

$ sudo ip link set wlan0 up
$ sudo ip addr add 192.168.2.1/24 dev wlan0
```

- All nodes should have the same `essid` but different ip addresses with the same subnet mask.

Make this script autorun at startup:

```sh
$ sudo mv ./setup_adhoc_a.sh /usr/local/bin/
$ sudo chmod a+x /usr/local/bin/setup_adhoc_a.sh
$ sudo crontab -e

# add the following line to the end of the file
@reboot /usr/local/bin/setup_adhoc_a.sh
```

# Start A Testbed Run

At ansible control node, run:

```sh
$ cd NoteOnGithub/Diplomarbeit/Codes/ansibles             # cd to the ansible playbooks folder
$ ansible-playbook -f 30 -i inventory.ini init.yaml
$ ansible-playbook -f 30 -i inventory.ini run_script.yaml -e "KD=<value1> stra=<value2> batch=<value3>"
$ ansible-playbook -f 30 -i inventory.ini fetch.yaml      # fetch logs
$ ansible-playbook -f 30 -i inventory.ini poweroff.yaml   # poweroff remotes
```

- `KD` is key distribution machanism, `stra` is recoding strategy, `batch` is the number of this run.
