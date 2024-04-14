#!/bin/bash

cd && \
source ~/venvXing/bin/activate && \

cd ~/NoteOnGithub/ && \
git restore . && git clean -df && \
cd Diplomarbeit/Codes/ansibles && \
ansible-playbook -f 30 -i inventory.ini init.yaml && \
ansible-playbook -f 30 -i inventory.ini run_script.yaml -e "KD=mhd925 stra=rand batch=10001" && \
cd ../RPis && \
python3 A.py --KD mhd925 --stra rand --batch 10001 && \
cd ../ansibles && \
ansible-playbook -f 30 -i inventory.ini fetch.yaml && \
cd && \


cd ~/NoteOnGithub/ && \
git restore . && git clean -df && \
cd Diplomarbeit/Codes/ansibles && \
ansible-playbook -f 30 -i inventory.ini init.yaml && \
ansible-playbook -f 30 -i inventory.ini run_script.yaml -e "KD=mhd925 stra=last batch=10001" && \
cd ../RPis && \
python3 A.py --KD mhd925 --stra last --batch 10001 && \
cd ../ansibles && \
ansible-playbook -f 30 -i inventory.ini fetch.yaml && \
cd && \


cd ~/NoteOnGithub/ && \
git restore . && git clean -df && \
cd Diplomarbeit/Codes/ansibles && \
ansible-playbook -f 30 -i inventory.ini init.yaml && \
ansible-playbook -f 30 -i inventory.ini run_script.yaml -e "KD=cff stra=rand batch=10001" && \
cd ../RPis && \
python3 A.py --KD cff --stra rand --batch 10001 && \
cd ../ansibles && \
ansible-playbook -f 30 -i inventory.ini fetch.yaml && \
cd && \


cd ~/NoteOnGithub/ && \
git restore . && git clean -df && \
cd Diplomarbeit/Codes/ansibles && \
ansible-playbook -f 30 -i inventory.ini init.yaml && \
ansible-playbook -f 30 -i inventory.ini run_script.yaml -e "KD=cff stra=last batch=10001" && \
cd ../RPis && \
python3 A.py --KD cff --stra last --batch 10001 && \
cd ../ansibles && \
ansible-playbook -f 30 -i inventory.ini fetch.yaml && \
cd && \


cd ~/NoteOnGithub/ && \
git restore . && git clean -df && \
cd Diplomarbeit/Codes/ansibles && \
ansible-playbook -f 30 -i inventory.ini init.yaml && \
ansible-playbook -f 30 -i inventory.ini run_script.yaml -e "KD=mhd925 stra=rand batch=10002" && \
cd ../RPis && \
python3 A.py --KD mhd925 --stra rand --batch 10002 && \
cd ../ansibles && \
ansible-playbook -f 30 -i inventory.ini fetch.yaml && \
cd && \


cd ~/NoteOnGithub/ && \
git restore . && git clean -df && \
cd Diplomarbeit/Codes/ansibles && \
ansible-playbook -f 30 -i inventory.ini init.yaml && \
ansible-playbook -f 30 -i inventory.ini run_script.yaml -e "KD=mhd925 stra=last batch=10002" && \
cd ../RPis && \
python3 A.py --KD mhd925 --stra last --batch 10002 && \
cd ../ansibles && \
ansible-playbook -f 30 -i inventory.ini fetch.yaml && \
cd && \


# cd ~/NoteOnGithub/ && \
# git restore . && git clean -df && \
# cd Diplomarbeit/Codes/ansibles && \
# ansible-playbook -f 30 -i inventory.ini init.yaml && \
# ansible-playbook -f 30 -i inventory.ini run_script.yaml -e "KD=mhd925 stra=all batch=10001" && \
# cd ../RPis && \
# python3 A.py --KD mhd925 --stra all --batch 10001 && \
# cd ../ansibles && \
# ansible-playbook -f 30 -i inventory.ini fetch.yaml && \
# cd && \
#
#
# cd ~/NoteOnGithub/ && \
# git restore . && git clean -df && \
# cd Diplomarbeit/Codes/ansibles && \
# ansible-playbook -f 30 -i inventory.ini init.yaml && \
# ansible-playbook -f 30 -i inventory.ini run_script.yaml -e "KD=cff stra=all batch=10001" && \
# cd ../RPis && \
# python3 A.py --KD cff --stra all --batch 10001 && \
# cd ../ansibles && \
# ansible-playbook -f 30 -i inventory.ini fetch.yaml && \
# cd && \


# 关机前的最后一步
cd ~/NoteOnGithub/Diplomarbeit/Codes/ansibles && \
ansible-playbook -f 30 -i inventory.ini poweroff.yaml ; \
# sleep 120 && \
# poweroff
