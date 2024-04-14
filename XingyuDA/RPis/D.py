import argparse
import random
import socket
import numpy as np
import json
import time
import configparser
import logging
import galois


config = configparser.ConfigParser()
config.read('config.ini')
m = config.getint('PARAMETERS', 'm')
p = config.getint('PARAMETERS', 'p')
n = config.getint('PARAMETERS', 'n')
total_runs = config.getint('PARAMETERS', 'total_runs')


parser = argparse.ArgumentParser()
parser.add_argument('--KD', type=str, choices=['cff', 'mhd925', 'mhd'], required=True)
parser.add_argument('--stra', type=str, choices=['rand', 'last', 'last2', 'all'], required=True)
parser.add_argument('--batch', type=int, required=True)
args = parser.parse_args()

KD = args.KD
strategy = args.stra
batch = args.batch


class TimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime("%H:%M:%S", ct)
            s = "%s.%03d" % (t, record.msecs)
        return s

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(f'D_log_{KD}_{strategy}_{batch}.txt')
formatter = TimeFormatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

D_listen_addr = ('', 10000)
socket_to_A = ('192.168.2.1', 10000)
socket_to_B = ('192.168.2.2', 10000)
socket_to_C = ('192.168.2.3', 10000)

# -----------------------------------------------------------------------------------------------------------------

test_json = '{"destination": "C", "packet": [92, 135, 139, 72, 56, 14, 127, 139, 169, 51, 90, 240, 128, 51, 139, 129, 102, 171, 148, 130], "timestamp": 1712240180.135978, "polluted": false}'


buffer_dec = []
current_run = 0
consecutive_fails = 0
GF = galois.GF(p)

if strategy == 'all':
    fail_threshold = 2
else:
    fail_threshold = 10


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(D_listen_addr)
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def decode() -> int:
    global buffer_dec

    A = GF(buffer_dec)
    rank = np.linalg.matrix_rank(A)

    return rank


def verify(msg: dict):
    d = msg['safekeys_D']
    # d = 1
    time.sleep(1.1)
    if random.randint(1, p ** d) != 1:  # [1, p^d]
        return False
    return True     # True: Attacker successfully fools the authenticator


def clear_socket_buffer():
    sock.setblocking(0)
    while True:
        try:
            data = sock.recv(1024)
        except BlockingIOError:
            break
    sock.setblocking(1)


while True:
    data, addr = sock.recvfrom(1024)
    msg_received = data.decode()
    msg_received = json.loads(msg_received)

    if msg_received['polluted']:

        consecutive_fails += 1

        if verify(msg_received) or consecutive_fails > fail_threshold:
            print(f'xxxxxxxxxx run={current_run} failed')
            logger.info(f'xxxxxxxxxx run={current_run} failed')

            sig = f'Done for this run {current_run}'

            send_sock.sendto(sig.encode(), socket_to_A)
            send_sock.sendto(sig.encode(), socket_to_B)
            send_sock.sendto(sig.encode(), socket_to_C)

            current_run += 1
            buffer_dec.clear()
            clear_socket_buffer()
            consecutive_fails = 0

        # else:
        #     # print(f'Discarded: {msg_received}')
        #     # logger.info(f'Discarded: {msg_received}')
    else:
        consecutive_fails = 0

        encoded_pkt = np.array(msg_received['packet'])   # covert msg['packet'] to numpy array
        coefficient = encoded_pkt[:m]
        buffer_dec.append(coefficient.tolist())

        rank = decode()

        # print(f'rank: {rank}')
        # logger.info(f'rank: {rank}')

        # print(f"Received from {addr}, rank: {rank}, msg: {msg_received}")
        # logger.info(f"Received from {addr}, rank: {rank}, msg: {msg_received}")

        if rank == m:
            msg_to_send = f"Done for this run {current_run}"

            print(f'---------- run={current_run}, buffer_size: {len(buffer_dec)} ----------')
            logger.info(f'---------- run={current_run}, buffer_size: {len(buffer_dec)} ----------')

            send_sock.sendto(msg_to_send.encode(), socket_to_A)
            send_sock.sendto(msg_to_send.encode(), socket_to_B)
            send_sock.sendto(msg_to_send.encode(), socket_to_C)

            current_run += 1
            buffer_dec.clear()
            clear_socket_buffer()

    if current_run >= total_runs:
        msg_to_send = 'Done for all'

        print('*********** All runs finished ***********')
        logger.info('*********** All runs finished ***********')

        send_sock.sendto(msg_to_send.encode(), socket_to_A)
        send_sock.sendto(msg_to_send.encode(), socket_to_B)
        send_sock.sendto(msg_to_send.encode(), socket_to_C)
        break

sock.close()
send_sock.close()

