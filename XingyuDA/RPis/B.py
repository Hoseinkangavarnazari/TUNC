import argparse
import json
import logging
import random
import socket
import threading
import time
import configparser


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
fh = logging.FileHandler(f'B_log_{KD}_{strategy}_{batch}.txt')
formatter = TimeFormatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# -----------------------------------------------------------------------------------------------------------------

B_listen_addr = ('', 10000)

routing_table = [
    {
        'destination': 'C',
        'next_socket': '192.168.2.3:10000',
        'hops': 1,
        'probability': 1.0
    },
    {
        'destination': 'D',
        'next_socket': '192.168.2.3:10000',
        'hops': 2,
        'probability': 1.0
    },
]

# -----------------------------------------------------------------------------------------------------------------

def select_route(probabilities):
    range = []
    start = 1
    for prob in probabilities:
        end = start + int(prob * 100) - 1
        range.append((start, end))
        start = end + 1

    r = random.randint(1, 100)  # [1, 100]
    for i, (start, end) in enumerate(range):
        if start <= r <= end:
            return i


def pollute_and_send_packet(msg: dict, from_addr: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    options = [route for route in routing_table if route['destination'] == msg['destination']]
    probabilities = [route['probability'] for route in options]
    choice = select_route(probabilities)
    next_ip, next_port = options[choice]['next_socket'].split(':')
    next_addr = (next_ip, int(next_port))

    msg_to_send_json = json.dumps({
        'destination': msg['destination'],
        'packet': msg['packet'],
        'polluted': True,
        'safekeys_C': msg['safekeys_C'],
        'safekeys_D': msg['safekeys_D']
    })

    sock.sendto(msg_to_send_json.encode(), next_addr)

    # print(f'forwarded from {from_addr} to {next_addr}: {msg_to_send_json}')
    # logger.info(f'forwarded from {from_addr} to {next_addr}: {msg_to_send_json}')

    sock.close()


def listen_for_stop_and_forward():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(B_listen_addr)

    while True:
        data, addr = sock.recvfrom(1024)
        msg_received = data.decode()

        if msg_received == 'Done for all':
            print('Received Done signal from D')
            logger.info('Received Done signal from D')
            break
        elif 'Done for this run' in msg_received:
            logger.info(f'{msg_received}')
        else:
            msg_received = json.loads(msg_received)
            pollute_and_send_packet(msg_received, addr)

    sock.close()


if __name__ == '__main__':
    t = threading.Thread(target=listen_for_stop_and_forward)
    t.start()
    t.join()
