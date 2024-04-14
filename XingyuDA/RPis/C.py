import argparse
import json
import logging
import random
import socket
import threading
import time
import numpy as np
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
fh = logging.FileHandler(f'C_log_{KD}_{strategy}_{batch}.txt')
formatter = TimeFormatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# -----------------------------------------------------------------------------------------------------------------

C_listen_addr = ('', 10000)

routing_table = [
    {
        'destination': 'D',
        'next_socket': '192.168.2.4:10000',
        'hops': 1,
        'probability': 1.0
    },
]

# -----------------------------------------------------------------------------------------------------------------

num_to_recode = int(np.ceil(m * 0.1))
buffer_rec = []


field_add_table = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0]
]

field_mul_table = [
    [0, 0, 0, 0],
    [0, 1, 2, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2]
]

def field_add(x, y):
    return field_add_table[x][y]

def field_mul(x, y):
    return field_mul_table[x][y]

def vector_field_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([field_add(x[i], y[i]) for i in range(len(x))], dtype=int)

def vector_field_mul(vec: np.ndarray, ele: int) -> np.ndarray:
    return np.array([field_mul(vec[i], ele) for i in range(len(vec))], dtype=int)


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


def recode() -> (np.ndarray, bool):
    global buffer_rec
    polluted = False

    if strategy == 'rand':
        if len(buffer_rec) < num_to_recode:
            num_to_recode_actual = len(buffer_rec)
        else:
            num_to_recode_actual = num_to_recode
        rows_to_use = random.sample(buffer_rec, num_to_recode_actual)
    elif strategy == 'last':
        rows_to_use = buffer_rec[-min(len(buffer_rec), num_to_recode):]     # last num_to_recode rows or all rows if less than num_to_recode
    elif strategy == 'last2':
        rows_to_use = buffer_rec[-min(len(buffer_rec), num_to_recode * 2):]     # last 2 * num_to_recode rows or all rows if less than 2 * num_to_recode
    elif strategy == 'all':
        rows_to_use = buffer_rec
    else:
        raise ValueError(f'Invalid strategy: {strategy}')

    recoded_packet = np.zeros(m, dtype=int)

    for msg in rows_to_use:

        if msg['polluted']:
            polluted = True     # if any of the rows is polluted, the recoded packet will be polluted

        coefficients_of_row = np.array(msg['packet'][:m], dtype=int)
        coefficient_rec = np.random.randint(0, p)   # [0, p)
        recoded_packet = vector_field_add(recoded_packet, vector_field_mul(coefficients_of_row, coefficient_rec))

    return recoded_packet, polluted


def verify(msg: dict):
    d = msg['safekeys_C']
    # d = 1
    if random.randint(1, p ** d) != 1:  # [1, p^d]
        return False    # failed verification
    return True


def clear_socket_buffer(sock):
    sock.setblocking(0)
    while True:
        try:
            data = sock.recv(1024)
        except BlockingIOError:
            break
    sock.setblocking(1)


def process_and_send_packet(msg: dict, from_addr):
    global buffer_rec
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if msg['polluted']:
        if not verify(msg):     # did not pass verification
            # print(f'Discarded from {from_addr}: {msg}')
            # logger.info(f'Discarded from {from_addr}: {msg}')

            sock.close()
            return

    options = [route for route in routing_table if route['destination'] == msg['destination']]
    probabilities = [route['probability'] for route in options]
    choice = select_route(probabilities)
    next_ip, next_port = options[choice]['next_socket'].split(':')
    next_addr = (next_ip, int(next_port))

    buffer_rec.append(msg)
    recoded_packet, polluted = recode()

    msg_to_send_json = json.dumps({
        'destination': msg['destination'],
        'packet': recoded_packet.tolist(),
        'polluted': polluted,
        'safekeys_C': msg['safekeys_C'],
        'safekeys_D': msg['safekeys_D']
    })

    sock.sendto(msg_to_send_json.encode(), next_addr)

    # print(f'forwarded from {from_addr} to {next_addr}: {msg_to_send_json}')
    # logger.info(f'forwarded from {from_addr} to {next_addr}: {msg_to_send_json}')

    sock.close()


def listen_for_stop_and_forward():
    global buffer_rec
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(C_listen_addr)

    while True:
        data, addr = sock.recvfrom(1024)
        msg_received = data.decode()

        if msg_received == 'Done for all':
            print('Received Done signal from D')
            logger.info('Received Done signal from D')

            break
        elif 'Done for this run' in msg_received:
            logger.info(f'{msg_received}')
            buffer_rec.clear()
            clear_socket_buffer(sock)
        else:
            msg_received = json.loads(msg_received)
            process_and_send_packet(msg_received, addr)

    sock.close()


if __name__ == '__main__':
    t = threading.Thread(target=listen_for_stop_and_forward)
    t.start()
    t.join()
