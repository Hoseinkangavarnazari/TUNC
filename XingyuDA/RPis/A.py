import argparse
import random
import numpy as np
import socket
import time
import json
import threading
import logging
import configparser
import pr_new

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
fh = logging.FileHandler(f'A_log_{KD}_{strategy}_{batch}.txt')
formatter = TimeFormatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# -----------------------------------------------------------------------------------------------------------------

A_listen_addr = ('', 10000)

routing_table = [
    {
        'destination': 'B',
        'next_socket': '192.168.2.2:10000',
        'hops': 1,
        'probability': 1.0
    },
    {
        'destination': 'C',
        'next_socket': '192.168.2.3:10000',
        'hops': 1,
        'probability': 0.5
    },
    {
        'destination': 'C',
        'next_socket': '192.168.2.2:10000',
        'hops': 2,
        'probability': 0.5
    },
    {
        'destination': 'D',
        'next_socket': '192.168.2.2:10000',
        'hops': 3,
        'probability': 0.5
    },
    {
        'destination': 'D',
        'next_socket': '192.168.2.3:10000',
        'hops': 2,
        'probability': 0.5
    },
]

# -----------------------------------------------------------------------------------------------------------------

stop_sending = False
current_run = 0
count_in_run = 0


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
    int_range = []
    start = 1
    for prob in probabilities:
        end = start + int(prob * 100) - 1
        int_range.append((start, end))
        start = end + 1

    r = random.randint(1, 100)
    for i, (start, end) in enumerate(int_range):
        if start <= r <= end:
            return i


def send_packet(pkt: np.ndarray, target: str, d21: int, d31: int):
    # AF_INET: IPv4, SOCK_DGRAM: UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    options = [route for route in routing_table if route['destination'] == target]
    probabilities = [route['probability'] for route in options]

    choice = select_route(probabilities)
    next_ip, next_port = options[choice]['next_socket'].split(':')

    next_addr = (next_ip, int(next_port))
    # dumps: serialize python dict to a JSON formatted string
    # tolist: convert numpy array to list for serialization
    msg_to_send_json = json.dumps({
        'destination': target,
        'packet': pkt.tolist(),
        'polluted': False,
        'safekeys_C': d21,
        'safekeys_D': d31
    })
    # encode: convert string to bytes, because socket can only send bytes
    sock.sendto(msg_to_send_json.encode(), next_addr)

    # print(f"Sent to {next_addr}: {msg_to_send_json}")
    # logger.info(f"Sent to {next_addr}: {msg_to_send_json}")

    sock.close()


def generate_diagonal_matrix():
    return np.eye(m, dtype=int)


def encode(original_matrix: np.ndarray, num_to_encode=m):
    rows_to_use_index = np.random.choice(original_matrix.shape[0], num_to_encode, replace=False)
    coefficients_enc = np.random.randint(0, p, num_to_encode) # [0, p)
    encoded_vector = np.zeros(m, dtype=int)

    for i, row in enumerate(rows_to_use_index):
        encoded_vector = vector_field_add(encoded_vector, vector_field_mul(original_matrix[row], coefficients_enc[i]))

    return encoded_vector


def listen_for_stop():
    global stop_sending, current_run, count_in_run
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(A_listen_addr)

    while True:
        data, addr = sock.recvfrom(1024)
        msg_received = data.decode()

        if 'Done for this run' in msg_received:
            print(f'---------- run={current_run}, count={count_in_run} ----------')
            logger.info(f'---------- run={current_run}, count={count_in_run} ----------')

            current_run += 1
            count_in_run = 0

        elif msg_received == 'Done for all':
            print('*********** All runs finished ***********')
            logger.info('*********** All runs finished ***********')
            stop_sending = True
            break

    sock.close()


def main():
    global stop_sending, count_in_run

    original_packet = generate_diagonal_matrix()

    listen_thread = threading.Thread(target=listen_for_stop)
    listen_thread.start()

    while not stop_sending:
        if count_in_run == 0:
            d_21_this_run, d_31_this_run = pr_new.calculate_safekeys(KD)

        count_in_run += 1
        encoded_packet = encode(original_packet)
        send_packet(encoded_packet, 'D', d_21_this_run, d_31_this_run)

        time.sleep(0.02)

    listen_thread.join()



if __name__ == "__main__":
    main()
