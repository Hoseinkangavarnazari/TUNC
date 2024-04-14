import numpy as np
import configparser


config = configparser.ConfigParser()
config.read('config.ini')
m = config.getint('PARAMETERS', 'm')
p = config.getint('PARAMETERS', 'p')
n = config.getint('PARAMETERS', 'n')
total_runs = config.getint('PARAMETERS', 'total_runs')

buffer_dec = np.empty((0, m), dtype=int)
buffer_rec = np.empty((0, m), dtype=int)


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


def generate_diagonal_matrix():
    return np.eye(m, dtype=int)


def encode(original_matrix):
    # num_rows_to_use = np.random.randint(1, m+1)
    num_rows_to_use = 2
    # random choose num_rows_to_use rows from original_matrix, one row can only be chosen once
    rows_to_use_index = np.random.choice(original_matrix.shape[0], num_rows_to_use, replace=False)
    coefficients = np.random.randint(0, p, num_rows_to_use)
    print(f'rows_to_use_index: {rows_to_use_index}')
    print(f'coefficients: {coefficients}')
    encoded_vector = np.zeros(m, dtype=int)

    for i, row in enumerate(rows_to_use_index):
        print(f'i: {i}, row: {row}, coefficients[i]: {coefficients[i]}, original_matrix[row]: {original_matrix[row]}')
        # print(f'type(coefficients[i]): {type(coefficients[i])}, type(original_matrix[row]): {type(original_matrix[row])}')
        encoded_vector = vector_field_add(encoded_vector, vector_field_mul(original_matrix[row], coefficients[i]))


    print(f'encoded_vector: {encoded_vector}')
    return encoded_vector


def decode(encoded_packet):
    global buffer_dec
    buffer_dec = np.vstack([buffer_dec, encoded_packet])
    rank = np.linalg.matrix_rank(buffer_dec)
    return rank


def recode(num_to_recode=10):
    global buffer_rec
    rows_to_use_index = np.random.choice(buffer_rec.shape[0], num_to_recode, replace=False)
    coefficients = np.random.randint(1, p+1, num_to_recode)
    recoded_packet = np.zeros(m, dtype=int)

    for i, row in enumerate(rows_to_use_index):
        recoded_packet = field_add(recoded_packet, field_mul(coefficients[i], buffer_rec[row], p), p)

    return recoded_packet

original_matrix = generate_diagonal_matrix()
print(f'Original Matrix:\n{original_matrix}\n')


num_received_packets = 0
while True:
    encoded_packet = encode(original_matrix)
    rank = decode(encoded_packet)
    print(f'rank: {rank}')
    num_received_packets += 1
    if rank == m:
        break
