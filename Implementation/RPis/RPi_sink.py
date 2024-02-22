import socket
from datetime import datetime

port = 12345
stored_file = 'received_data.txt'
log_file = 'log.txt'


def log_message(message):
    timestamp = datetime.now().strftime("%m-%d %H:%M:%S.%f")[:-3]
    with open(log_file, 'a') as log:
        log.write(f"[{timestamp}] {message}\n")


def receive_data(port, output_file):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', port))
        s.listen()
        log_message(f"Listening on port {port}...")

        while True:
            conn, addr = s.accept()
            with conn:
                log_message(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    log_message(f"Received data: {data.decode('utf-8')[:10]}")
                    with open(output_file, 'a') as f:
                        f.write(data.decode('utf-8') + '\n')


if __name__ == "__main__":
    receive_data(port, stored_file)


# sudo lsof -i:12345
# sudo kill -9 <PID>
