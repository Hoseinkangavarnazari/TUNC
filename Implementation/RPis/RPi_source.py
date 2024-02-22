import socket
import argparse
import random
import string
import time
from datetime import datetime

targets = [("192.168.1.103", 12345), ("192.168.1.105", 12345)]
test_file = "data.txt"
log_file = "sender_log.txt"


def log_message(message):
    timestamp = datetime.now().strftime("%m-%d %H:%M:%S.%f")[:-3]
    with open(log_file, 'a') as log:
        log.write(f"[{timestamp}] {message}\n")


def generate_data_file(num_lines, chars_per_line, filename):
    char_set = string.ascii_letters + string.digits
    with open(filename, 'w') as file:
        for _ in range(num_lines):
            random_line = ''.join(random.choice(char_set) for _ in range(chars_per_line))
            file.write(random_line + '\n')
    log_message("Data file generated successfully.")


def send_data(filename):
    for ip, port in targets:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((ip, port))
                with open(filename, 'r') as file:
                    for line in file:
                        s.sendall(line.encode('utf-8'))
                        log_message(f"Data {line[:10]} sent successfully to {ip}:{port}.")
                        time.sleep(0.1)

            except ConnectionError as e:
                log_message(f"Connection error with {ip}:{port}: {e}")
            except FileNotFoundError:
                log_message(f"File not found: {filename}")
            except Exception as e:
                log_message(f"Error with {ip}:{port}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Send file data to Raspberry Pis.")
    parser.add_argument('--generate', action='store_true', help="Generate a new data file before sending.")
    args = parser.parse_args()

    if args.generate:
        generate_data_file(num_lines=5, chars_per_line=1000, filename=test_file)
    else:
        send_data(test_file)


if __name__ == "__main__":
    main()
