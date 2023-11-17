import pandas as pd

# Read data from the text file
with open('Results/Packet100-1101MAC5-51ExNum:10000.txt', 'r') as file:
    lines = file.readlines()

# Extract relevant information from each line
data = {'PacketSize': [], 'MACSize': [], 'TimeResult': []}

for line in lines:
    parts = line.split('-')

    packet_size = None
    mac_size = None
    time_result = None

    for part in parts:
        if 'Result' in part:
            try:
                time_result = float(part.split(':')[-1])
            except ValueError:
                print(f"Error converting Result to float in line: {line}")
        elif 'PacketSize' in part:
            try:
                packet_size = int(part.split(':')[-1])
            except ValueError:
                print(f"Error converting PacketSize to int in line: {line}")
        elif 'MACSize' in part:
            try:
                mac_size = int(part.split(':')[-1])
            except ValueError:
                print(f"Error converting MACSize to int in line: {line}")

    if packet_size is not None and mac_size is not None and time_result is not None:
        data['PacketSize'].append(packet_size)
        data['MACSize'].append(mac_size)
        data['TimeResult'].append(time_result)

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('time_data_table.csv', index=False)


