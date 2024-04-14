import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_a_log(path) -> pd.DataFrame:
    data = []
    with open(path, 'r') as f:
        for line in f:
            match = re.search(r'---------- run=(\d+), count=(\d+) ----------', line)
            if match:
                data.append({'run': int(match.group(1)), 'count': int(match.group(2))})
    return pd.DataFrame(data)


def process_d_log(path) -> pd.DataFrame:
    data = []
    with open(path, 'r') as f:
        for line in f:
            # Check if the line indicates a failed run
            match_failed = re.search(r'xxxxxxxxxx run=(\d+) failed', line)
            # Extended regex to capture both run and buffer_size
            match_run = re.search(r'---------- run=(\d+), buffer_size: (\d+) ----------', line)

            if match_failed:
                data.append({'run': int(match_failed.group(1)), 'status': 'failed', 'buffer_size': None})
            elif match_run:
                data.append({'run': int(match_run.group(1)), 'status': '', 'buffer_size': int(match_run.group(2))})

    return pd.DataFrame(data)


def AD(path_a, path_d) -> pd.DataFrame:
    df_a = process_a_log(path_a)
    df_d = process_d_log(path_d)
    df_merged = pd.merge(df_a, df_d, on='run', how='outer')
    df_merged['count'] = df_merged['count'].fillna(df_merged['count'].mean()).astype(int)
    return df_merged


def load_and_merge_logs(kd, stra, batch_list):
    base_path = '/Users/xingyuzhou/NoteOnGithub/Diplomarbeit/Codes/HWresults/'
    df_list = []

    for batch in batch_list:
        path_a = f'{base_path}A_log_{kd}_{stra}_{batch}.txt'
        path_d = f'{base_path}D_log_{kd}_{stra}_{batch}.txt'

        if os.path.exists(path_a) and os.path.exists(path_d):
            df_list.append(AD(path_a, path_d))
        else:
            print(f'{kd}_{stra}_{batch} not found!')

    return pd.concat(df_list)

# ----------------------------

# 10, p=2, interval=0, re=5

# ----------------------------

# 500, p=4, interval=1, re=5 (mhd ok, cff not)

# ----------------------------

# 1000, p=4, interval=0, re=5 (mhd ok, cff not)

# ----------------------------

# 2000, p=4, interval=0, re=25 (mhd ok, cff not)

# ----------------------------

# 3000, p=4, interval=1.1, re=25 (mhd ok, cff not)

# ----------------------------

# 4000, p=4, interval=(4000:1.1, >4000:1.6), re=25 (mhd ok, cff ok)

# ----------------------------

# 10000, p=4, interval=1.1, re=5 (mhd ok, cff ok)

# ----------------------------

df_cff_rand_tc = load_and_merge_logs('cff', 'rand', [10000, 10001])
df_cff_last_tc = load_and_merge_logs('cff', 'last', [10000, 10001])
df_cff_all_tc = load_and_merge_logs('cff', 'all', [4000])

df_mhd925_rand_tc = load_and_merge_logs('mhd925', 'rand', [10000, 10001, 10002])
df_mhd925_last_tc = load_and_merge_logs('mhd925', 'last', [10000, 10001, 10002])
df_mhd925_all_tc = load_and_merge_logs('mhd925', 'all', [4000])

# ----------------------------

df_cff_rand_fr_5 = load_and_merge_logs('cff', 'rand', [10000, 10001])
df_cff_last_fr_5 = load_and_merge_logs('cff', 'last', [10000, 10001])
df_cff_all_fr_5 = load_and_merge_logs('cff', 'all', [4000, 4001])

df_mhd925_rand_fr_5 = load_and_merge_logs('mhd925', 'rand', [1000, 10000, 10001, 10002, 1001, 1002])
df_mhd925_last_fr_5 = load_and_merge_logs('mhd925', 'last', [1000, 10000, 10001, 10002, 1001, 1002])
df_mhd925_all_fr_5 = load_and_merge_logs('mhd925', 'all', [4000, 4001, 4002, 1000])

# ----------------------------

df_cff_rand_fr_25 = load_and_merge_logs('cff', 'rand', [4000, 4001])
df_cff_last_fr_25 = load_and_merge_logs('cff', 'last', [4000, 4001])
df_cff_all_fr_25 = load_and_merge_logs('cff', 'all', [4000, 4001])

df_mhd925_rand_fr_25 = load_and_merge_logs('mhd925', 'rand', [4000, 4001, 2000, 2001, 2002, 2003, 2004, 2005, 3000])
df_mhd925_last_fr_25 = load_and_merge_logs('mhd925', 'last', [4000, 4001, 2000, 2001, 2002, 2003, 2004, 2005, 3000])
df_mhd925_all_fr_25 = load_and_merge_logs('mhd925', 'all', [4000, 4001, 4002, 1000])

# ----------------------------


def plot_failed_ratio(df_list, config_list):

    failed_ratio_list = []
    colors = ['#F08C55', '#6EC8C8', '#C66934', '#429F9F', '#9D4712', '#057778']

    for df in df_list:
        # 输出各 df 行数，以及 status 为 failed 的行数
        print(f'{df.shape[0]}, failed: {df[df["status"] == "failed"].shape[0]}')
        num_total_runs = df.shape[0]
        num_failed_runs = df[df['status'] == 'failed'].shape[0]

        failed_ratio = num_failed_runs / num_total_runs
        failed_ratio_list.append(failed_ratio)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(config_list, failed_ratio_list, color=colors, width=0.3)

    for bar, text in zip(bars, failed_ratio_list):
        height = bar.get_height()
        display_text = f'{text:.3%}' if text > 0 else f'< 1/{num_total_runs}'
        plt.text(bar.get_x() + bar.get_width() / 2, height, display_text, ha='center', va='bottom', fontsize=12, color='black')

    plt.ylabel('Failed Transmisson Ratio', fontsize=14)
    plt.yscale('log')
    plt.xlabel('Key Distribution Mechanism and Recoding Strategy', fontsize=14)
    plt.xticks(range(len(config_list)), config_list, fontsize=10)
    plt.grid(which='both', linestyle='-.', linewidth=0.5)
    plt.show()

def plot_count_scatter(df_list, config_list):

    fig = plt.figure(figsize=(10, 6))
    offset_range = np.linspace(-0.08, 0.08, 3)
    colors = ['#F08C55', '#6EC8C8', '#C66934', '#429F9F', '#9D4712', '#057778']

    for i, df in enumerate(df_list):
        success_runs = df[df['status'] == '']
        # success_runs = success_runs[success_runs['count'] < 2000]
        # success_runs = success_runs.head(500)
        print(f'{df.shape[0]}')

        x_value_base = i
        offsets = np.random.choice(offset_range, success_runs.shape[0])
        x_values = [x_value_base + offset for offset in offsets]

        y_values = success_runs['count'].tolist()

        mean_count = success_runs['count'].mean()

        color = colors[i % len(colors)]
        plt.scatter(x_values, y_values, alpha=0.3, s=80, c=color)
        # mean_label = "Transmisson Count Mean" if i == 0 else None
        mean_label = f'{mean_count:.2f}'
        plt.hlines(mean_count, i - 0.2, i + 0.2, 'r', label=mean_label)

    plt.xticks(range(len(config_list)), config_list, fontsize=10)
    plt.xlabel('Key Distribution Mechanism and Recoding Strategy', fontsize=14)
    plt.ylabel('Number of Packets Sent', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(which='both', linestyle='-.', linewidth=0.5)
    plt.show()

def plot_buffer_size_scatter(df_list, config_list):

    fig = plt.figure(figsize=(10, 6))
    offset_range = np.linspace(-0.03, 0.03, 3)
    colors = ['#F08C55', '#6EC8C8', '#C66934', '#429F9F', '#9D4712', '#057778']

    for i, df in enumerate(df_list):
        success_runs = df[df['status'] == '']
        # success_runs = success_runs[success_runs['count'] < 2000]
        # success_runs = success_runs.head(500)
        print(f'{df.shape[0]}')

        x_value_base = i
        offsets = np.random.choice(offset_range, success_runs.shape[0])
        x_values = [x_value_base + offset for offset in offsets]

        y_values = success_runs['buffer_size'].tolist()

        mean_count = success_runs['buffer_size'].mean()

        color = colors[i % len(colors)]
        plt.scatter(x_values, y_values, alpha=0.3, s=80, c=color)

        mean_label = f'{mean_count:.2f}'
        plt.hlines(mean_count, i - 0.2, i + 0.2, 'r', label=mean_label)

    plt.xticks(range(len(config_list)), config_list, fontsize=10)
    plt.xlabel('Key Distribution Mechanism and Recoding Strategy', fontsize=14)
    plt.ylabel('Number of Packets Received by the Node D', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(which='both', linestyle='-.', linewidth=0.5)
    plt.show()


def plot_buffer_size_box(df_list, config_list):

    fig = plt.figure(figsize=(10, 6))

    data = []
    for i, df in enumerate(df_list):
        success_runs = df[df['status'] == '']
        print(f'{df.shape[0]}')

        y_values = success_runs['buffer_size'].tolist()
        data.append(y_values)

    plt.boxplot(data, labels=config_list, showmeans=True)
    plt.xticks(range(1, len(config_list) + 1), config_list, fontsize=10)
    plt.xlabel('Key Distribution Mechanism and Recoding Strategy', fontsize=14)
    plt.ylabel('Number of Packets Received', fontsize=14)
    plt.grid(which='both', linestyle='-.', linewidth=0.5)
    plt.show()


def plot_count_mean_bar(df_list, config_list):

        fig = plt.figure(figsize=(10, 6))
        colors = ['#F08C55', '#6EC8C8', '#C66934', '#429F9F', '#9D4712', '#057778']

        mean_list = []
        for i, df in enumerate(df_list):
            success_runs = df[df['status'] == '']

            mean_count = success_runs['count'].mean()
            mean_list.append(mean_count)
            print(f'{df.shape[0]}, mean: {mean_count}')

        bars = plt.bar(config_list, mean_list, color=colors, width=0.3)

        for bar, text in zip(bars, mean_list):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{text:.3f}', ha='center', va='bottom', fontsize=12, color='black')

        plt.ylabel('Average Number of Packets Sent', fontsize=14)
        plt.xlabel('Key Distribution Mechanism and Recoding Strategy', fontsize=14)
        plt.xticks(range(len(config_list)), config_list, fontsize=10)
        plt.grid(which='both', linestyle='-.', linewidth=0.5)
        plt.show()


def inSpan(data, percentage):

    confidence = 1 - percentage

    # default assumption
    CIdown, CIup, mean = CI(data, confidence)

    result = False
    xPrecentLess = mean - (percentage * mean)
    xPercentMore = mean + (percentage * mean)

    if (xPrecentLess <= CIdown and CIup <= xPercentMore):
        result = True

    return result


def CI(data, confidence):

    mean = np.mean(data)
    std = np.std(data)
    n = len(data)

    if confidence == 0.9:
        z = 1.645
    elif confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    else:
        print("Invalid confidence level")
        return

    tempdown = mean - z*std/np.sqrt(n)
    tempup = mean + z*std/np.sqrt(n)
    tempmean = mean

    avgDist = (tempup-tempdown)/2

    print("CI: ", tempdown, tempup, tempmean, "distance half:", avgDist)

    return [tempdown, tempup, tempmean]


# ----------------------------

data_mhd925_rand_tc = df_mhd925_rand_tc[df_mhd925_rand_tc['status'] == '']['count'].tolist()
data_mhd925_last_tc = df_mhd925_last_tc[df_mhd925_last_tc['status'] == '']['count'].tolist()
data_mhd925_all_tc = df_mhd925_all_tc[df_mhd925_all_tc['status'] == '']['count'].tolist()

data_cff_rand_tc = df_cff_rand_tc[df_cff_rand_tc['status'] == '']['count'].tolist()
data_cff_last_tc = df_cff_last_tc[df_cff_last_tc['status'] == '']['count'].tolist()
data_cff_all_tc = df_cff_all_tc[df_cff_all_tc['status'] == '']['count'].tolist()

percent = 0.05

print(f'mhd925_rand_tc: {inSpan(data_mhd925_rand_tc, percent)}\n')
print(f'mhd925_last_tc: {inSpan(data_mhd925_last_tc, percent)}\n')
print(f'mhd925_all_tc: {inSpan(data_mhd925_all_tc, percent)}\n')

print(f'cff_rand_tc: {inSpan(data_cff_rand_tc, percent)}\n')
print(f'cff_last_tc: {inSpan(data_cff_last_tc, percent)}\n')
print(f'cff_all_tc: {inSpan(data_cff_all_tc, percent)}\n')

# print(df_mhd925_rand_tc.head(10))
# print(df_mhd925_last_tc.head(10))
# print(df_mhd925_all_tc.head(10))
#
# print(df_cff_rand_tc.head(10))
# print(df_cff_last_tc.head(10))
# print(df_cff_all_tc.head(10))

# ----------------------------

Xticks = ['CFF-based KD\nRandom ' r'$(10\%m)$', 'Max-Hamming PKD\nRandom ' r'$(10\%m)$',
          'CFF-based KD\nLast ' r'$(10\%m)$', 'Max-Hamming PKD\nLast ' r'$(10\%m)$',
          'CFF-based KD\nAll', 'Max-Hamming PKD\nAll']

plot_count_mean_bar([df_cff_rand_tc, df_mhd925_rand_tc, df_cff_last_tc, df_mhd925_last_tc, df_cff_all_tc, df_mhd925_all_tc], Xticks)
plot_failed_ratio([df_cff_rand_fr_5, df_mhd925_rand_fr_5, df_cff_last_fr_5, df_mhd925_last_fr_5, df_cff_all_fr_5, df_mhd925_all_fr_5], Xticks)
plot_buffer_size_box([df_cff_rand_tc, df_mhd925_rand_tc, df_cff_last_tc, df_mhd925_last_tc, df_cff_all_tc, df_mhd925_all_tc], Xticks)


plot_failed_ratio([df_cff_rand_fr_25, df_mhd925_rand_fr_25, df_cff_last_fr_25, df_mhd925_last_fr_25, df_cff_all_fr_25, df_mhd925_all_fr_25], Xticks)
