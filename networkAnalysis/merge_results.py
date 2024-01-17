import os
import re
import platform
import shutil

import pandas as pd


def merge_results(from_folder, to_folder):
    def merge_batches(attacker_type, distr_strategy, attack_model, num_compromised_nodes, r_keyset_size):
        dfs = []    # List to store dataframes from all batches

        if distr_strategy != 'cff':
            for file in os.listdir(from_folder):
                # Check for required elements in the file name.
                if re.match(r"csv\d+_", file) and file.endswith(".csv") and all(x in file for x in [attacker_type, distr_strategy, attack_model, f'_{num_compromised_nodes}c', f'_{r_keyset_size}keys']):
                    df = pd.read_csv(os.path.join(from_folder, file))
                    dfs.append(df)
        else:
            for file in os.listdir(from_folder):
                # Check for required elements in the file name.
                if re.match(r"csv\d+_", file) and file.endswith(".csv") and all(x in file for x in [attacker_type, distr_strategy, attack_model, f'_{num_compromised_nodes}c']):
                    df = pd.read_csv(os.path.join(from_folder, file))
                    dfs.append(df)

        if not dfs:
            return False

        # Combine all dataframes.
        combined_df = pd.concat(dfs, ignore_index=True)

        # Determine the total number of runs from the combined dataframe.
        count_runs = combined_df.shape[0]

        # Generate the combined CSV file name.
        if distr_strategy != 'cff':
            merged_csv_name = f'csv_{attacker_type}_{distr_strategy}_{attack_model}_{count_runs}runs_{num_compromised_nodes}c_{r_keyset_size}keys.csv'
        else:
            merged_csv_name = f'csv_{attacker_type}_{distr_strategy}_{attack_model}_{count_runs}runs_{num_compromised_nodes}c.csv'

        combined_df.to_csv(os.path.join(to_folder, merged_csv_name), index=False)
        print(f'Merged {len(dfs)} CSV files into {merged_csv_name}')
        return True

    count_merged = 0
    for attacker_type in ['tpa', 'dpa']:
        for distr_strategy in ['rd', 'mhd', 'cff']:
            for attack_model in ['our', 'other']:
                for num_compromised_nodes in range(1, 30):      # 30 should > the max number of compromised nodes
                    if distr_strategy != 'cff':
                        for r_keyset_size in range(1, 30):  # 30 should > the max number of r_keyset_size
                            if merge_batches(attacker_type, distr_strategy, attack_model, num_compromised_nodes, r_keyset_size):
                                count_merged += 1
                    else:
                        if merge_batches(attacker_type, distr_strategy, attack_model, num_compromised_nodes, None):
                            count_merged += 1

    print(f'Total {count_merged} merged CSV files.')


if __name__ == '__main__':

    if platform.system() == 'Darwin':
        m_result_folder = '/Users/xingyuzhou/Downloads/cff1w100x'
    else:
        m_result_folder = '/home/xingyu/Downloads/cff10w'

    result_merged_folder = m_result_folder + '/merged'
    if os.path.exists(result_merged_folder):
        reponse = input(f'{result_merged_folder} already exists. Do you want to overwrite it? [y/n]').strip().lower()
        if reponse == 'y':
            shutil.rmtree(result_merged_folder)
        else:
            exit(0)
    os.makedirs(result_merged_folder)

    merge_results(from_folder=m_result_folder, to_folder=result_merged_folder)
