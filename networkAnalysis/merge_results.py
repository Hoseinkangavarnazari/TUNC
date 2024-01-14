import os
import re

import pandas as pd


def merge_results(num_c_node_options, num_keyset_options, folder):
    def merge_batches(attacker_type, distr_strategy, attack_model, num_compromised_nodes, r_keyset_size):
        dfs = []    # List to store dataframes from all batches

        for file in os.listdir(folder):
            # Check for required elements in the file name.
            if re.match(r"csv\d+_", file) and file.endswith(".csv") and all(x in file for x in [attacker_type, distr_strategy, attack_model, f'_{num_compromised_nodes}c', f'_{r_keyset_size}keys']):
                df = pd.read_csv(os.path.join(folder, file))
                dfs.append(df)

        if not dfs:
            return

        # Combine all dataframes.
        combined_df = pd.concat(dfs, ignore_index=True)

        # Determine the total number of runs from the combined dataframe.
        count_runs = combined_df.shape[0]

        # Generate the combined CSV file name.
        merged_csv_name = f'csv_{attacker_type}_{distr_strategy}_{attack_model}_{count_runs}runs_{num_compromised_nodes}c_{r_keyset_size}keys.csv'

        result_merged_folder = folder + '/merged'
        os.makedirs(result_merged_folder, exist_ok=True)

        combined_df.to_csv(os.path.join(result_merged_folder, merged_csv_name), index=False)
        print(f'Merged {len(dfs)} CSV files into {merged_csv_name}')

    for attacker_type in ['tpa', 'dpa']:
        for distr_strategy in ['rd', 'mhd_n', 'mhd_a']:
            for attack_model in ['our', 'other']:
                for num_compromised_nodes in range(1, num_c_node_options + 1):
                    for r_keyset_size in range(1, num_keyset_options + 1):
                        merge_batches(attacker_type, distr_strategy, attack_model, num_compromised_nodes, r_keyset_size)


if __name__ == '__main__':
    m_num_c_node_options = 10
    m_num_keyset_options = 11
    m_result_folder = '/Users/xingyuzhou/Downloads/dpaComplete50x'

    merge_results(m_num_c_node_options, m_num_keyset_options, m_result_folder)
