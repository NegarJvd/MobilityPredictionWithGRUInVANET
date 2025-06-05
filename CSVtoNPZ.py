import os

import numpy as np
import pandas as pd

def process_single_csv_to_npz(csv_file_path, output_npz_path, seq_len=10):
    """
    Processes a single CSV file into sequences and saves them as .npz.

    Args:
        csv_file_path (str): Path to the CSV file.
        output_npz_path (str): Path to save the .npz file.
        seq_len (int): Length of input sequences.
    """
    all_sequences = []

    df = pd.read_csv(csv_file_path)
    grouped = df.groupby('node_id')

    for node_id, group in grouped:
        group = group.sort_values('time')
        if len(group) >= seq_len:
            coords = group[['x', 'y']].to_numpy()
            for start in range(len(coords) - seq_len + 1):
                seq = coords[start:start + seq_len]
                if not np.isnan(seq).any():
                    all_sequences.append(seq)

    if len(all_sequences) == 0:
        print(f"No valid sequences in {csv_file_path}")
        return

    data = np.array(all_sequences)
    np.savez_compressed(output_npz_path, data=data)
    print(f"Saved {data.shape[0]} sequences to '{output_npz_path}'")



if __name__ == '__main__':
    csv_file = './data/csv_files/mobility_30.csv'
    npz_file = './data/npz_files/mobility_30.npz'
    sequence_length = 10

    if not os.path.exists('./data/npz_files'):
        os.makedirs(os.path.dirname('./data/npz_files'), exist_ok=True)

    process_single_csv_to_npz(csv_file, npz_file, sequence_length)
