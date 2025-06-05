import os
import re
import csv

def parse_tcl_to_csv (file_name, tcl_folder, csv_folder):
    pattern = re.compile(r'\$ns_ at ([\d.]+) "\$node_\((\d+)\) setdest ([\d.]+) ([\d.]+) [\d.]+"')

    input_file = os.path.join(tcl_folder, file_name)
    output_file = os.path.join(csv_folder, file_name.replace('.tcl', '.csv'))

    mobility_data = []

    with open(input_file, 'r') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                time = float(match.group(1))
                node_id = int(match.group(2))
                x = float(match.group(3))
                y = float(match.group(4))
                mobility_data.append([node_id, time, x, y])

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['node_id', 'time', 'x', 'y'])  # header
        writer.writerows(mobility_data)


if __name__ == '__main__':
    input_folder = './data/tcl_files'
    output_folder = './data/csv_output'

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tcl'):
            parse_tcl_to_csv(filename, input_folder, output_folder)
            print(f"Processed {filename}")
