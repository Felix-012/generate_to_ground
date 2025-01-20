import re
import csv


# Function to process each line and extract relevant information
def process_line(line):
    # Regex pattern to extract trial number, value, and parameters
    pattern = r"Trial (\d+) finished with value: ([\d\.]+) and parameters: ({.*})"
    match = re.search(pattern, line)

    if match:
        trial_number = match.group(1)
        value = float(match.group(2))
        parameters = eval(match.group(3))  # safely evaluate the dictionary string

        # Create a list of values to write in CSV
        row = [trial_number, f"{value:.3f}"]  # limit value to 3 decimal places
        row.extend(parameters.values())  # add the parameter values

        return row
    return None


# Read the file and extract relevant information
input_file = '/vol/ideadata/ce90tate/nohup.out'  # Replace with your file name
output_file = '/vol/ideadata/ce90tate/data/trials_summary.csv'

with open(input_file, 'r') as file, open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header row
    header = ['Trial Number', 'Value'] + [f'layer_{i}' for i in range(16)]
    csv_writer.writerow(header)

    for line in file:
        row = process_line(line)
        if row:
            csv_writer.writerow(row)

print(f"Data extracted and saved to {output_file}")
