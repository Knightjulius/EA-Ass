

import csv
import re

# Read the DAT file and split each line by various whitespace delimiters
with open('data/run-3/data_f18_LABS/IOHprofiler_f18_DIM50.dat', 'r') as dat_file:
    data = [re.split(r'\s+', line.strip()) for line in dat_file]

# Write the data to a CSV file with multiple columns
with open('output.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(data)