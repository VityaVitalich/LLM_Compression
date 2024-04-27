import os
import json

# Function to process each results.json file
def process_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return {
            dataset: info.get("acc_norm,none", info.get("acc,none"))
            for dataset, info in data["results"].items()
        }

# The directory to search in (current directory in this case)
base_dir = './Llama7b_tulu_quik_2bit_lora_lowlr/'

# Patterns to match directories
patterns = ['_']

# Output file
output_file = 'summary.txt'

# Initialize a dictionary to hold all the results
all_results = {}

# Iterate over each item in the base directory
for item in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, item)) and any(item.startswith(pattern) for pattern in patterns):
        json_file = os.path.join(base_dir, item, 'results.json')
        print(json_file)
        if os.path.isfile(json_file):
            all_results[item] = process_json_file(json_file)

# Writing results to the output file
with open(output_file, 'w') as file:
    for directory, results in all_results.items():
        for dataset, acc in results.items():
           file.write(str(directory) + '-' + str(dataset) + ': ' + str(acc) + '\n')

print("Summary written")

