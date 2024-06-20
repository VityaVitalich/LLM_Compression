import os
import json
import csv

def parse_results(directory_path):
    data = []

    # Iterate over all directories in the specified path
    for dir_name in os.listdir(directory_path):
        dir_path = os.path.join(directory_path, dir_name)
        
        if os.path.isdir(dir_path):
            # Extract the layer name and number from the directory name
            parts = dir_name.split('_')
            layer_name = parts[1]
            layer_number = parts[-1].split('-')[-1]
            
            # Path to the results.json file
            results_file = os.path.join(dir_path, 'results.json')
            
            if os.path.isfile(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                    # Extract the acc_norm value
                    acc_norm = results['results']['hellaswag']['acc_norm,none']
                    
                    # Append the extracted data to the list
                    data.append([layer_name, layer_number, acc_norm])
    
    # Define the output CSV file path
    output_file = os.path.join(directory_path, 'acc_norm_results.csv')

    # Write the data to a CSV file
    with open(output_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['layer_name', 'layer_number', 'acc_norm'])
        # Write the data rows
        csvwriter.writerows(data)
    

# Specify the path to the directory containing the subdirectories
directory_path = './'

# Call the function
parse_results(directory_path)

