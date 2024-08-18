## go to folders where data is present PeerRead/data/acl_2017/dev/reviews and read al the files and store their names in an array. 
## now make a json file that has this array by format: "paper":"34"and so on...

import os
import json

def collect_paper_names(directory):
    # Create an empty dictionary to store paper names
    papers = {}

    # Traverse the directory and read all files
    for index, filename in enumerate(os.listdir(directory)):
        # Ensure we're only adding files (not directories)
        if os.path.isfile(os.path.join(directory, filename)):
            # Add the filename to the dictionary with the format "paper": "34"
            paper_key = f"paper_{index + 1}"
            papers[paper_key] = filename[:-5]
    
    return papers

def save_to_json(data, output_file):
    # Write the dictionary to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    # Define the directory where the data is present
    directory_path = '../data/iclr_2017/test'
    
    # Collect the paper names
    paper_names = collect_paper_names(directory_path+'/reviews')

    # Define the output JSON file path
    output_file_path = directory_path + '/papers.json'

    # Save the collected paper names to a JSON file
    save_to_json(paper_names, output_file_path)

    print(f"Paper names have been saved to {output_file_path}")
