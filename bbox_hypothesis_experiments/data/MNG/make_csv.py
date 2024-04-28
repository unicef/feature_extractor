import os
import csv

# Define the directory path where the files are located
directory_path = "/work/alex.unicef/raw_data/MNG/school/"
output_path = "/work/alex.unicef/feature_extractor/bbox_hypothesis_experiments/data/MNG/"

# Get a list of filenames in the directory
filenames = os.listdir(directory_path)

# Create a CSV file and write the data
output_file = os.path.join(output_path, "school_data.csv")
with open(output_file, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["index", "filename"])  # Write the header row
    
    # Write each file's index and filename to the CSV file
    for idx, filename in enumerate(filenames):
        csv_writer.writerow([idx, filename])

print(f"CSV file created successfully at: {output_file}. Number of files: {len(filenames)}")

print(f"CSV file created successfully at: {output_file}")

