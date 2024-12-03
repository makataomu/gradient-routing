import os
import re

import pandas as pd


def extract_model_type(filename):
    # Extract any model type that appears between 'o_' and the next underscore
    match = re.search(r"o_([^_]+)_", filename)
    if match:
        return match.group(1)
    return None


def process_csvs(directory):
    # Keep track of all unique model types found
    model_types_found = set()

    # First pass - collect all unique model types
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            model_type = extract_model_type(filename)
            if model_type:
                model_types_found.add(model_type)

    print(f"Found the following model types: {', '.join(sorted(model_types_found))}")

    # Second pass - process files
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Extract model type from filename
            model_type = extract_model_type(filename)

            if model_type:
                filepath = os.path.join(directory, filename)
                print(f"Processing {filename}...")

                try:
                    # Read CSV
                    df = pd.read_csv(filepath)

                    # Replace run_type column with correct model type
                    df["run_type"] = model_type

                    # Save back to same file
                    df.to_csv(filepath, index=False)
                    print(f"Updated run_type to {model_type} in {filename}")

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
            else:
                print(f"Could not extract model type from {filename}")


if __name__ == "__main__":
    directory = "data_recomputed_era"

    # Create backup directory
    backup_dir = "data_recomputed_era_backup"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"Created backup directory: {backup_dir}")

        # Create backups before processing
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                src = os.path.join(directory, filename)
                dst = os.path.join(backup_dir, filename)
                import shutil

                shutil.copy2(src, dst)
                print(f"Created backup of {filename}")

    # Process all CSVs
    process_csvs(directory)
    print("Processing complete!")
