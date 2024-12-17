import os
import glob
import pandas as pd
import argparse

def combine_csv_files(directory, output_file="combined_edema_data.csv"):
    try:
        # Recursively find all CSV files in the directory
        csv_files = sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True))
        
        if not csv_files:
            print("No CSV files found in the specified directory.")
            return

        print(f"Found {len(csv_files)} CSV files. Combining them...")

        # Combine all CSV files into a single DataFrame
        combined_df = pd.concat(
            (pd.read_csv(f) for f in csv_files), 
            ignore_index=True
        )

        # Save the combined DataFrame to a CSV file
        combined_df.to_csv(output_file, index=False)
        print(f"Combined data saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple CSV files into one.")
    parser.add_argument("--directory", required=True, type=str, help="Path to the directory containing CSV files.")
    parser.add_argument("--output", default="combined_edema_data.csv", type=str, help="Name of the output CSV file.")
    args = parser.parse_args()
    
    combine_csv_files(args.directory, args.output)
