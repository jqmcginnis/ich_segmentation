import os
import glob
import pandas as pd
import argparse

def combine_csv_files(directory, output_file="combined_volumes.csv"):
    try:
        # Recursively find all CSV files in the directory
        csv_files = sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True))
        
        if not csv_files:
            print("No CSV files found in the specified directory.")
            return

        print(f"Found {len(csv_files)} CSV files. Combining them...")

        combined_data = []

        for file in csv_files:
            # Read each CSV file
            df = pd.read_csv(file)

            # Transpose rows for class-based stats and add filename as an identifier
            transposed_df = df.set_index("class").T
            transposed_df["filename"] = os.path.basename(file)
            
            # Append to the combined data list
            combined_data.append(transposed_df)

        # Combine all transposed dataframes into a single dataframe
        combined_df = pd.concat(combined_data, ignore_index=True)

        # Reorder columns with 'filename' first
        columns = ["filename"] + [col for col in combined_df.columns if col != "filename"]
        combined_df = combined_df[columns]

        # Save the combined DataFrame to a CSV file
        combined_df.to_csv(output_file, index=False)
        print(f"Combined data saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple CSV files with transposed rows into one.")
    parser.add_argument("--directory", required=True, type=str, help="Path to the directory containing CSV files.")
    parser.add_argument("--output", default="combined_volumes.csv", type=str, help="Name of the output CSV file.")
    args = parser.parse_args()
    
    combine_csv_files(args.directory, args.output)
