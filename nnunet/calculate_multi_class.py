import nibabel as nib
import numpy as np
import os
import csv
import argparse

# Define the class labels
CLASS_LABELS = {
    "SV": 1,
    "V3": 2,
    "V4": 3
}

def calculate_volumes_per_class(mask_path):
    try:
        # Load the NIfTI file
        nifti_img = nib.load(mask_path)
        data = nifti_img.get_fdata()
        header = nifti_img.header

        # Calculate voxel volume in mm³
        voxel_size = np.prod(header.get_zooms())

        # Prepare results dictionary
        results = {}
        total_voxels = 0
        total_volume = 0.0

        for label_name, label_value in CLASS_LABELS.items():
            # Count the number of voxels for the current class
            class_voxels = np.sum(data == label_value)
            class_volume = class_voxels * voxel_size  # in mm³

            # Store results
            results[label_name] = {
                "voxels": class_voxels,
                "volume_mm3": class_volume
            }

            # Update total for all classes except background
            if label_name != "background":
                total_voxels += class_voxels
                total_volume += class_volume

        # Add total results
        results["total"] = {
            "voxels": total_voxels,
            "volume_mm3": total_volume
        }

        # Prepare the output CSV file name
        csv_file = str(mask_path).replace(".nii.gz", "_volumes.csv")

        # Save results to CSV
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["class", "voxels", "volume (mm³)"])
            # Write results
            for class_name, metrics in results.items():
                writer.writerow([class_name, metrics["voxels"], metrics["volume_mm3"]])

        print(f"Results saved to {csv_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate volumes per class from a NIfTI mask file.")
    parser.add_argument("--mask", required=True, type=str, help="Path to the NIfTI mask file.")
    args = parser.parse_args()

    calculate_volumes_per_class(args.mask)
