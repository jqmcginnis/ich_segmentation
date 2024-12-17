import nibabel as nib
import numpy as np
import os
import csv
import argparse

def calculate_lesion_volume(mask_path):
    try:
        # Load the NIfTI file
        nifti_img = nib.load(mask_path)
        data = nifti_img.get_fdata()
        header = nifti_img.header
        
        # Calculate voxel volume in mm³
        voxel_size = np.prod(header.get_zooms())
        
        # Calculate number of lesion voxels
        lesion_voxels = np.sum(data == 1)
        
        # Calculate lesion volume
        lesion_volume = lesion_voxels * voxel_size  # in mm³
        
        # Prepare the output CSV file name
        csv_file = str(mask_path).replace(".nii.gz", ".csv")
        
        # Save results to CSV
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "ph_voxels", "ph_volume (mm³)"])
            writer.writerow([os.path.basename(mask_path), lesion_voxels, lesion_volume])
        
        print(f"Results saved to {csv_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate lesion volume from a NIfTI mask file.")
    parser.add_argument("--mask", required=True, type=str, help="Path to the NIfTI mask file.")
    args = parser.parse_args()
    
    calculate_lesion_volume(args.mask)
