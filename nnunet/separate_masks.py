import os
import argparse
import nibabel as nib
import numpy as np
import glob


def separate_masks(combined_img):
    """
    Separates a combined lesion mask into binary masks for each label.
    Assumes labels: SV=1, V3=2, V4=3.
    """
    combined_data = combined_img.get_fdata()

    # Create binary masks
    sv_mask = (combined_data == 1).astype(np.uint8)  # Binary mask for SV
    v3_mask = (combined_data == 2).astype(np.uint8)  # Binary mask for V3
    v4_mask = (combined_data == 3).astype(np.uint8)  # Binary mask for V4

    sv_img = nib.Nifti1Image(sv_mask, combined_img.affine, combined_img.header)
    v3_img = nib.Nifti1Image(v3_mask, combined_img.affine, combined_img.header)
    v4_img = nib.Nifti1Image(v4_mask, combined_img.affine, combined_img.header)

    return sv_img, v3_img, v4_img


def process_mask(file_path, output_dir):
    """
    Processes a single mask file to separate it into binary masks.
    """
    # Load the combined mask
    combined_img = nib.load(file_path)

    # Separate into individual binary masks
    sv_img, v3_img, v4_img = separate_masks(combined_img)

    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(file_path))[0]  # Strip extension
    sv_filename = os.path.join(output_dir, f"{base_name}_sv.nii.gz")
    v3_filename = os.path.join(output_dir, f"{base_name}_v3.nii.gz")
    v4_filename = os.path.join(output_dir, f"{base_name}_v4.nii.gz")

    # Save the separated binary masks
    nib.save(sv_img, sv_filename)
    nib.save(v3_img, v3_filename)
    nib.save(v4_img, v4_filename)

    print(f"Saved: {sv_filename}, {v3_filename}, {v4_filename}")


def main(input_dir, output_dir):
    # Get all .nii.gz files in the input directory
    mask_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))

    if not mask_files:
        print("No .nii.gz files found in the input directory.")
        return

    for mask_file in mask_files:
        process_mask(mask_file, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate any combined lesion mask into binary masks.")
    parser.add_argument("--input", required=True, help="Directory containing combined lesion masks.")
    parser.add_argument("--output", required=True, help="Output directory for separated binary masks.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    main(args.input, args.output)
