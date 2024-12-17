import os
import argparse
import nibabel as nib
import numpy as np
import glob




def combine_masks(img, seg_sv, seg_v3, seg_v4):
    """Combines masks based on priority rules and handles overlaps, using V4 label for SV and V4 overlap."""
    combined = np.zeros_like(seg_sv.get_fdata(), dtype=np.int16)
    
    sv_data = seg_sv.get_fdata()
    v3_data = seg_v3.get_fdata()
    v4_data = seg_v4.get_fdata()
    
    # Apply labels: SV=1, V3=2, V4=3
    combined[sv_data > 0] = 1
    combined[v3_data > 0] = 2
    combined[v4_data > 0] = 3
    
    # Resolve overlaps
    combined[(sv_data > 0) & (v3_data > 0)] = 2
    combined[(v4_data > 0) & (v3_data > 0)] = 2
    
    # Handle SV and V4 overlap by using V4 label and issuing a warning
    if np.any((sv_data > 0) & (v4_data > 0)):
        combined[(sv_data > 0) & (v4_data > 0)] = 3
        print("Overlap between seg-SV and seg-V4 found, which should not happen. Using V4 label for these overlaps.")
    
    return nib.Nifti1Image(combined, img.affine, img.header)


def main(sv_dir, v4_dir, v3_dir, output_dir):
    # Get sorted file lists
    # Get sorted list of .nii.gz files in each directory

    sv_files = sorted(glob.glob(os.path.join(sv_dir, "*.nii.gz")))
    v4_files = sorted(glob.glob(os.path.join(v4_dir, "*.nii.gz")))
    v3_files = sorted(glob.glob(os.path.join(v3_dir, "*.nii.gz")))
    # Ensure all directories have the same filenames
    assert len(sv_files) == len(v4_files) == len(v3_files), "Input directories must contain matching filenames."
    
    # Process each file
    for i in range(len(sv_files)):
        sv_path = sv_files[i]
        v4_path = v4_files[i]
        v3_path = v3_files[i]
        
        # Load NIfTI files
        img = nib.load(sv_path)  # Use any file for affine and header
        seg_sv = nib.load(sv_path)
        seg_v4 = nib.load(v4_path)
        seg_v3 = nib.load(v3_path)
        
        # Combine masks
        combined_mask = combine_masks(img, seg_sv, seg_v3, seg_v4)
        
        # Save the combined mask
        output_filename = os.path.basename(sv_path).replace("_seg-sv.nii.gz", "_seg-comb.nii.gz")
        output_path = os.path.join(output_dir, output_filename)
        nib.save(combined_mask, output_path)
        print(f"Saved combined mask: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine segmentation masks from three directories.")
    parser.add_argument("--sv", required=True, help="Directory containing seg-SV masks")
    parser.add_argument("--v4", required=True, help="Directory containing seg-V4 masks")
    parser.add_argument("--v3", required=True, help="Directory containing seg-V3 masks")
    parser.add_argument("--output", required=True, help="Output directory for combined masks")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    main(args.sv, args.v4, args.v3, args.output)
