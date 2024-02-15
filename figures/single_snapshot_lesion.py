import argparse
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def create_axial_snapshots(image_path, lesion_mask_path, threshold):
    # Load the NIfTI image using nibabel
    img = nib.load(image_path)
    data = img.get_fdata()

    # Load the lesion mask image using nibabel
    lesion_mask_img = nib.load(lesion_mask_path)
    lesion_mask_data = lesion_mask_img.get_fdata()

    # Iterate over all axial slices
    # Iterate over all axial slices
    for i in range(data.shape[2]):
        # Create the axial snapshot
        axial_slice = data[:, :, i]
        mask_slice = lesion_mask_data[:,:,i]

        cmap = plt.cm.gray
        norm = plt.Normalize(axial_slice.min(), axial_slice.max())

        rgba = cmap(norm(axial_slice))

        rgba[mask_slice==1] = 1, 0 , 0

        # Plot the axial slice
        plt.imshow(np.rot90(rgba), interpolation='nearest')
        plt.axis('off')

        # Save the image as PNG with sequential filenames
        png_filename = os.path.splitext(image_path)[0] + f"_{str(i).zfill(4)}.png"
        plt.savefig(png_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Axial snapshot {i} saved as {png_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create axial snapshots of a NIfTI image.")
    parser.add_argument("--image", required=True, help="Path to the NIfTI image")
    parser.add_argument("--mask", required=True, help="Path to the lesion mask image")
    args = parser.parse_args()

    create_axial_snapshots(args.image, args.mask, args.threshold)
