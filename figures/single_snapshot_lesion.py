import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib.colors import ListedColormap
import argparse

def overlay_slices(image_path, mask_path):
    # Load the NIfTI image and mask
    image = nib.load(image_path)
    image_data = image.get_fdata()
    mask = nib.load(mask_path)
    mask_data = mask.get_fdata()


    # Create a custom colormap
    red_color = [0.5, 0, 0, 1.0]  # [R, G, B, Alpha]
    colors = [red_color, red_color]
    cmap = ListedColormap(colors)

    # Loop through all slices and save each as a PNG file
    for slice_index in range(image_data.shape[2]):
        # Get the current slice from the image and mask
        image_slice = image_data[:, :, slice_index]
        mask_slice = mask_data[:, :, slice_index]

        # Create a masked array to set transparency for mask value 0
        masked_data = np.ma.masked_equal(mask_slice, 0)

        # Plot the image and overlay the masked data
        plt.imshow(np.rot90(image_slice), cmap='gray')
        plt.imshow(np.rot90(masked_data), cmap=cmap, alpha=0.75)#, interpolation='linear')

        # Remove the colorbar
        plt.colorbar().remove()
        plt.axis('off')

        # Save the current slice as a PNG file
        plt.savefig(f'slice_{slice_index}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()  # Clear the figure for the next iteration

    print('Slices saved as PNG files.')

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Overlay voxel data on NIfTI images.')
    parser.add_argument('--image', type=str, help='Path to the NIfTI image file.')
    parser.add_argument('--mask', type=str, help='Path to the NIfTI mask file.')

    # Parse the arguments
    args = parser.parse_args()

    # Overlay slices with the provided image and mask paths
    overlay_slices(args.image, args.mask)
