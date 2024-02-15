import nibabel as nib
import glob
import argparse

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Argument parsing
parser = argparse.ArgumentParser(description='Plot lesion frequency maps.')
parser.add_argument('--template_path', default='sub-mni152_space-mni_t1.nii.gz', help='Path to the template file')
parser.add_argument('--lesion_folder', required=True, help='Path to the lesion segmentations folder')
parser.add_argument('--pattern', default='_seg.nii.gz', help='Pattern to match the lesion files')
parser.add_argument('--slices', type=int, nargs='+', default=[132,113,93,84,73,44], help='Slice numbers to plot')

args = parser.parse_args()

# Load the template
template_img = nib.load(args.template_path)
template_data = template_img.get_fdata()

# Initialize an array to store the sum of all lesion segmentations
sum_lesions = np.zeros_like(template_data)

# Use a glob pattern to match all relevant files recursively
lesion_paths = glob.glob(args.lesion_folder + f'/**/*{args.pattern}', recursive=True)

length_lesions = len(lesion_paths)

# Iterate through all matching lesion segmentation files and add them up
for lesion_path in lesion_paths:
    lesion_img = nib.load(lesion_path)
    lesion_data = lesion_img.get_fdata()
    lesion_data[lesion_data >1] = 1
    sum_lesions += lesion_data

frequency_map_thresholded = sum_lesions

print(frequency_map_thresholded.max())
print(frequency_map_thresholded.min())

frequency_map_img = nib.Nifti1Image(frequency_map_thresholded, affine=template_img.affine, header=template_img.header)
frequency_map_path = "freq_map.nii.gz"
nib.save(frequency_map_img, frequency_map_path)

frequency_map_img = length_lesions* frequency_map_thresholded
slices_to_plot = args.slices

# Create a figure for plotting slices
fig, axes = plt.subplots(1, len(slices_to_plot), figsize=(25, 5))

# Reference to the displayed frequency map for the colorbar
freq_map_display = None

# Determine the global maximum and minimum across all slices to plot
global_max = np.max([frequency_map_thresholded[:, :, slice_idx].max() for slice_idx in slices_to_plot])
global_min = np.min([frequency_map_thresholded[:, :, slice_idx].min() for slice_idx in slices_to_plot])

# Your existing plotting code with modifications for consistent scaling
for idx, slice_number in enumerate(slices_to_plot):
    # Plot the original image
    axes[idx].imshow(template_data[:, :, slice_number].T, cmap='gray', origin='lower')
    freq_map_display = axes[idx].imshow(frequency_map_thresholded[:, :, slice_number].T, cmap="hot", origin='lower', alpha=0.75, vmin=global_min, vmax=global_max)
    axes[idx].axis("off")

# Increase 'left' to move it more to the right, decrease 'width' to make it thinner, and adjust 'height' as needed
cbar_ax = fig.add_axes([0.92, 0.25, 0.01, 0.35])  

# Create the colorbar with a specified aspect to control its width (narrowness)
# Decreasing the 'aspect' value makes it wider, increasing makes it narrower 
cbar = fig.colorbar(freq_map_display, cax=cbar_ax, orientation='vertical', pad=0.01, aspect=20)

# Configure the colorbar ticks
cbar.locator = ticker.MaxNLocator(integer=True)
cbar.update_ticks()

# Show the plot with the adjusted colorbar
plt.show()

