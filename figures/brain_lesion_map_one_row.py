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
parser.add_argument('--lesion_folder1', required=True, help='Path to the lesion segmentations folder')
parser.add_argument('--pattern', default='_processed.nii.gz', help='Pattern to match the lesion files')
parser.add_argument('--slices', type=int, nargs='+', default=[132,113,93,84,73,44], help='Slice numbers to plot')

args = parser.parse_args()

# Load the template
template_img = nib.load(args.template_path)
template_data = template_img.get_fdata()

# Function to process lesions
def process_lesions(lesion_folder):
    sum_lesions = np.zeros_like(template_data)
    lesion_paths = glob.glob(lesion_folder + f'/**/*{args.pattern}', recursive=True)
    for lesion_path in lesion_paths:
        lesion_img = nib.load(lesion_path)
        lesion_data = lesion_img.get_fdata()
        lesion_data[lesion_data > 1] = 1
        sum_lesions += lesion_data
    return sum_lesions

# Process the lesion folder
sum_lesions1 = process_lesions(args.lesion_folder1)

# Calculating max for consistent color scaling
global_max = sum_lesions1.max()
global_min = sum_lesions1.min()

# Create a figure for plotting slices
fig, axes = plt.subplots(1, len(args.slices), figsize=(25, 5)) # Adjusted to 1 row

# Plotting for lesion set and adding titles
for idx, slice_number in enumerate(args.slices):
    axes[idx].imshow(template_data[:, :, slice_number].T, cmap='gray', origin='lower')
    img1 = axes[idx].imshow(sum_lesions1[:, :, slice_number].T, cmap="hot", origin='lower', alpha=0.75)
    img1.set_clim(vmin=global_min, vmax=global_max)
    axes[idx].axis("off")

# Adjust the colorbar to reflect the heatmap
cbar_ax = fig.add_axes([0.92, 0.155, 0.01, 0.685])
cbar = fig.colorbar(img1, cax=cbar_ax, orientation='vertical', pad=0.01)  # Adjusted to use img1 for the colorbar
cbar.locator = ticker.MaxNLocator(integer=True)
cbar.update_ticks()

# Add a title to the color bar
cbar.set_label('Number of patients', rotation=90, labelpad=20, fontsize=12)

plt.show()
