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
parser.add_argument('--lesion_folder1', required=True, help='Path to the first lesion segmentations folder')
parser.add_argument('--lesion_folder2', required=True, help='Path to the second lesion segmentations folder')
parser.add_argument('--pattern', default='processed.nii.gz', help='Pattern to match the lesion files')
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

# Process both lesion folders
sum_lesions1 = process_lesions(args.lesion_folder1)
sum_lesions2 = process_lesions(args.lesion_folder2)

# Calculating global max/min for consistent color scaling
global_max = max(sum_lesions1.max(), sum_lesions2.max())
global_min = min(sum_lesions1.min(), sum_lesions2.min())

# Create a figure for plotting slices in a 2x6 grid
fig, axes = plt.subplots(2, len(args.slices), figsize=(6.69, 2.5))

# Plotting for both lesion sets and adding titles
for idx, slice_number in enumerate(args.slices):
    # First row for lesion_folder1
    axes[0, idx].imshow(template_data[:, :, slice_number].T, cmap='gray', origin='lower')
    img1 = axes[0, idx].imshow(sum_lesions1[:, :, slice_number].T, cmap="hot", origin='lower', alpha=0.75)
    img1.set_clim(vmin=global_min, vmax=global_max)
    axes[0, idx].axis("off")
    
    # Second row for lesion_folder2
    axes[1, idx].imshow(template_data[:, :, slice_number].T, cmap='gray', origin='lower')
    img2 = axes[1, idx].imshow(sum_lesions2[:, :, slice_number].T, cmap="hot", origin='lower', alpha=0.75)
    img2.set_clim(vmin=global_min, vmax=global_max)
    axes[1, idx].axis("off")

# Adding label A and B to the left of the first column
fig.text(0.1, 0.70, "A", fontsize=9, ha='center', va='center')#, weight='bold')
fig.text(0.1, 0.28, "B", fontsize=9, ha='center', va='center')#, weight='bold')

# Adjust the colorbar to reflect the heatmaps
cbar_ax = fig.add_axes([0.92, 0.1125, 0.0075, 0.765])
cbar = fig.colorbar(img2, cax=cbar_ax, orientation='vertical', pad=0.01)  # Using img2 for the colorbar
cbar.locator = ticker.MaxNLocator(integer=True)
cbar.ax.tick_params(labelsize=6)
cbar.update_ticks()

# Add a title to the color bar
cbar.set_label('N', rotation=90, labelpad=7, fontsize=9)

# plt.tight_layout()
plt.savefig("Fig2.tiff", dpi=300)
