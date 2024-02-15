import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Argument parsing
parser = argparse.ArgumentParser(description='Plot lesion frequency maps.')
parser.add_argument('--template_path', default='sub-mni152_space-mni_t1.nii.gz', help='Path to the template file')
parser.add_argument('--atlas_path', default='JHU-ICBM-labels-1mm_warped.nii.gz', help='Path to the template file')
parser.add_argument('--lesion_folder', required=True, help='Path to the lesion segmentations folder')
parser.add_argument('--pattern', default='_seg.nii.gz', help='Pattern to match the lesion files')
parser.add_argument('--slices', type=int, nargs='+', default=[54,74,94,114], help='Slice numbers to plot')

args = parser.parse_args()

# Load the template
template_img = nib.load(args.template_path)
template_data = template_img.get_fdata()

# Load the atlas
atlas_img = nib.load(args.atlas_path)
atlas_data = atlas_img.get_fdata()

# Initialize an array to store the sum of all lesion segmentations
sum_lesions = np.zeros_like(template_data)

# Use a glob pattern to match all relevant files recursively
lesion_paths = glob.glob(args.lesion_folder + f'/**/*{args.pattern}', recursive=True)

# Iterate through all matching lesion segmentation files and add them up
for lesion_path in lesion_paths:
    lesion_img = nib.load(lesion_path)
    lesion_data = lesion_img.get_fdata()
    lesion_data[lesion_data >1] = 1
    sum_lesions += lesion_data

# Normalize and save the sum
max_value = sum_lesions.max()
if max_value > 0:
    sum_lesions = sum_lesions / max_value

# Set the threshold
threshold = 0.1

# Create a masked array where values below the threshold are set to NaN
frequency_map_thresholded = np.where(sum_lesions > threshold, sum_lesions, np.nan)

frequency_map_img = nib.Nifti1Image(frequency_map_thresholded, affine=template_img.affine, header=template_img.header)
frequency_map_path = "freq_map.nii.gz"
nib.save(frequency_map_img, frequency_map_path)

# Additional code as defined before

corticospinal_tract_R = np.equal(atlas_data, 7)
corticospinal_tract_L = np.equal(atlas_data, 8)

corticospinal_tract_R = corticospinal_tract_R.astype(float)
corticospinal_tract_L = corticospinal_tract_L.astype(float)

# Create a colormap that goes from transparent to red (for example)
colors = [(1, 0, 0, 0),(1,0,0,0.5), (1, 0, 0, 1)]  # R -> G -> B
n_bin = 10  # Discretizes the interpolation into bins
cmap_name = 'custom1'
cm_custom = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

# Create a colormap that goes from transparent to red (for example)
colors2 = [(0, 0, 1, 0),(0,0,1,0.5), (0, 0, 1, 1)]  # R -> G -> B
n_bin = 10  # Discretizes the interpolation into bins
cmap_name2 = 'custom2'
cm_custom2 = mcolors.LinearSegmentedColormap.from_list(cmap_name2, colors2, N=100)
# Define slices to plot
slices_to_plot = args.slices

# Create a figure for plotting slices
fig, axes = plt.subplots(1, len(slices_to_plot), figsize=(25, 5))

# Reference to the displayed frequency map for the colorbar
freq_map_display = None

for idx, slice_number in enumerate(slices_to_plot):
    # Plot the original image
    axes[idx].imshow(template_data[:, :, slice_number].T, cmap='gray', origin='lower')
    # Overlay the thresholded frequency map using the custom colormap and store the reference
    if freq_map_display is None:
        freq_map_display = axes[idx].imshow(frequency_map_thresholded[:, :, slice_number].T, cmap="jet", origin='lower', alpha=0.7)
    else:
        axes[idx].imshow(frequency_map_thresholded[:, :, slice_number].T, cmap="jet", origin='lower', alpha=0.7)
    axes[idx].imshow(corticospinal_tract_L[:, :, slice_number].T, cmap=cm_custom2, origin='lower', alpha=0.35)
    axes[idx].imshow(corticospinal_tract_R[:, :, slice_number].T, cmap=cm_custom2, origin='lower', alpha=0.35)

    # axes[idx].set_title(f'MNI152 Lesion Frequency Map (Slice {slice_number})')
    axes[idx].axis("off")

# Add a colorbar for the frequency map
cbar = fig.colorbar(freq_map_display, ax=axes.ravel().tolist(), orientation='vertical', pad=0.01, aspect=40)
cbar.set_label('Lesion Frequency', rotation=270, labelpad=15)

# plt.tight_layout()
plt.savefig("mni152_lesion_map_slices.png", pad_inches=0, transparent=True, dpi=300)
plt.show()