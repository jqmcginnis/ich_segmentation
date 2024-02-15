import subprocess
import logging
import shutil

def rigid_registration(fixed, moving, output_transform="output_transform.mat", warped_output="warped_output.nii"):
    """Perform rigid registration of two images using Greedy."""
    logging.info(f"Starting rigid registration: {moving} to {fixed}")

    # Construct the greedy command
    greedy_command = [
        'greedy',
        '-d', '3', # 3D registration
        '-a',  # affine registration
        '-dof', '6'  # rigid registration
        '-m', 'NMI',  # mutual information metric
        '-i', fixed, moving,
        '-o', output_transform,
        '-n', '100x50x10',  # Change iterations per level as needed
        '-ia-image-centers',  # Initialize aligning image centers

    ]

    # Run Greedy
    subprocess.run(greedy_command)

    # Apply the transformation to the moving image
    greedy_apply_command = [
        'greedy',
        '-d', '3',
        '-rf', fixed,
        '-rm', moving, warped_output,
        '-r', output_transform
    ]

    subprocess.run(greedy_apply_command)


def apply_affine(fixed, moving, transform, output, invert=False, label=False):
    """Apply affine transformation using Greedy, with an option for label images."""
    logging.info("Applying affine transformation.")

    # Construct the Greedy command
    greedy_command = [
        'greedy',
        '-d', '3',
        '-rf', fixed,
        '-rm', moving, output,
        '-r', transform
    ]

    # Use nearest-neighbor interpolation for label images
    if label:
        greedy_command += ['-ri', 'LABEL 0.2vox']

    if invert:
        greedy_command.append(',-1')

    subprocess.run(greedy_command)
