import subprocess
import logging
import shutil
import re
import glob
import shlex

def find_files_with_string_in_name(directory_path, string_in_name):
    # This pattern will match any files that include the specified string in their names
    # '**' allows for searching recursively in all subdirectories
    pattern = f"{directory_path}/**/*{string_in_name}*"
    
    # Use glob.glob with the pattern and recursive=True to find matching file paths
    files = glob.glob(pattern, recursive=True)
    
    return files

def getSubjectID(path):
    """
    This function extracts the ICH ID from the file name, removing any prefixes and returning only the numeric part with 'ICH' prepended.

    Parameters:
    -----------
    path : str
        Path to data file or the filename itself.
    
    Returns:
    --------
    found : str 
        Extracted ICH ID, formatted as 'ICHxxxxx'.
    """
    # Extract the last part of the path after the last slash (if present)
    filename = path.split("/")[-1]
    
    # Adjusted regular expression to correctly capture only the numeric part after 'bwsICH' or 'wICH'
    try:
        # This regular expression now correctly ignores the prefix and captures only the digits
        digits = re.search(r'(?:bwsICH|wICH)(\d+)', filename).group(1)
        found = 'ICH' + digits  # Prepend 'ICH' to the captured digits
    except AttributeError:
        found = ''
    
    return found

def rigid_registration(fixed, moving, output_transform="output_transform.mat", warped_output="warped_output.nii", threads=8):
    """Perform rigid registration of two images using Greedy."""
    logging.info(f"Starting rigid registration: {moving} to {fixed}")

    # Construct the greedy command
    greedy_command = [
        'greedy',
        '-d', '3', # 3D registration
        '-threads', f'{threads}',
        '-a',  # affine registration
        '-dof', '6',  # rigid registration
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
        '-threads', f'{threads}',
        '-d', '3',
        '-rf', fixed,
        '-rm', moving, warped_output,
        '-r', output_transform
    ]

    subprocess.run(greedy_apply_command)


def apply_affine(fixed, moving, transform, output, invert=False, label=False, threads=8):
    """Apply affine transformation using Greedy, with an option for label images."""
    logging.info("Applying affine transformation.")

    # Construct the Greedy command
    if invert:
        greedy_command = f"greedy -threads {threads} -d 3 -rf {fixed} -rm {moving} {output} -r {transform},-1"
    else:
        greedy_command = f"greedy -threads {threads} -d 3 -rf {fixed} -rm {moving} {output} -r {transform}"

    # Use nearest-neighbor interpolation for label images
    if label:
        greedy_command += f" -ri LABEL 0.2vox"
        # greedy_command += f" -ri NN"

    print(greedy_command)

    subprocess.run(shlex.split(greedy_command))
