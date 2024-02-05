
# Edit Sform and Qform of NIfTI Images

This script edits the Sform and Qform of NIfTI (Neuroimaging Informatics Technology Initiative) images using the FSL (FMRIB Software Library) tool `fslorient`.

## Prerequisites

Before using this script, ensure you have the following installed:
- Python 3.x
- `argparse`
- `nibabel`
- FSL (FMRIB Software Library)

## Usage

Run the script with the following command:

```bash
python3 edit_sform_qform.py -i /path/to/input_directory -n <number_of_workers>
```

### Arguments

- `-i, --input_directory`: Path to the folder containing all NIfTI images.
- `-n, --number_of_workers`: Number of parallel processing cores to utilize (default is `os.cpu_count() - 1`).

## Functionality

The script edits the Sform and Qform of NIfTI images in the specified directory using the `fslorient` tool. It sets the Sform equal to the Qform and updates the Sformcode and Qformcode entries to 1 (Scanner Anat) or 2 (Aligned Anat). The values of Sformcode and Qformcode need to be defined in the script. The images will be overwritten with the updated headers.