import os
import argparse
import logging
import multiprocessing
import glob
from pipeline import process_image_segmentation
from utils import find_files_with_string_in_name

def main():

    """Main function to process images."""
    parser = argparse.ArgumentParser(description='Image Registration and Skull Stripping.')
    parser.add_argument('--image_directory', type=str, required=True, help='Path to images.')
    parser.add_argument('--seg_directory', type=str, required=True, help='Path to the segmentation directory.')
    parser.add_argument('--atlas_path', type=str, required=True, help='Path to the atlas image')
    parser.add_argument('--ct_label', type=str, default='.nii.gz', help='CT image label')
    parser.add_argument('--seg_label', type=str, default='.nii.gz', help='CT segmentation label')

    parser.add_argument('--num_processes', type=int, default=4, help="Number of processes in parallel.")
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    # Configure logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')

    logging.basicConfig(level=numeric_level)

    # glob label_directory and save to list
    image_files = sorted(find_files_with_string_in_name(args.image_directory, args.ct_label))
    seg_files = sorted(find_files_with_string_in_name(args.seg_directory, args.seg_label))

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        pool.starmap(process_image_segmentation, [(image, seg, args.atlas_path) for image, seg in zip(image_files, seg_files)])

if __name__ == "__main__":
    main()
