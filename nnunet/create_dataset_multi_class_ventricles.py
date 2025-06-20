import argparse
import pathlib
from pathlib import Path
import json
import os
from collections import OrderedDict
import shutil
import sys
import nibabel as nib
import numpy as np
import re
from tqdm import tqdm

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer."""
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] " if default == "yes" else " [y/N] "

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def binarize_mask(mask_file, threshold):
    """Binarizes a mask file based on the threshold."""
    img = nib.load(mask_file)
    data = img.get_fdata()
    binarized_data = np.where(data > threshold, 1, 0)
    return nib.Nifti1Image(binarized_data.astype(np.int16), img.affine, img.header)

def combine_masks(img, seg_lv, seg_v3, seg_v4):
    """Combines masks based on priority: V3 > V4 > LV."""
    combined = np.zeros_like(seg_lv.get_fdata(), dtype=np.int16)
    
    lv_data = seg_lv.get_fdata()
    v3_data = seg_v3.get_fdata()
    v4_data = seg_v4.get_fdata()
    
    # Apply based on increasing priority
    combined[lv_data > 0] = 1
    combined[v4_data > 0] = 2
    combined[v3_data > 0] = 3
    
    return nib.Nifti1Image(combined, img.affine, img.header)


def process_labels(img_path, label_paths, threshold, output_path, taskname, scan_num):
    """Processes each set of masks (LV, V3, V4) to binarize and combine."""
    lv_file = label_paths['LV']
    v3_file = label_paths['V3']
    v4_file = label_paths['V4']
    
    # Binarize each mask
    lv_binarized = binarize_mask(lv_file, threshold)
    v3_binarized = binarize_mask(v3_file, threshold)
    v4_binarized = binarize_mask(v4_file, threshold)
    
    # Combine masks
    combined_mask = combine_masks(nib.load(img_path), lv_binarized, v3_binarized, v4_binarized)
    
    # Save combined mask
    output_file = os.path.join(output_path, f'{taskname}_{scan_num:04d}.nii.gz')
    nib.save(combined_mask, output_file)
    
    return output_file

def extract_ich_id(filename):
    match = re.search(r"(ICH\d+_\d{8})", filename)
    return match.group(1) if match else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert dataset to nn-UNet format.')
    parser.add_argument('--image_directory', help='Path to image directory.', required=True)
    parser.add_argument('--label_directory', help='Path to label directory.', required=True)
    parser.add_argument('--output_directory', help='Path to output directory.', required=True)
    parser.add_argument('--taskname', help='Specify the task name, e.g. Hippocampus', default='IVH_Multi_Class', type=str)
    parser.add_argument('--tasknumber', help='Specify the task number, has to be greater than 500', default=807, type=int)
    parser.add_argument('--split_dict', help='Specify the splits using a json file.', required=True)
    parser.add_argument('--binarize_labels', action='store_true', help="Binarize the label for nn-unet.")
    parser.add_argument('--threshold', type=float, default=1e-12, help="Binarization threshold for the labels.")

    args = parser.parse_args()

    path_in_images = Path(args.image_directory)
    path_in_labels = Path(args.label_directory)
    path_out = Path(os.path.join(os.path.abspath(args.output_directory), f'Dataset{args.tasknumber}_{args.taskname}'))
    path_out_imagesTr = Path(os.path.join(path_out, 'imagesTr'))
    path_out_imagesTs = Path(os.path.join(path_out, 'imagesTs'))
    path_out_labelsTr = Path(os.path.join(path_out, 'labelsTr'))
    path_out_labelsTs = Path(os.path.join(path_out, 'labelsTs'))

    # make the directories
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_imagesTs).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTr).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labelsTs).mkdir(parents=True, exist_ok=True)

    train_image, test_image, train_image_labels, test_image_labels = [], [], [], []
    conversion_dict = {}

    images = sorted(list(path_in_images.rglob('*.nii.gz')))

    with open(args.split_dict) as f:
        splits = json.load(f)
    
    valid_train_imgs = [item for item in splits["train"]]# for item in sublist]
    valid_test_imgs = [item for item in splits["test"]] # for item in sublist]

    scan_cnt_train, scan_cnt_test = 0, 0

    for img_file in tqdm(images):
        ich_id = extract_ich_id(str(img_file))

        print(ich_id)

        # Identify corresponding label files by label type
        label_paths = {
            'LV': Path(os.path.join(path_in_labels, "LV", f'{ich_id}_ct_0000_LV_csf.nii.gz')),
            'V3': Path(os.path.join(path_in_labels, "V3", f'{ich_id}_ct_0000_V3_csf.nii.gz')),
            'V4': Path(os.path.join(path_in_labels, "V4", f'{ich_id}_ct_0000_V4_csf.nii.gz'))
        }

        print(label_paths)


        if not all(label_path.exists() for label_path in label_paths.values()):
            print(f"Skipping {ich_id} due to missing label files.")
            continue

        if f'{ich_id}_ct_0000.nii.gz' in valid_train_imgs:
            scan_cnt_train += 1
            img_file_nnunet = os.path.join(path_out_imagesTr, f'{args.taskname}_{scan_cnt_train:04d}_0000.nii.gz')
            shutil.copyfile(os.path.abspath(img_file), img_file_nnunet)
            conversion_dict[str(os.path.abspath(img_file))] = img_file_nnunet
            
            seg_file_nnunet = process_labels(img_file, label_paths, args.threshold, path_out_labelsTr, args.taskname, scan_cnt_train)
            train_image.append(str(img_file_nnunet))
            train_image_labels.append(str(seg_file_nnunet))

        elif f'{ich_id}_ct_0000.nii.gz' in valid_test_imgs:
            scan_cnt_test += 1
            img_file_nnunet = os.path.join(path_out_imagesTs, f'{args.taskname}_{scan_cnt_test:04d}_0000.nii.gz')
            shutil.copyfile(os.path.abspath(img_file), img_file_nnunet)
            conversion_dict[str(os.path.abspath(img_file))] = img_file_nnunet
            
            seg_file_nnunet = process_labels(img_file, label_paths, args.threshold, path_out_labelsTs, args.taskname, scan_cnt_test)
            test_image.append(str(img_file_nnunet))
            test_image_labels.append(str(seg_file_nnunet))

    json_dict = OrderedDict({
        'name': args.taskname,
        'description': args.taskname,
        'file_ending': ".nii.gz",
        'tensorImageSize': "3D",
        'reference': "TBD",
        'licence': "TBD",
        'release': "0.0",
        'channel_names': {"0": "ct"},
        'labels': {"background": 0, "LV": 1, "V3": 2, "V4": 3},
        'numTraining': scan_cnt_train,
        'numTest': scan_cnt_test,
        'training': [{'image': img, "label": lbl} for img, lbl in zip(train_image, train_image_labels)],
        'test': [img for img in test_image]
    })

    with open(os.path.join(path_out, "dataset.json"), "w", encoding="utf-8") as outfile:
        json.dump(json_dict, outfile, indent=4)

    with open(os.path.join(path_out, "conversion_dict.json"), "w") as outfile:
        json.dump(conversion_dict, outfile, indent=4)
