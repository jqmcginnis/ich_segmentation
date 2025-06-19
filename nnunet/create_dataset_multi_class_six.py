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
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def process_single_mask(label_path):
    img = nib.load(label_path)
    data = img.get_fdata().astype(np.uint8)

    return nib.Nifti1Image(data, img.affine, img.header)

def extract_ich_id(filename):
    match = re.search(r"(ICH\d+)_(\d{8})", filename)
    return match.group() if match else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert single multiclass label dataset to nn-UNet format.')
    parser.add_argument('--image_directory', required=True)
    parser.add_argument('--label_directory', required=True)
    parser.add_argument('--output_directory', required=True)
    parser.add_argument('--taskname', default='IVH_CSF_Multi_Class', type=str)
    parser.add_argument('--tasknumber', default=810, type=int)
    parser.add_argument('--split_dict', required=True)

    args = parser.parse_args()

    path_in_images = Path(args.image_directory)
    path_in_labels = Path(args.label_directory)
    path_out = Path(os.path.join(os.path.abspath(args.output_directory), f'Dataset{args.tasknumber}_{args.taskname}'))
    path_out_imagesTr = path_out / 'imagesTr'
    path_out_imagesTs = path_out / 'imagesTs'
    path_out_labelsTr = path_out / 'labelsTr'
    path_out_labelsTs = path_out / 'labelsTs'

    for p in [path_out, path_out_imagesTr, path_out_imagesTs, path_out_labelsTr, path_out_labelsTs]:
        p.mkdir(parents=True, exist_ok=True)

    train_image, test_image, train_image_labels, test_image_labels = [], [], [], []
    conversion_dict = {}

    images = sorted(path_in_images.rglob('*.nii.gz'))

    with open(args.split_dict) as f:
        splits = json.load(f)

    valid_train_imgs = [item for item in splits["train"]]
    valid_test_imgs = [item for item in splits["test"]]

    scan_cnt_train, scan_cnt_test = 0, 0

    for img_file in tqdm(images):
        ich_id = extract_ich_id(str(img_file))
        if ich_id is None:
            continue

        label_path = path_in_labels / f"{ich_id}_seg-multiclass.nii.gz"
        if not label_path.exists():
            print(f"Skipping {ich_id} due to missing label file.")
            continue

        if f'{ich_id}_ct_0000.nii.gz' in valid_train_imgs:
            scan_cnt_train += 1
            img_out = path_out_imagesTr / f'{args.taskname}_{scan_cnt_train:04d}_0000.nii.gz'
            shutil.copyfile(img_file, img_out)
            conversion_dict[str(img_file)] = str(img_out)

            label_out = path_out_labelsTr / f'{args.taskname}_{scan_cnt_train:04d}.nii.gz'
            combined_mask = process_single_mask(label_path)
            nib.save(combined_mask, label_out)

            train_image.append(str(img_out))
            train_image_labels.append(str(label_out))

        elif f'{ich_id}_ct_0000.nii.gz' in valid_test_imgs:
            scan_cnt_test += 1
            img_out = path_out_imagesTs / f'{args.taskname}_{scan_cnt_test:04d}_0000.nii.gz'
            shutil.copyfile(img_file, img_out)
            conversion_dict[str(img_file)] = str(img_out)

            label_out = path_out_labelsTs / f'{args.taskname}_{scan_cnt_test:04d}.nii.gz'
            combined_mask = process_single_mask(label_path)
            nib.save(combined_mask, label_out)

            test_image.append(str(img_out))
            test_image_labels.append(str(label_out))

    json_dict = OrderedDict({
        'name': args.taskname,
        'description': args.taskname,
        'file_ending': ".nii.gz",
        'tensorImageSize': "3D",
        'reference': "TBD",
        'licence': "TBD",
        'release': "0.0",
        'channel_names': {"0": "ct"},
        'labels': {
            "background": 0,
            "LV_CSF": 1,
            "V4_CSF": 2,
            "V3_CSF": 3,
            "LV_IVH": 4,
            "V4_IVH": 5,
            "V3_IVH": 6
        },
        'numTraining': scan_cnt_train,
        'numTest': scan_cnt_test,
        'training': [{'image': img, "label": lbl} for img, lbl in zip(train_image, train_image_labels)],
        'test': [img for img in test_image]
    })

    with open(path_out / "dataset.json", "w", encoding="utf-8") as outfile:
        json.dump(json_dict, outfile, indent=4)

    with open(path_out / "conversion_dict.json", "w") as outfile:
        json.dump(conversion_dict, outfile, indent=4)
