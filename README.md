#### ICH Segmentation
This github repository hosts source code for training an nn-UNet based model for ICH segmentation.

#### Incoming Datastructure

```
jmcginnis@chouffe:/media/raid3/FelixH/ICH_Scans/ICH_nnunet/final_training$ tree
.
└── ICH_nnUNet_training_100
    ├── CT
    │   ├── test
    │   │   ├── images_test
    │   │   │   ├── ICH00117_20190502_ct.nii.gz
    │   │   │   ├── ICH00118_20190114_ct.nii.gz
    │   │   │   ├── ICH00119_20201103_ct.nii.gz
...
    │   │   └── lesionmask_test
    │   │       ├── ICH00117_lesionmask.nii.gz
    │   │       ├── ICH00118_lesionmask.nii.gz
    │   │       ├── ICH00119_lesionmask.nii.gz
...
    │   └── training
    │       ├── images
    │       │   ├── ICH00001_20190703_ct.nii.gz
    │       │   ├── ICH00002_20190728_ct.nii.gz
    │       │   ├── ICH00003_20200506_ct.nii.gz
...
    │       └── lesionmask
    │           ├── ICH00001_lesionmask.nii.gz
    │           ├── ICH00002_lesionmask.nii.gz
    │           ├── ICH00003_lesionmask.nii.gz
...
```

#### Generate Dataset

```
python3 create_dataset.py --image_directory /media/raid3/FelixH/ICH_Scans/ICH_nnunet/final_training/ICH_nnUNet_training_100/CT/training/images/ --label_directory /media/raid3/FelixH/ICH_Scans/ICH_nnunet/final_training/ICH_nnUNet_training_100/CT/training/lesionmask/ --taskname ICHSegmentationSubmission --tasknumber 802 --split_dict train_dict.json --output_directory /home/jmcginnis/git_repositories/nnUNetV2_database/nnUNet_raw/
```

#### Plan and preprocess
```
nnUNetv2_plan_and_preprocess -d 802 --verify_dataset_integrity --verbose
```

#### Train 2D nn-UNet
```
nnUNetv2_train 802 2d 0 --npz
nnUNetv2_train 802 2d 1 --npz
nnUNetv2_train 802 2d 2 --npz
nnUNetv2_train 802 2d 3 --npz
nnUNetv2_train 802 2d 4 --npz
```

#### Train 3D nn-UNet
```
nnUNetv2_train 802 3d_fullres 0 --npz
nnUNetv2_train 802 3d_fullres 1 --npz
nnUNetv2_train 802 3d_fullres 2 --npz
nnUNetv2_train 802 3d_fullres 3 --npz
nnUNetv2_train 802 3d_fullres 4 --npz
```

