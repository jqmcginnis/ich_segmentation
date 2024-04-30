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

Training the ICH segmentation network:
```
python3 create_dataset.py --image_directory /media/raid3/FelixH/ICH_Scans/ICH_nnunet/final_training/ICH_nnUNet_training_100/CT/training/images/ --label_directory /media/raid3/FelixH/ICH_Scans/ICH_nnunet/final_training/ICH_nnUNet_training_100/CT/training/lesionmask/ --taskname ICHSegmentationSubmission --tasknumber 802 --split_dict train_dict.json --output_directory /home/jmcginnis/git_repositories/nnUNetV2_database/nnUNet_raw/
```
Training an IVH segmentation network:
```
python3 create_dataset.py --image_directory /home/jmcginnis/raid_access2/Julian/nnUNet_training1/scans --label_directory /home/jmcginnis/raid_access2/Julian/nnUNet_training1/lesionsmasks --taskname ICH_PrelabelingIVH --tasknumber 803 --split_dict train_dict_ivh_prelabeling.json --output_directory /home/jmcginnis/git_repositories/nnUNetV2_database/nnUNet_raw/
```

Training a revised ICH segmentation network:

```
python3 create_dataset.py --image_directory /home/jmcginnis/raid_access2/FelixH/ICH/ICH_lesionsegmentation/CT/PH/trainingset/scans --label_directory /home/jmcginnis/raid_access2/FelixH/ICH/ICH_lesionsegmentation/CT/PH/trainingset/lesionmasks --taskname ICH_Segmentation_PH --tasknumber 804 --split_dict 804_train_dict_ich_ph.json --output_directory /home/jmcginnis/git_repositories/nnUNetV2_database/nnUNet_raw/
```

Training the final IVH segmentation network:
```
python3 create_dataset.py --image_directory /home/jmcginnis/raid_access2/FelixH/ICH/ICH_lesionsegmentation/CT/IVH/nnUNet_training_100/scans/final_training/nnUNet/training --label_directory  /home/jmcginnis/raid_access2/FelixH/ICH/ICH_lesionsegmentation/CT/IVH/nnUNet_training_100/lesionmasks/final_training/nnUNet/training  --taskname ICH_Segmentation_IVH_final --tasknumber 805 --split_dict 805_train_dict_ich_ivh_final.json --output_directory /home/jmcginnis/git_repositories/nnUNetV2_database/nnUNet_raw/
```

Training a prelebaling network for edema segmentation:
```
python3 create_dataset.py --image_directory /home/jmcginnis/raid_access2/FelixH/ICH/ICH_lesionsegmentation/CT/edema/nnUNet_training_presegmentation/scans --label_directory  /home/jmcginnis/raid_access2/FelixH/ICH/ICH_lesionsegmentation/CT/edema/nnUNet_training_presegmentation/lesionmasks  --taskname ICH_Segmentation_IVH_final --tasknumber 806 --split_dict 806_train_dict_edema_prelabeling.json --output_directory /home/jmcginnis/git_repositories/nnUNetV2_database/nnUNet_raw/
```



#### Plan and preprocess
```
nnUNetv2_plan_and_preprocess -d 802 --verify_dataset_integrity --verbose
```

#### Train 2D nn-UNet
```
nnUNetv2_train 805 2d 0 --npz
nnUNetv2_train 805 2d 1 --npz
nnUNetv2_train 805 2d 2 --npz
nnUNetv2_train 805 2d 3 --npz
nnUNetv2_train 805 2d 4 --npz
```

#### Train 3D nn-UNet
```
nnUNetv2_train 805 3d_fullres 0 --npz
nnUNetv2_train 805 3d_fullres 1 --npz
nnUNetv2_train 805 3d_fullres 2 --npz
nnUNetv2_train 805 3d_fullres 3 --npz
nnUNetv2_train 805 3d_fullres 4 --npz
```

#### To run inference

Do not forget to maintain the same naming convention (e.g. `_0000.nii.gz` for your inference images) as in your training set!
You may do this with the linux `rename 's/ct.nii.gz/_0000.nii.gz' *` tool.

Run inference:
```
nnUNetv2_predict -d 802 -i /path/to/img -o /path/to/output -c 3d_fullres --verbose 
```
### Citation

Please cite nn-UNet if you decide to train similar networks as we do.
```
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

If you find our code helpful in setting up the nnUNet dataset format, or you use our plotting code, please consider citing us as well.



