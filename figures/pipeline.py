import logging
from utils import rigid_registration, apply_affine, getSubjectID

def process_image_segmentation(image, segmentation, atlas):

    print(getSubjectID(image))
    print(getSubjectID(segmentation))

    assert getSubjectID(image) == getSubjectID(segmentation), 'different subjects, aborting!'


    processed_image = image.replace(".nii.gz", "_processed.nii.gz")
    affine = image.replace(".nii.gz", "_affine.mat")
    processed_segmentation = segmentation.replace(".nii.gz", "_processed.nii.gz")

    try:
        rigid_registration(fixed=atlas,
                            moving=image,
                            output_transform=affine,
                            warped_output=processed_image)
        apply_affine(fixed=atlas,
                      moving=segmentation,
                      output=processed_segmentation,
                      transform=affine,
                      invert=False,
                      label=True)

    except Exception as e:
        logging.info(e)