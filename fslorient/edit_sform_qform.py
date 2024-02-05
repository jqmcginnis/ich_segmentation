import argparse
import os
import nibabel as nib
import datetime
import multiprocessing
from utils import getfileList, split_list

def edit_sform_qform(im_list, sformcode, qformcode):
    """
    This function applies fslorient to edit the sform and the qform of an image. 
    It sets the sform equal to the qform and sets the sformcode and qformcode entries.
    The images will be overwritten and the result is a nifit image with an appropriate header.

    Parameters:
    -----------
    im_path : str
        Path of the image of which we want to change the header
    sformcode : int
        Value of sformcode entry, can either be 1 (Scanner Anat) or 2 (Aligned Anat)
    qformcode : int
        Value of qformcode entry, can either be 1 (Scanner Anat) or 2 (Aligned Anat)
    
    Returns:
    --------
    None 
        This function edits the header of the input image.
    """

    for i in range(0,len(im_list)):
        im_path = im_list[i]
        # load image
        img = nib.load(im_path)

        # create output folder and BIDS target folder if they do not exist
        if not os.path.exists(im_path):
            raise ValueError(f'No such file: {im_path}!')

        # check sformcode and qformcode
        if not (sformcode==1 or sformcode==2):
            raise ValueError(f'No valid number for sformcode, please use 1 (Scanner Anat) or 2 (Aligned Anat)!')
        if not (qformcode==1 or qformcode==2):
            raise ValueError(f'No valid number for qformcode, please use 1 (Scanner Anat) or 2 (Aligned Anat)!')
        

        # run fslorient copyqform2sform to set sform equal to qform
        print(f'{datetime.datetime.now()}: copyqform2sform...')
        os.system(f'fslorient -copyqform2sform {im_path}')

        # run fslorient setqformcode to set sform equal to qform
        print(f'{datetime.datetime.now()}: setqformcode...')
        os.system(f'fslorient -setqformcode {qformcode} {im_path}')

        # run fslorient setsformcode to set sform equal to qform
        print(f'{datetime.datetime.now()}: setsformcode...')
        os.system(f'fslorient -setsformcode {sformcode} {im_path}')

        print(f'{datetime.datetime.now()} {im_path}: fslorient DONE!')

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Edit sform and qform using fslorient.')
    parser.add_argument('-i', '--input_directory', help='Folder containing all images.', required=True)
    parser.add_argument('-n', '--number_of_workers', help='Number of parallel processing cores.', type=int, default=os.cpu_count()-1)

    # read the arguments
    args = parser.parse_args()

    # get directory
    dir = args.input_directory
    # get number of workers
    n_workers = args.number_of_workers
    # set sformcode and qformcode
    sform_code = 2
    qform_code = 2

    # get a list with all image files
    im_ls = getfileList(path=dir,
                        suffix='ICH0*')
    im_ls = [str(x) for x in im_ls]
    # split list for multiprocessing
    files = split_list(alist=im_ls,
                       splits=n_workers)

    # initialize multithreading
    pool = multiprocessing.Pool(processes=n_workers)
    # call function in multiprocessing setting
    for i in range(0, n_workers):
        pool.apply_async(edit_sform_qform, args=(files[i], sform_code, qform_code))

    pool.close()
    pool.join()
    
    print(f'{datetime.datetime.now()}: sform and qform editing DONE!')

