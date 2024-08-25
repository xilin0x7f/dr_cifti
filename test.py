# Author: 赩林, xilin0x7f@163.com
import argparse
import nibabel as nib
import numpy as np

import subprocess

def convertCifti2Nifti(cifti_file_path, nifti_file_path):
    cifti_file = nib.load(cifti_file_path)
    cifti_data = cifti_file.get_fdata()

    nii_img = nib.Nifti1Image(cifti_data.T[:, np.newaxis, np.newaxis, :], np.eye(4))

    nib.save(nii_img, nifti_file_path)
    pass

convertCifti2Nifti(
    r'/mnt/h/melodic_test/group.ica/melodic_IC.dscalar.nii',
    r'/mnt/h/melodic_test/group.ica/melodic_IC.nii.gz'
)

def convert_nifti_to_cifti(nifti_path, cifti_ref_path, cifti_out_path):
    nii_header = nib.load(nifti_path)
    data = nii_header.get_fdata()
    data = data[:, 0, 0, :].T

    cifti_img = nib.load(cifti_ref_path)
    new_cifti_img = nib.Cifti2Image(data, header=cifti_img.header, nifti_header=cifti_img.nifti_header)
    nib.save(new_cifti_img, cifti_out_path)

convert_nifti_to_cifti(
    r'/mnt/h/melodic_test/group.ica/melodic_IC.nii.gz',
                       r'/mnt/h/melodic_test/group.ica/melodic_IC.dscalar.nii',
                       r'/mnt/h/melodic_test/group.ica/melodic_IC_new.dscalar.nii')