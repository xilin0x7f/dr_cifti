# Author: 赩林, xilin0x7f@163.com
import argparse
import nilearn
import nibabel as nib
import numpy as np

cifti_ref_path = r'F:\melodic_test\group.ica\melodic_IC.dscalar.nii'
nifti_path = r'F:\melodic_test\grot\dr_stage3_ic0000_tstat1.nii.gz'
cifti_out_path = r'F:\melodic_test\group.ica\dr_stage3_ic0000_tstat1.dscalar.nii'
#%%
cifti_img = nib.load(cifti_ref_path)

nii_header = nib.load(nifti_path)
data = nii_header.get_fdata()
if data.ndim < 4:
    data = data.reshape(-1, 1)

data = data[:cifti_img.get_fdata().shape[1], :].T

cifti_header = cifti_img.header.copy()
cifti_header.matrix[0]._maps = [cifti_header.matrix[0]._maps[0] for i in range(data.shape[0])]
new_cifti_img = nib.Cifti2Image(data, header=cifti_header, nifti_header=cifti_img.nifti_header)
nib.save(new_cifti_img, cifti_out_path)
