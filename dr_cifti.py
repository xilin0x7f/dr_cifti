#!/opt/hky-venv/bin/python
# Author: 赩林, xilin0x7f@163.com
import argparse
import os

import nibabel as nib
import numpy as np
import subprocess

def usage():
    print(r'''
    dr_cifti.py melodic_IC.dscalar.nii  output_dir ts1 ts2 ts3
    ''')

def parser():
    parser = argparse.ArgumentParser(description="dual regression for cifti data using fsl dual_regression")
    parser.add_argument('melodic_ic', help='melodic_IC.dscalar.nii')
    parser.add_argument('output_dir', help='output_dir')
    parser.add_argument('ts_files', nargs='+', help='subject ts files')
    return parser.parse_args()

def convert_cifti_to_nifti(cifti_file_path, nifti_file_path):
    cifti_file = nib.load(cifti_file_path)
    cifti_data = cifti_file.get_fdata()
    nii_img = nib.Nifti1Image(cifti_data.T[:, np.newaxis, np.newaxis, :], np.eye(4))
    nib.save(nii_img, nifti_file_path)

def convert_nifti_to_cifti(nifti_path, cifti_ref_path, cifti_out_path):
    nii_header = nib.load(nifti_path)
    data = nii_header.get_fdata()
    data = data[:, 0, 0, :].T

    cifti_img = nib.load(cifti_ref_path)
    new_cifti_img = nib.Cifti2Image(data, header=cifti_img.header, nifti_header=cifti_img.nifti_header)
    nib.save(new_cifti_img, cifti_out_path)


def dr_cifti(melodic_ic, output_dir, ts_files):
    convert_cifti_to_nifti(melodic_ic, melodic_ic.replace('dscalar.nii', 'nii.gz'))
    for ts_file in ts_files:
        convert_cifti_to_nifti(ts_file, ts_file.replace('dtseries.nii', 'nii.gz'))

    command = [
        'dual_regression',
        melodic_ic.replace('dscalar.nii', 'nii.gz'),
        '1', '-1', '1',
        output_dir
    ]
    command += [ts_file.replace('dtseries.nii', 'nii.gz') for ts_file in ts_files]

    print(command)
    subprocess.run(command)

def main():
    args = parser()
    dr_cifti(args.melodic_ic, args.output_dir, args.ts_files)


if __name__ == '__main__':
    main()
