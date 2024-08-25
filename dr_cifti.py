#!/opt/hky-venv/bin/python
# Author: 赩林, xilin0x7f@163.com
import argparse
import os
import glob
import nibabel as nib
import numpy as np
import subprocess
from joblib import Parallel, delayed

def usage():
    print(r'''
    dr_cifti.py melodic_IC.dscalar.nii  output_dir ts1 ts2 ts3
    ''')

def parser():
    parser = argparse.ArgumentParser(description="dual regression for cifti data using fsl dual_regression")
    parser.add_argument('melodic_ic', help='melodic_IC.dscalar.nii')
    parser.add_argument('output_dir', help='output_dir')
    parser.add_argument('ts_files', nargs='+', help='subject ts files')
    parser.add_argument('-j', '--job', required=False, default=1)
    return parser.parse_args()


# noinspection PyTypeChecker
def convert_cifti_to_nifti(cifti_file_path, nifti_file_path, lazy=True):
    if os.path.exists(nifti_file_path) and lazy:
        return

    cifti_file = nib.load(cifti_file_path)
    cifti_data = cifti_file.get_fdata()
    data4save = cifti_data.T

    data4save = np.vstack((
        data4save,
        np.zeros(shape=(int(np.ceil(data4save.shape[0]/1000) * 1000 - data4save.shape[0]), data4save.shape[1]))
    ))

    data4save = data4save.reshape(1000, int(data4save.shape[0]/1000), 1, data4save.shape[1])

    nii_img = nib.Nifti1Image(data4save, np.eye(4))
    nib.save(nii_img, nifti_file_path)


# noinspection PyProtectedMember,PyUnresolvedReferences
def convert_nifti_to_cifti(nifti_path, cifti_ref_path, cifti_out_path):
    cifti_img = nib.load(cifti_ref_path)

    nii_header = nib.load(nifti_path)
    data = nii_header.get_fdata()

    if data.ndim < 4:
        data = data.reshape(-1, 1)
    else:
        data = data.reshape(-1, data.shape[-1])

    data = data[:cifti_img.get_fdata().shape[1], :].T
    cifti_header = cifti_img.header.copy()
    cifti_header.matrix[0]._maps = [cifti_header.matrix[0]._maps[0] for _ in range(data.shape[0])]
    new_cifti_img = nib.Cifti2Image(data, header=cifti_header, nifti_header=cifti_img.nifti_header)
    nib.save(new_cifti_img, cifti_out_path)


def dr_cifti(melodic_ic, output_dir, ts_files, n_job):
    convert_cifti_to_nifti(melodic_ic, melodic_ic.replace('dscalar.nii', 'nii.gz'))
    tasks = (
        delayed(convert_cifti_to_nifti)(ts_file, ts_file.replace('dtseries.nii', 'nii.gz')) for ts_file in ts_files
    )
    Parallel(n_jobs=n_job)(tasks)

    command = [
        'dual_regression',
        melodic_ic.replace('dscalar.nii', 'nii.gz'),
        '1', '-1', '1',
        output_dir
    ]
    command += [ts_file.replace('dtseries.nii', 'nii.gz') for ts_file in ts_files]

    print(command)
    subprocess.run(command)

    nii_files_path = glob.glob(os.path.join(output_dir, '*nii.gz'))

    tasks = (
        delayed(convert_nifti_to_cifti)
        (nii_file_path, melodic_ic, nii_file_path.replace('nii.gz', 'dscalar.nii'))
        for nii_file_path in nii_files_path
    )

    Parallel(n_jobs=n_job)(tasks)

def main():
    args = parser()
    dr_cifti(args.melodic_ic, args.output_dir, args.ts_files, args.job)


if __name__ == '__main__':
    main()
