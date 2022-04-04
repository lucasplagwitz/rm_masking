import argparse
import logging
import os
import numpy as np
import nibabel as nib
import scipy.ndimage
import torch
import cv2

try:
    from rm_masking.unet import UNet
    package_dir = os.path.dirname(__file__)
except ModuleNotFoundError:
    # compatibility with direct "python predict.py ..."
    from unet import UNet

logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./models/only_mouse.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Folder of input images',
                        default="/path/to/nifti_input")
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        default="/path/to/nifti_output", help='Folder of output images')
    parser.add_argument('--threshold', '-t', metavar='THRESHOLD', nargs='+',
                        default=.5, help='Value to distinguish between information and background.')
    return parser.parse_args()


def run(in_files, out_files, model=None, threshold=0.5):
    if model is None:
        model = os.path.join(package_dir, 'models/only_mouse.pth')

    if isinstance(in_files, list):
        in_files = in_files[0]

    if isinstance(out_files, list):
        out_files = out_files[0]

    if in_files == out_files:
        raise ValueError("In- and outputfolder should not be the same.")
    if not os.path.exists(in_files):
        raise ValueError(f"Input-Folder {in_files} not found.")
    if not os.path.exists(out_files):
        raise ValueError(f"Output-Folder {out_files} not found.")

    net = UNet(n_channels=1, n_classes=2)

    device = "cpu"
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(os.listdir(in_files)):

        try:
            nii = nib.load(os.path.join(in_files, filename))
        except:
            logging.warning(f"File {filename} was ignored. Nibabel cannot import that file."
                            f"Please only use Niftis.")
            continue

        nii_val = nii.get_fdata()
        if len(nii_val.shape) != 3:
            logging.warning(f"Please only use 3d niftis got size {nii_val.shape}")

        nifti_list = []
        mask_list = []
        for dim in range(np.shape(nii_val)[2]):
            # scale to (80, 80)
            img = nii_val[:, :, dim] / 255  # scaling has to be removed for future models
            original_size1 = np.shape(img)
            output = np.zeros((np.max(img.shape), np.max(img.shape)))
            output[:np.shape(img)[0], :np.shape(img)[1]] = img
            original_size2 = np.shape(output)
            output_scaled = cv2.resize(output, dsize=(80, 80),
                                       interpolation=cv2.INTER_CUBIC)
            output_scaled = torch.from_numpy(output_scaled)

            # predict
            mask_scaled = net(output_scaled.reshape((1, 1, 80, 80)).to(device=device).float()).cpu().detach().numpy()

            # scale up to original source
            mask_scaled = np.argmax(mask_scaled[0, :], axis=0)
            mask = cv2.resize(mask_scaled.astype('float32'),
                              dsize=original_size2, interpolation=cv2.INTER_CUBIC)
            mask = mask[:original_size1[0], :original_size1[1]]
            mask = (mask > threshold).astype(int)
            mask = scipy.ndimage.binary_fill_holes(mask).astype(int)
            img *= 255  # scaling has to be removed for future models

            # apply mask to image
            nifti_list.append((mask * img).reshape((original_size1[0], original_size1[1], 1)).astype(nii.dataobj.dtype))
            mask_list.append(mask.reshape((original_size1[0], original_size1[1], 1)).astype(nii.dataobj.dtype))

        nii_out = nib.Nifti1Image(np.concatenate(nifti_list, axis=2),
                                  affine=nii.affine)
        mask_out = nib.Nifti1Image(np.concatenate(mask_list, axis=2),
                                   affine=nii.affine)

        nib.save(mask_out, os.path.join(out_files, filename.split(".")[0] + "_mask"))
        nib.save(nii_out, os.path.join(out_files, filename.split(".")[0] + "_fmri"))
        logging.info(f"Successful masking of {filename}")


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = args.output
    threshold = args.threshold
    model = args.model
    run(in_files, out_files, model, threshold)
