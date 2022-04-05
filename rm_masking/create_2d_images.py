import os
import nibabel as nib
import numpy as np
import matplotlib.image
import cv2


def create_2d_images(input_img: str = "./../data/nii/mouse/img/",
                     input_mask: str = "./../data/nii/mouse/mask/",
                     output_img: str = "./../data/png/mouse/img/",
                     output_mask: str = "./../data/png/mouse/mask/",
                     input_img_prfix = "_mean",
                     input_mask_prfix = "_3D",
                     split_test: bool = True):
    """
    split_test creates another folder for test_images (output_img -> /../../mouse_test)

    """
    counter = 0
    all = len([x for x in os.listdir(input_img) if x.split(".")[-1] == "nii"])
    for f in os.listdir(input_img):
        a = f[:len(f)-4-len(input_img_prfix)]
        if not f.split(".")[-1] == "nii":
            print(f"Ignore file {f} - no .nii file")
            continue
        if not os.path.exists(os.path.join(input_mask, a+input_mask_prfix+".nii")):
            print("No mask found.")
            continue

        try:
            el_img = nib.load(os.path.join(input_img, f)).get_fdata()
            el_mask = nib.load(os.path.join(input_mask, a+input_mask_prfix+".nii")).get_fdata()
            if len(el_img.shape) != 3 or len(el_mask.shape) != 3:
                raise ValueError("Need 3d input NIIs.")
        except:
            continue
        max_axis = np.max(np.shape(el_img)[:-1])
        img_mean = np.zeros((max_axis, max_axis, el_img.shape[-1]))
        img_mask = np.zeros((max_axis, max_axis, el_img.shape[-1]))
        img_mean[:el_img.shape[0], :el_img.shape[1], :] = el_img
        img_mask[:el_img.shape[0], :el_img.shape[1], :] = el_mask

        for i in range(el_img.shape[2]):
            matplotlib.image.imsave(os.path.join(output_img, a + f'_{i}.png'),
                                    cv2.resize(img_mean[:, :, i], dsize=(80, 80),
                                               interpolation=cv2.INTER_CUBIC)
                                    )

            matplotlib.image.imsave(os.path.join(output_mask, a + f'_{i}.png'),
                                    cv2.resize((img_mask[:, :, i] > 0).astype(int),
                                               dsize=(80, 80), interpolation=cv2.INTER_NEAREST_EXACT)
                                    )

        counter += 1

        if split_test and counter == all - 2:
            output_img = os.path.join(output_img, "../../mouse_test/img")
            output_mask = os.path.join(output_mask, "../../mouse_test/mask")


if __name__ == '__main__':
    create_2d_images()

    #create_2d_images(input_img = "./../data/nii/rat/img/",
    #                 input_mask = "./../data/nii/rat/mask/",
    #                 output_img = "./../data/png/rat/img/",
    #                 output_mask = "./../data/png/rat/mask/",
    #                 input_img_prfix = "",
    #                 input_mask_prfix = "_bm")
