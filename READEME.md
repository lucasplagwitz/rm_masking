### Rat-Mouse Brain Masking with U-Net

This project demonstrates the application of the U-Net architecture for masking mouse and rat brains. 
It focuses on:
1. the possibility/necessity of transfer-learning from mouse to rat,
2. an analysis of required training data size, and
3. providing a Nifti in-out brain masker.

The code and net structure is inspired by a [PyTorch U-Net implementation](https://github.com/milesial/Pytorch-UNet).
The training process of the model will be added in one of the next versions.

### Installation

- Download the repository and navigate to the source folder.
- Activate the Python environment if available.
- Install all dependencies with ```pip install -r requirements.txt```.
- Start the prediction with ```python predict.py -i /path/to/input_niis/ -o /path/to/output_niis/```

### References
[1] [O. Ronneberger, P. Fischer, T. Brox: U-Net: Convolutional Networks for Biomedical Image Segmentation. 
Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015 pp 234-241.](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)