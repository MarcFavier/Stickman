# Stickman
Deep Learning project for pose estimation on simplified stickman dataset

Training and test images are toys example, you can take a look in the Dataset/Test folder.
Training images are generated with code in src/generator folder.

Two architectures have been studied in this project : a VGG and another network strongly inspired from Mobile Net V2
(https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)
Those networks are built in the src/model folder. The files heads.py and backbones.py combine modules defined in blocks.py.


The creation of the inspired Mobile-Net network, its training and evaluation have been the only part in the project completely coded by myself.
Dataset generation and VGG network have been given as a starting point in this project.


