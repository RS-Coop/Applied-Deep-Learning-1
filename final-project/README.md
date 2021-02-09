# Sparse Recurrent CNNs for 3D Point Cloud Semantic Segmentation
In this project I examine sparse 3D CNNs equipped with a recurrent module for the task of semantic segmentation and object detection in sequential LiDAR data. Note this project is still a work in progress.

## References and Inspiration
In addition to a multitude of specific citations, there were two main sources of inspiration and reference for my work.

Zhijian Liu and Haotian Tang at the MIT Han Lab with their work on [*torchsparse*](https://github.com/mit-han-lab/torchsparse) and [*Efficient Methods for 3D Deep Learning*](https://github.com/mit-han-lab/e3d). Their work was very informative, and their model provides the backbone for my project.

The idea for the use of recurrent layers is partly inspired by the work of Rui Huang and other Google researchers in their paper [*An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds*](https://arxiv.org/abs/2007.12392). Their techniques, methods, and results all provided me with valuable insight.

Further references can be found in the paper in the *report* folder.

## Structure
Much of the structure and some of the files are quite self-explanatory, so I will discuss only that which is most important.

- *core*: All of the nueral network code
	- *data*: Datasets and associated loading functionality. Note that the actual data is much too large to provide here. Details (including download information) on the nuscenes dataset can be found [here](https://www.nuscenes.org/nuscenes), and the SemanticKITTI dataset [here](http://www.semantic-kitti.org/).
	- modules: Required pieces of e3d from the MIT Han Lab.
	- sparse\_rnn.py: Custom rnn modules with sparse functionality. Yet to be implemented.

- **.ipynb*: These are a number of Jupyter Notebooks which were used for testing various components -- they can mostly be ignored.

- *report*: This folder contains all my official report materials. Specifically, there is a pdf (*main*) and a notebook which produces the plots.

- *main.py*: You can run this file to try training and then evaluating a model usng the nuScenes or SemanticKITTI datasets. Parameters can be changed inside.
