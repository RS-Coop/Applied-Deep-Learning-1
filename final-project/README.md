# Final Project: Sparse Recurrent CNNs for 3D Point Cloud Semantic Segmentation

In this projet I examine sparse 3D CNNs equipped with a recurrent module for the task of semantic segmentation and object detection in sequential LiDAR data. Note this project is still a work in progress.

## References and Contributions
I want to give a huge shoutout to Zhijian Liu and Haotian Tang at the MIT Han Lab for their work on [*torchsparse*](https://github.com/mit-han-lab/torchsparse) and [*Efficient Methods for 3D Deep Learning*](https://github.com/mit-han-lab/e3d). Their work was very informative and inspirational, and their model provides the backbone for my project.

The idea for the use of recurrent layers is partly inspired by the work of Rui Huang and other Google researchers in their paper [*An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds*](https://arxiv.org/abs/2007.12392)

Further references can be found in the paper in the *report* folder.

## Structure

Some of the files and structure are quite self-explanatory, so I will only include the most important one.

- core: All of the network code
	- data: Datasets and associated loading functionality, note that the datasets are not provided here
	- modules: Required pieces of e3d from the MIT Han Lab
	- sparse\_rnn.py: Custom rnn modules with sparse functionality 

- testing: This folder contains a few notebooks that consist of my testing random stuff out. There isn't much structure there so be warned.

- report: This folder contains all my official report materials. Specifically, there is a pdf (*main*) and a notebook which produces the plots.

- main.py: You can run this file to try training and then evaluating a model usng the nuScenes dataset. Parameters can be changed inside.
