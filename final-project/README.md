# Final Project: Sparse Recurrent CNNs for 3D Point Cloud Semantic Segmentation

Before we discuss the structure of this repository I will make a note about the state of my project, what I have achieved, and more importantly where I have failed. The goal was to train a model that uses sparse convolutions and some sort of sparse convolutional recurrent module to do semantic segmentation of sequential LiDAR point cloud scenes. I suppose I bit off a bit more than I could chew, and in most of this I have failed. Although, I will note that this is not without lack of trying as I have literally spent the entirety of the last four days on this project. Furthermore, I beleive that with just a little bit more time and effort (which unfortunately I do not have) I could have achieved quite a bit more than it appears I have.

I ended up needing to do a lot more research and 'figuring out' than I expected, and that took up a lot of my time. As well, implementing all of the structure necessary for the project (e.g. the dataset, trainin, etc.) took a lot of time. So where do I stand as it is? Well the structure is implemented, but I am having trouble fine-tuning the network for my dataset. I thought this would be an easy task given the power of the network, but I seem to be wrong or doing something wrong. I was hoping to report results on the pretrained model by mapping the output of that to the labels I use. I couldn't figure out where this mapping lived until just before I write this, and while this is something I am capable of doing I just dont have the time. The sparse recurrent modules are unfinished, but I know what I need to do with them -- it is just a matter of work and time.

None of this is to serve as an excuse, but merely to explain the (what I would consider poor) state of my final project. I plan to continue my work, and hopefully I can share this after doing so. I really think my goals hold promise and are interesting. Anyways I hope you can consider the whole of my project and what I set out to do. Thanks for the great semester, see you next year!

## Shoutouts
I want to give a huge shoutout to Zhijian Liu and Haotian Tang for their work on *torchsparse* and *Efficient Methods for 3D Deep Learning*. Their work was very informative and inspirational, also they answered some key questions on GitHub.

## Structure

### testing
This folder contains a few notebooks that consist of my testing random stuff out. There isn't much structure there so be warned.

### report
This foldre contains all my official report materials. Specifically, there is a pdf *main* of my report and a notebook which produces the plots.

### core
This is the bulk of my work and contains all the structure for my model. Data handling lives in the data/datasets sub folder and if you download the nuScenes mini split you can unzip into there according to their instructions. The modules file is made up almost entirely of code from mit-han-lab/e3d (i.e. SPVNAS) which I need to build their pretrained model.

# main.py
You can run this file to try training and then evaluating a model usng the nuScenes dataset. Parameters can be changed inside.
