# Deep Learning Report 6
This report will focus on developing and proposing my final project for this class. At the time of writing this I have a core idea, and a few specific thoughts, but I do not have all the details figured out. In the first portion of this report I will document my process towards achieving a clearer path forward. I will then consolidate  my investigation into a more formal proposal.

In trying to formulate a project I have tried to think about the larger computer vision topics we have discussed in this class, and which of these I find interesting or presents room for extension. As well, I have wanted to take a step beyond what we have discussed and look for topics that incorporate something new. In exploring potential ideas I stumbled across a paper that I think gets at each of these points.

[An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds](https://arxiv.org/abs/2007.12392) is a very recent paper (ECCV 2020 in August) from Google research. Using the Waymo Open Dataset the authors achieve state-of-the-art performance in object detection from 3D point cloud data. The key piece of the authors work is in using an LSTM recurrent network inside their model to incorporate data from past frames. A deeper discussion of the methods and results will be given in a Jupyter Notebook as specified below.

This paper incorporates two new topics from outside of class -- recurrent networks, 3D convolutions, and graph convolutions -- while staying within a central topic I am familiar with and interested in from class. Specifically, this is object detection and semantic segmentation, and as an example, the paper uses a U-Net backbone which is a model explicitly discussed in class. This gives me a solid understanding of integral ideas going in, but leaves me some room to learn new things as well. Furthermore, our continuing discussion of object detection has given me a host of thoughts about how to extend the author's work.

As a final comment on the intrigue of this paper I would note that I am interested and involved in robotics. This is a field for which LiDAR data provides a host of oportunities, so obtaining efficient and effective means for handling this data is imperative.  

## 3D LiDAR Object Detection
In this Jupyter notebook I will be sumarizing the paper discussed above. Sumarizing is really a poor word as my goal is to delve into the ideas, methods, and results of the paper to inform my own project. In addition to discussing the paper itself, this will also include an investigation into the dataset being used and any core tools previously unknown to me.

## Recurrent 3D LiDAR Object Detection
Here I present my formal project proposal. The proposal is informed by my research conducted in the previously mentioned Notebook.

## Report Reflection
In my reading, summary, and subsequent investigation of the 3D LiDAR object detection paper I was exposed to a number of new topics in deep learning. Notably, sparse 3D convolutions and graph convolutions are two rather new and exciting techniques. As well, the Waymo Open Dataset seems like an incredible resource. Beyond this it was motivating to read a technical paper that used many of the topics we have covered in class or I have covered in my reports. Without my learning this semester I would not have been able to understand their process, but instead I am excited to extend it. Overall, in this report I have developed a beginning understanding of some new topics, and dived into a previously unexplored (for me) type of data. Furthermore, I have formulated a solid idea of my final project with many interesting goals. Much of my work this week was in researching and thinking, and I was disappointed not to be able to write up as much or delve as deep into some of the new topics.

# NOTE
The Waymo data is very large and so I ended up not includig it in the repo.

