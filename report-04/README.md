# Report 4

In this report I have continued my quest in learning TensorFlow and understanding more about implementing and working with neural networks. I have completed the second week of the [Deep Learning for Robotics class](https://software.intel.com/content/www/us/en/develop/training/course-deep-learning-robotics.html), and a number of TensorFlow computer vision tutorials. Unfortunately I spent a lot of time this week dealing with various bugs and other software development issues not always related to deep learning. For example, trying to get dropout working in forward passes, getting the simulations working in Jupyter, and diving into TensorFlow material. This kept me back from doing as much as I would have liked, but it was definitely a learning experience.

# Code

## Computer Vision.ipynb

This file is a Jupyter Notebook with two TensorFlow tutorials on computer vision.

Requirements:
- TensorFlow
- matplotlib
- seaborn
- numpy
- pillow
- tf-nightly (if you want to create the datasets)


## dl-4-robots

This file contains the materials from week 2 of the Deep Learning for Robotics. All that needs to be looked at is the Jupyter Notebook **Deep Learning for Robotics - Inverse Kinematics.ipynb**. This contains the two exercises for the class and all the simulations. The rest of the main content is inside the **env_sim** folder where the simulations, network, training, training data, and anything else are defined. I have removed some extraneous files that were originally included by the class.

Requirements:
- TensorFlow
- PyTorch
- numpy
- scipy
- pygame
- pylab
- glob
- seaborn
