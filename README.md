# DSG-2016

This repo contains all the code for Data Science Game 2016 Preliminary Challenge
https://inclass.kaggle.com/c/data-science-game-2016-online-selection

Problem Description: Given a dataset of satellite images of roofs, the task was to predict the orientation of the roofs into 4 different categories- North-South, East-West, Flat or Other

Methodology: We used an ensemble of pre-trained Convolutonal Neural Networks - GoogLenet and VGG16 using Keras. 
Our model was able to achieve a prediction accuracy of 83.7%

Description: 

### Data Extraction: convert images to numpy array files

Code: '1_teamtabs_creating_numpy_arrays_from_images_final.py'

* We scaled images to 3 sizes: (224x 224), (192 x 192) and (256 x 256) and converted them into numpy arrays
* The original train data comprising of 800 images was split into train and validation sets using an 80:20 split ratio.

### Transfer Learning on VGGNet

Code:  '2_teamtabs_vgg_training_final.py'

* We finetuned a pretrained VGG-16 with the first 20 layers frozen. 
* We then applied data augmentation techniques like flip, rotate etc while training

References: 
[1] Very Deep Convolutional Networks for Large-Scale Image Recognition, K. Simonyan, A. Zisserman- https://arxiv.org/pdf/1409.1556.pdf 

[2] Keras implementation: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3 

### Transfer Learning on GoogLeNet

Code: '3_teamtabs_googlenet_training_final.py’

* We finetuned a modified version of the pretrained GoogLeNet (Inception V1)
* We applied data augmentation 
* The 2 auxiliary classifiers of the GoogleNet were removed to speed-up the tuning process

References:
[1] Going deeper with convolutions. Szegedy, Christian, et al.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
[2] Keras implementation: https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14

### Making Predictions using the two models

Codes: '4_teamtabs_vgg_predict_final.py' & '5_teamtabs_googlenet_predict_final.py'
* We apply data augmentation to the test images and take average prediction to get the final label.

### Create Ensembles

Code: '6_teamtabs_ensembles_final.r'
 
Ensembles are created by averaging the class-probabilities obtained from various models as follows:







