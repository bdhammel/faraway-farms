# Faraway Farms

Counting sheep without falling asleep


#### Abstract 
Identify, locate, and count objects of interest given a Satallite image.  

In rural and hard to reach regions of the world, farmers struggle to qualify for financial loans. A large factor is do to the difficulty in having a credit agency visit their land to itemize assets and evaluate creditworthiness. This project investigates the feasibility of using a convolutional neural network to itemize assets through remote sensing. That is, an object detection model is employed to analyze satellite images.

Do to the limited availability of labeled farm equipment in satellite images, I explore a proof-of-principle model using readily available datasets in the public domain. 

[Google Slides](http://bit.ly/faraway-farms)

## Examples

Examples on using the scripts contained in this package can be found in the folder `workspace`.

## Installation

The contained file, `requirements.txt`, contains the python package dependance for this projects. Installation of these can be performed via 

~~~
pip install -r requirements.txt
~~~ 

## Technical Discussion

Two models are implemented in the projects. A classification model, witch labels a patch of an image with the most probable class. And an object detection model, which locates, counts and identifies specific objects of interest.

### Patch Identification

The model employed for patch identification uses some of the convolutional layers from Inception V3 for feature extraction, and two dense layers as a custom classifier. 

Inception V3 was chosen because it is the most accurate model per number of parameters. In theory, fewer parameters will reduce the likely hood of overfitting. 

In addition to this, the convolutional layers after the 6th concatenation node are dropped. This was done for the same reason.


### Object Detection 

RetinaNet was selected as the object detection model because of its success in identifying densely packed objects [1]. This is attributed to the implementation of a weighted loss function, dubbed focal loss, which addresses the issue of class imbalance between objects of interest and the image background.


[1] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)


## Tests

Tests can be executed with the following command:

~~~
python -m tests/pipeline/test_pipeline.py
~~~

## Next Steps