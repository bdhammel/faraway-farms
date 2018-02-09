# Faraway Farms

Counting sheep without falling asleep


#### Abstract 

Identify, locate, and count objects of interest given a satellite image.  

In rural and hard to reach regions of the world, farmers struggle to qualify for financial loans. A large factor is do to the difficulty in having a credit agency visit their land to itemize assets and evaluate creditworthiness. This project investigates the feasibility of using a convolutional neural network to itemize assets through remote sensing.

Do to the limited availability of labeled farm equipment in satellite images, I explore a proof-of-principle model using readily available datasets in the public domain. 

[Google Slides](http://bit.ly/faraway-farms)

## Examples

Examples on using the scripts contained in this package can be found in the folder `workspace`.

## Installation

The contained file, `requirements.txt`, contains the python package dependancies for this projects. Installation of these can be performed via 

~~~
pip install -r requirements.txt
~~~ 

## Data and weights

Please see the folder [datasets](./datasets) for a description of the data used in this project.


## Technical Discussion

Two models are implemented in the projects. A classification model, witch labels a subsection, patch, of an image with the most probable class. And an object detection model, which locates, counts and identifies specific objects of interest.

### Patch Identification

The model employed for patch identification uses some of the convolutional layers from Inception V3 for feature extraction, and two dense layers as a custom classifier. 

Inception V3 was chosen because it is the most accurate model per number of parameters. In theory, fewer parameters will reduce the likely hood of overfitting. 

In addition to this, the convolutional layers after the 6th concatenation node are dropped. This was done to further reduce the number of parameters to avoid overfitting.

### Object Detection 

RetinaNet was selected as the object detection model because of its success in identifying densely packed objects [1]. This is attributed to the implementation of a weighted loss function, dubbed focal loss, which addresses the issue of class imbalance between objects-of-interest and the image background.

A modified version of the open-source package built by the authors of [1] is included in this repo. Because of the rapidly evolving capabilities of the open-source repo, **I recommend using an unaltered version of [2] and not the one contained here.**

[1] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

[2] [RetinaNet](https://github.com/fizyr/keras-retinanet)

## Tests

Tests can be executed with the following command:

~~~
python -m tests/pipeline/test_pipeline.py
~~~

These perform a sanity check on the handling and cleaning of raw data and the construction of a usable dataset for these models.


## Technical Notes

Functions to accomplish the following actions are stored in the `pipeline` directory. 

### Preprocessing images

 - Images are loaded into the program and converted to 8 bit, i.e. Pixel range between [0,255]
 - Only 3 color channel (RGB) images are valid. If the image has a 4th color channel (alpha layer), this channel is dropped. If the image has only one channel (B&W), the image is stacked 3 times.
 - Images need to be of the form "channel-last." i.e. the shape of an image needs to be (width, height, color channels). This is oppose to some formats which are (color channels, width, height).

**Example**

The following script imports and image and cleans it. `check_data` performs a test to make sure everything is in the expected format. 

```python
import pipeline.raw_data.utils as raw_utils
import pipeline.utils as pipe_utils

raw_image = raw_utils.read_raw_image('/path/to/image.jpg')
clean_image = pipe_utils.image_save_preprocessor(raw_image)
pipe_utils.check_data(clean_image, raise_exception=True)
```

Preprocessing and cleaning functions specific to a dataset are located in the files `pipeline/raw_data/clean_<dataset>.py`. These handle more specific actions. For example, the processing of bounding boxes in object detection images. These files should be well commented; however, I will try to add a description of them here.


### Processing images

Before an image is feed into a model (patch identification or object detection) more adjustments are made:

 - The order of the color channel is reversed, from RGB -> BGR. Honestly, I don't know why. I do this to be consistent with the authors of RetinaNet.
 - Images are normalized on a per channel basis, by subtracting the mean (determined from ImageNet images)
 - Images need to be of a specific shape: `(200,200,3)` for the patch identifier, and `(400,400,3)` for the object detector.


**Example**

The function `preprocess_image_for_model` and `as_batch` will perform the above steps to prepare the image for the model.

```python
batch = pipe_utils.as_batch(clean_image, shape=(200,200,3))
Xinput = pipe_utils.preprocess_image_for_model(batch)
```