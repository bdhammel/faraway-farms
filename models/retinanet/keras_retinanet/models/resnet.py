"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings

import keras
from ..models import retinanet

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)


def resnet_retinanet(num_classes, backbone=50, weights='imagenet', **kwargs):

    inputs = keras.layers.Input(shape=(400, 400, 3))

    resnet = keras.applications.resnet50.ResNet50(
            input_tensor=inputs,
            include_top=False,
            weights=weights
    )

    bottleneck_layers = ['activation_22', 'activation_40', 'activation_49'] 

    # create the full model
    model = retinanet.retinanet_bbox(
            inputs=inputs, 
            num_classes=num_classes, 
            backbone=resnet, 
            bottleneck_layers=bottleneck_layers,
            **kwargs)

    return model


def resnet50_retinanet(num_classes, weights='imagenet', skip_mismatch=True, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone=50, inputs=inputs, weights=weights, skip_mismatch=skip_mismatch, **kwargs)


def resnet101_retinanet(num_classes, inputs=None, weights='imagenet', skip_mismatch=True, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone=101, inputs=inputs, weights=weights, skip_mismatch=skip_mismatch, **kwargs)


def resnet152_retinanet(num_classes, inputs=None, weights='imagenet', skip_mismatch=True, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone=152, inputs=inputs, weights=weights, skip_mismatch=skip_mismatch, **kwargs)


def ResNet50RetinaNet(num_classes, skip_mismatch=True, **kwargs):
    warnings.warn("ResNet50RetinaNet is replaced by resnet50_retinanet and will be removed in a future release.")
    return resnet50_retinanet(num_classes, *args, skip_mismatch=skip_mismatch, **kwargs)


def ResNet101RetinaNet(inputs, num_classes, skip_mismatch=True, **kwargs):
    warnings.warn("ResNet101RetinaNet is replaced by resnet101_retinanet and will be removed in a future release.")
    return resnet101_retinanet(num_classes, inputs, *args, skip_mismatch=skip_mismatch, **kwargs)


def ResNet152RetinaNet(inputs, num_classes, skip_mismatch=True, **kwargs):
    warnings.warn("ResNet152RetinaNet is replaced by resnet152_retinanet and will be removed in a future release.")
    return resnet152_retinanet(num_classes, inputs, *args, skip_mismatch=skip_mismatch, **kwargs)
