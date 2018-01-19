import keras 
import numpy as np

import sys, os

# This should be removed and put into utils or something
from skimage import transform 

PROJ_DIR = "/Users/bdhammel/Documents/insight/harvesting/"

if PROJ_DIR not in sys.path:
    sys.path.append(PROJ_DIR)

from pipeline.train_and_test import import_uc_merced


class PatchIdentifier:

    def __init__(self, path=None):
        """load the model by importing a pre-existing model or initializing a 
        new one

        Args
        ----
        path (str) : path to a saved model with .h5 extension 
        """

        if path:
            self._model = keras.models.load_model(path)
        else:
            self._model = self._init_model()


    def _init_model(self):
        """Initialize a new model 
        """

        # Used the convolutional layers from inception for feature extraction 
        # and speeding up the training process by leveraging pre-trained weights
        base_model = keras.applications.inception_v3.InceptionV3(
                weights='imagenet', 
                include_top=False
        )

        # Construct custom classifier on top of inception 
        h = base_model.output
        h = keras.layers.GlobalAveragePooling2D()(h)
        h = keras.layers.Dense(1024, activation='relu')(h)
        h = keras.layers.Dropout(0.5)(h)
        predictions = keras.layers.Dense(5, activation='softmax')(h)

        return keras.models.Model(inputs=base_model.input, outputs=predictions)

    
    def train(self, Xtrain, Ytrain, epochs=5, batch_size=32, fix_layers=None):
        """Train the model

        Args
        ----
        Xtrain
        Ytrain
        epochs
        batch_size
        fix_layer (int, int) : layers not to train, of the form 
            (start_layer, end_layer)
        """

        # Sanity check
        if not self._images_are_ok(Xtrain):
            raise Exception

        if not self._labels_are_ok(Ytrain):
            raise Exception

        # Turn off training of weights, used during transfer learning
        if fix_layers:
            start_layer, end_layer = fix_layers
            for layer in np.asarray(self._model.layers)[start_layer:end_layer]:
                layer.trainable = False

        self._model.compile(
                optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy']
        )

        self._model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size)


    def save(self, path="./saved_models/ucmerced.h5"):
        self._model.save(path)


    def probabilities(self, X):
        """
        """
        return self._model.predict(X)

    def predict(self, X):
        """
        """
        return np.argmax(self._model.predict(X), 1)


    def evaluate(self, X, Y):
        """
        """

        # Sanity check
        if not self._images_are_ok(X):
            raise Exception

        if not self._labels_are_ok(Y):
            raise Exception

        return self._model.evaluate(X, Y)

    def _images_are_ok(self, images):
        """
        TODO: actually do this function 
        """
        return True


    def _labels_are_ok(self, labels):
        """
        TODO: actually do this function 
        """
        return True


def _train_on_uc_merced():
    """
    """

    # Load in data
    data_path = "/Users/bdhammel/Documents/insight/data/UCMerced_LandUse/"

    dataset = import_uc_merced.DataSet(data_path)
    Xtrain, Ytrain = dataset.get_train_data()
    Xtest, Ytest = dataset.get_test_data()

    one_hot_ytrain = keras.utils.to_categorical(Ytrain-1, 5)
    one_hot_ytest = keras.utils.to_categorical(Ytest-1, 5)

    ## Augment data

    model = PatchIdentifier()

    # Only train the top of the model, used the features from Resnet
    model.train(Xtrain, one_hot_ytrain, epochs=5, fix_layers=(0,311))

    print(model.evaluate(Xtest, one_hot_ytest))

    # Save all of your hard work 
    model.save("saves_models/merced.h5")


if __name__ == "__main__":
    #_train_on_uc_merced()
    pass

