"""CNN for doing classification on patches of an image

Model architecture used the convolutional layers from Inception V3 for feature
extraction. A custom dense network is then attached to the top of the network 
for customized object classification.

Model predicts 6 classes which correspond to PATCH_CLASS_TO_ID defined in 
pipeline.utils

Examples
--------
workspace/patch_training.ipynb : Example of training the patch classifier


Notes
-----
I found that cutting the convolutional layers at "mixed6" allowed the model to 
generalize better and not overfit. The two "Dropout" layers also helped this. 

I did not do an exhaustive search of this parameter space, it is possible that
a more accurate model configuration can be readably attainable.

Currently, only images with pixel dimensions (200, 200, 3) is supported


TODO
----
 - Make CNN identify a dynamic number of classes
 - Add test to insure that number of predictions corresponded the verbose
 labels defined in pipeline.utils
 - Add test to ensure that images and labels being passed to the model are OK

"""
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pipeline imoprt patch_pipeline


class PatchIdentifier:
    """Model class to control training, evaluation, and initialization of 
    the patch identifier CNN

    """

    def __init__(self, path=None, model_fn=None):
        """load the model by importing a pre-existing model or initializing a 
        new one

        Args
        ----
        path (str) : path to a saved model with .h5 extension 
        model_fn (function returning a keras.model instance) 
        """

        if path:
            self._model = keras.models.load_model(path)
        elif model_fn:
            self._model = model_fn()

        self._augmentor = None

    
    def train(self, Xtrain, Ytrain, epochs=5, batch_size=32,
            fix_layers=None, validation_data=None
        ):
        """Train the model

        Args
        ----
        Xtrain  (np.array) : training images of shape (-1, 200, 200, 3)
        Ytrain (np.array) : one-hot-encoded categorical labels. shape (-1, 6)
        epochs (int) : number of epochs to train for 
        batch_size (int) : number of images to run in once batch
        fix_layer (int, int) : layers not to train, of the form 
            (start_layer, end_layer). Used for transfer learning and fine tuning
        """

        # Sanity check
        """ Add this capability:

        if not self._images_are_ok(Xtrain):
            raise Exception

        if not self._labels_are_ok(Ytrain):
            raise Exception
        """

        # Turn off training of weights, used during transfer learning
        if fix_layers:
            fixed_layers = list(range(*fix_layers))
            print("Model has {} layers, freezing layers {}-{}".format(
                len(self._model.layers),
                fix_layers[0],
                fix_layers[1])
            )
        else:
            fixed_layers = []

        for l, layer in enumerate(self._model.layers):
            if fix_layers and l in fixed_layers:
                layer.trainable = False
            else:
                layer.trainable = True

        self._model.compile(
                optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy']
        )

        if not self._augmentor:
            self._model.fit(
                    Xtrain, 
                    Ytrain, 
                    epochs=epochs, 
                    validation_data=validation_data,
                    batch_size=batch_size
            )
        else:
            self._model.fit_generator(
                    self._augmentor.flow(Xtrain, Ytrain, batch_size=batch_size),
                    validation_data=validation_data,
                    epochs=epochs
            )


    def attach_augmentor(self, augmentor):
        """Attached a Keras augmenter to

        Args
        ----
        augmentor (ImageDataGenerator) : and augmentation instance to be called 
        during training
        """
        self._augmentor = augmentor


    def save(self, weight_names):
        """Save the model weights

        Args
        ----
        weight_name (str) : name of the weight file with .h5 extension
        """
        save_dir = "./models/saved_models/ucmerced.h5"
        path = os.path.join(save_dir, weight_names)
        self._model.save(path)


    def probabilities(self, X):
        """Perform inference on an image and return the probabilities of the 
        result 

        Args
        ----
        X (np.array) : a batch of images to perform inference on 

        Returns
        -------
        (list, floats) : [ %class0, %class1, ...]
        """
        X = np.atleast_1d(X)
        return self._model.predict(X) 


    def predict(self, X):
        """Return the prediction of the most likely class

        Args
        ----
        X (np.array) : a batch of images to perform inference on 

        Returns
        -------
        (str) : verbose name of the most likely class
        """
        return np.argmax(self._model.predict(X), 1)


    def evaluate(self, X, Y):
        """Evaluate the accuracy of the model

        Args
        ----
        X (np array) : the image data set to evaluate of shape (-1, 200, 200, 3)
        Y (np array) : one_hot_encoded labels of the image
        """

        # Sanity check
        if not self._images_are_ok(X):
            raise Exception

        if not self._labels_are_ok(Y):
            raise Exception

        return self._model.evaluate(X, Y)


def model_fn():
    """Initialize a new model 
    
    Define the model architecture here:

    Returns
    -------
    keras.models.Model
    """

    # Used the convolutional layers from inception for feature extraction 
    # and speeding up the training process by leveraging pre-trained weights
    base_model = keras.applications.inception_v3.InceptionV3(
            weights='imagenet', 
            include_top=False,
            input_shape=(200,200,3)
    )

    # Construct custom classifier on top of inception 
    h = base_model.get_layer("mixed6").output
    h = keras.layers.GlobalAveragePooling2D()(h)
    h = keras.layers.Dropout(0.5)(h)
    h = keras.layers.Dense(424, activation='relu')(h)
    h = keras.layers.Dropout(0.5)(h)
    predictions = keras.layers.Dense(6, activation='softmax')(h)

    return keras.models.Model(inputs=base_model.input, outputs=predictions)



def train(save_weights=False):
    """Train the patch identification model

    Args
    ----
    save_weights (bool) : save the model once the epochs are finished
    """

    # Load in data
    data_path = "../datasets/whu_rs19/"

    dataset = import_uc_merced.DataSet(data_path)
    Xtrain, Ytrain = dataset.get_train_data()
    Xtest, Ytest = dataset.get_test_data()

    one_hot_ytrain = keras.utils.to_categorical(Ytrain, 6)
    one_hot_ytest = keras.utils.to_categorical(Ytest, 6)

    model = PatchIdentifier()

    auger = ImageDataGenerator(
                rotation_range=90,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True
    )

    model.attach_augmentor(auger)

    # Only train the top of the model, use the features from Inception
    # mixed5 at index 164
    # mixed6 at index 196
    model.train(
            Xtrain, 
            one_hot_ytrain, 
            epochs=20, 
            fix_layers=(0, 195), 
            validation_data=(Xtest[:64], one_hot_ytest[:64])
    )

    print("Finished Training")
    print("Accuracy: {:.2f}%".format(100*model.evaluate(Xtest, one_hot_ytest)[1]))

    # Save all of your hard work 
    if save_weights:
        model.save("./models/saved_models/patch_identifier.h5")

