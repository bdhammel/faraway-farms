import numpy as np
import os
from PIL import Image, ImageDraw
import skimage.io as skio
from skimage.color import grey2rgb

import requests
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt


from pipeline import utils as pipe_utils


MY_API_KEY = os.environ.get('GOOGLE_MAP_KEY', None)
STATIC_MAP_BASE_URL = 'https://maps.googleapis.com/maps/api/staticmap?center={xcenter},{ycenter}&zoom={zoom}&size=400x400&maptype=satellite&key={api_key}'


def fetch_image(xcenter, ycenter, zoom, api_key):
    """Download a satellite image from google

    Args
    -----
    """


    url = STATIC_MAP_BASE_URL.format(
            xcenter=xcenter,
            ycenter=ycenter,
            zoom=zoom,
            api_key=api_key
    ) 

    with NamedTemporaryFile() as f:
        f.write(requests.get(url).content)
        f.seek(0)
        img_data = skio.imread(f.name)

    img_data = __process_google(img_data)
    img = pipe_utils.SatelliteImage(img_data)
    
    return img 


def __process_google(img):

    data = img.data

    # drop alpha channel
    # make sure RGB channels
    if np.ndim(data.shape) == 3:
        if data.shape[-1] == 4:
            data = data[...,:3]
    else:
        data = grey2rgb(data)
    
    # make sure data is [0, 255]
    if data.max() < 1 and data.min() > 0:
        data *= 255
        data = data.astype(np.uint8)

    return data


if __name__ == '__main__':

    if MY_API_KEY is None:
        raise Exception("You need to set your GOOGLE_MAP_KEY as an enviroment variable")

    params = {
        'xcenter':39.13851,
        'ycenter':-122.24515,
        'zoom':19,
        'api_key':MY_API_KEY
    }

    img = fetch_image(**params)
    img.show()


