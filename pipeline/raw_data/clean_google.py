import numpy as np
import os
from PIL import Image, ImageDraw
import skimage.io as skio
import requests
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt


from pipeline import utils as pipe_utils


MY_API_KEY = os.environ.get('GOOGLE_MAP_KEY', None)
STATIC_MAP_BASE_URL = 'https://maps.googleapis.com/maps/api/staticmap?center={xcenter},{ycenter}&zoom={zoom}&size=400x400&maptype=satellite&key={api_key}'


def fetch_image(params):
    if MY_API_KEY is None:
        raise Exception("You need to set your GOOGLE_MAP_KEY as an enviroment variable")
    url = STATIC_MAP_BASE_URL.format(**params) 

    with NamedTemporaryFile() as f:
        f.write(requests.get(url).content)
        f.seek(0)
        img_data = skio.imread(f.name)

    img = pipe_utils.SatelliteImage(img_data)
    
    return img 


def preprocess_gogle(img):
    pass
    


if __name__ == '__main__':
    params = {
        'xcenter':40.714728,
        'ycenter':-73.998672,
        'zoom':19,
        'api_key':MY_API_KEY
    }

    img = fetch_image(params)
    img.show()


