# Folders containing training and test data are stored here


These folders are populated using the `pipeline.raw_data` scripts.

Folder hierarchy should be:

```
 - datasets
    - raw_images
        - img1.jpg
        - img2.png
        - img3.tif
    - uc_merced
        - xtest.p
        - xtrain.p
        - ytest.p
        - ytrain.p
```


## Cleaning and processing UC Merced aerial images

Download the raw data from the UC Merced website, link below.

Example execution of creating cleaned train and test sets:

```python
import  pipeline.raw_data import clean_uc_merced

data = clean_uc_merced.load("<path/to/image/directory/>")

```



## References 

 - UC Merced
     - Dataset: [http://weegee.vision.ucmerced.edu/datasets/landuse.html](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

 - EuroSAT
     - Paper: [https://arxiv.org/pdf/1709.00029.pdf](https://arxiv.org/pdf/1709.00029.pdf)
     - Dataset: [http://madm.dfki.de/downloads](http://madm.dfki.de/downloads)

 - DSTL
     - Dataset: [https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)




