# Datasets

**Folders containing training and test data are stored here**

These folders within this directory are populated using the `pipeline.raw_data` scripts.
You should not have to add anything yourself to this folder, the cleaning scrips
located in the above directory should handle this.

Folder hierarchy should be something like this:

```
 - datasets
    - raw_images
        - img1.jpg
        - img2.png
        - img3.tif
    - obj_detection
        - DOTA
            + images
            - test_annotation.csv
            - train_annotation.csv
            - classes_annotation.csv
        + DSTL
    - patch_identificaiton
        - uc_merced
            - xtest.p
            - xtrain.p
            - ytest.p
            - ytrain.p
        + whu_rs19
```


## Examples

Download the raw data from the UC Merced website, link below.

Example execution of creating cleaned train and test sets:

```python
import  pipeline.raw_data import clean_uc_merced

data = clean_uc_merced.load("<path/to/image/directory/>")

```


## References 

 - UC Merced (categorical)
     - Dataset: [http://weegee.vision.ucmerced.edu/datasets/landuse.html](http://weegee.vision.ucmerced.edu/datasets/landuse.html)  
 - EuroSAT (categorical)
     - Paper: [https://arxiv.org/pdf/1709.00029.pdf](https://arxiv.org/pdf/1709.00029.pdf)
     - Dataset: [http://madm.dfki.de/downloads](http://madm.dfki.de/downloads)
 - DSTL (object identification)
     - Dataset: [https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)
 - DOTA (object identification)
     - Paper: [https://arxiv.org/abs/1711.10398](https://arxiv.org/abs/1711.10398)
     - Dataset: [http://captain.whu.edu.cn/DOTAweb/](http://captain.whu.edu.cn/DOTAweb/)
 - WHU-RS19 (categorical)
     - Dataset: [http://captain.whu.edu.cn/repository_En.html](http://captain.whu.edu.cn/repository_En.html)
 - AID (categorical)
     - Dataset: [http://www.lmars.whu.edu.cn/xia/AID-project.html](http://www.lmars.whu.edu.cn/xia/AID-project.html)
 - RSI-CB (categorical)
     - Dataset: [https://github.com/lehaifeng/rsi-cb](https://github.com/lehaifeng/rsi-cb)
     - Paper: [https://arxiv.org/abs/1705.10450](https://arxiv.org/abs/1705.10450)


## Acknowledgments 

A sincere thank you to Dr. Gui-Song Xia (http://captain.whu.edu.cn/xia_En.html) for his hard work in compiling the DOTA dataset and giving me early access to benchmark the RetinaNet model used in this script. 

