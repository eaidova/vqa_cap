# Data preparation
## 1. Download data
The easiest way to download all data - use downloading script:
```bash
tools/download.sh
``` 

Alternatively you can download data by yourself using following links:
  * [GloVe representation](http://nlp.stanford.edu/data/glove.6B.zip)
  * VQA2.0 questions and annotations:
    - [train questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip)
    - [val questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip)
    - [test questions](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip)
    - [train annotation](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip)
    - [val annotation](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)
  * Faster RCNN image features
    - [trainval images](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip)
    - [test images](https://imagecaption.blob.core.windows.net/imagecaption/test2015_36.zip)
    
**Note! All components of the dataset should be unpacked to `data` directory**
## 2. Preprocess data
```bash
tools/process.sh
```