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
    - [train images](nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip)
    - [val images](nlp.cs.unc.edu/data/lxmert_data/val2014_obj36.zip)
    - [test images](nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/test2015_obj36.zip)
    
**Note! All components of the dataset should be unpacked to `data` directory**
## 2. Preprocess data
```bash
tools/process.sh
```
