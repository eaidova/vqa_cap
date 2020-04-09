## Script for downloading data

# GloVe Vectors
wget -P data http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove
rm data/glove.6B.zip

# Questions
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip data/v2_Questions_Train_mscoco.zip -d data
rm data/v2_Questions_Train_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip data/v2_Questions_Val_mscoco.zip -d data
rm data/v2_Questions_Val_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip data/v2_Questions_Test_mscoco.zip -d data
rm data/v2_Questions_Test_mscoco.zip

# Annotations
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip data/v2_Annotations_Train_mscoco.zip -d data
rm data/v2_Annotations_Train_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip data/v2_Annotations_Val_mscoco.zip -d data
rm data/v2_Annotations_Val_mscoco.zip

# Caption Annotation
wget -P data http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip data/annotations_trainval2014.zip -d data
rm data/annotations_trainval2014.zip

#Image Features
wget nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data
unzip data/train2014_obj36.zip -d data && rm data/train2014_obj36.zip
wget nlp.cs.unc.edu/data/lxmert_data/val2014_obj36.zip -P data
unzip data/val2014_obj36.zip -d data && rm data/val2014_obj36.zip
wget nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/test2015_obj36.zip -P data
unzip data/test2015_obj36.zip -d data && rm data/test2015_obj36.zip
