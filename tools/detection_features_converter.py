"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import pickle
import numpy as np
import utils.common_utils as utils


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
train_infile = 'data/train2014_obj36.tsv'
val_infile = 'data/val2014_obj36.tsv'
train_data_file = 'data/train36.hdf5'
val_data_file = 'data/val36.hdf5'
train_indices_file = 'data/train36_imgid2idx.pkl'
val_indices_file = 'data/val36_imgid2idx.pkl'
train_ids_file = 'data/train_ids.pkl'
val_ids_file = 'data/val_ids.pkl'

feature_length = 2048
num_fixed_boxes = 36


def tsv_to_pickle(infile, image_ids, indices, img_bb, img_features, spatial_img_features):
    counter = 0
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['num_boxes'] = int(item['num_boxes'])
            image_id = item['img_id']
            image_w = float(item['img_w'])
            image_h = float(item['img_h'])
            bboxes = np.frombuffer(
                base64.b64decode(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            indices[image_id] = counter
            img_bb[counter, :, :] = bboxes
            img_features[counter, :, :] = np.frombuffer(
                    base64.b64decode(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
            spatial_img_features[counter, :, :] = spatial_features
            counter += 1
        return indices, img_bb, img_features, spatial_img_features

if __name__ == '__main__':
    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")

    if os.path.exists(train_ids_file) and os.path.exists(val_ids_file):
        with open(train_ids_file, 'rb') as train_file:
            train_imgids = pickle.load(train_file)
        with open(val_ids_file, 'rb') as val_file:
            val_imgids = pickle.load(val_file)
    else:
        train_imgids = utils.load_imageid('data/train2014')
        val_imgids = utils.load_imageid('data/val2014')
        pickle.dump(train_imgids, open(train_ids_file, 'wb'))
        pickle.dump(val_imgids, open(val_ids_file, 'wb'))

    train_indices = {}
    val_indices = {}

    train_img_features = h_train.create_dataset(
        'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
    train_spatial_img_features = h_train.create_dataset(
        'spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')

    val_img_bb = h_val.create_dataset(
        'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
    val_spatial_img_features = h_val.create_dataset(
        'spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')

    print("reading tsv...")
    train_indices, train_img_bb, train_img_features, train_spatial_img_features = tsv_to_pickle(train_infile, train_imgids, train_indices, train_img_bb, train_img_features, train_spatial_img_features)
    val_indices, val_img_bb, val_img_features, val_spatial_img_features = tsv_to_pickle(val_infile,
                                                                                                val_imgids,
                                                                                                val_indices,
                                                                                                val_img_bb,
                                                                                                val_img_features,
                                                                                                val_spatial_img_features)
    pickle.dump(train_indices, open(train_indices_file, 'wb'))
    pickle.dump(val_indices, open(val_indices_file, 'wb'))
    h_train.close()
    h_val.close()
    print("done!")
