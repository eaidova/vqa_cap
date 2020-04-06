"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

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
infile = 'data/test2015_obj36.tsv'
test_data_file = 'data/test36.hdf5'
test_indices_file = 'data/test36_imgid2idx.pkl'
test_ids_file = 'data/test_ids.pkl'

feature_length = 2048
num_fixed_boxes = 36


if __name__ == '__main__':
    h_test = h5py.File(test_data_file, "w")

    if os.path.exists(test_ids_file):
        with open(test_ids_file, 'rb') as test_file:
            test_imgids = pickle.load(test_file)
    else:
        test_imgids = utils.load_imageid('data/test2015')
        pickle.dump(test_imgids, open(test_ids_file, 'wb'))

    test_indices = {}

    test_img_features = h_test.create_dataset(
        'image_features', (len(test_imgids), num_fixed_boxes, feature_length), 'f')
    test_img_bb = h_test.create_dataset(
        'image_bb', (len(test_imgids), num_fixed_boxes, 4), 'f')
    test_spatial_img_features = h_test.create_dataset(
        'spatial_features', (len(test_imgids), num_fixed_boxes, 6), 'f')

    test_counter = 0

    print("reading tsv...")
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

         #   test_imgids.remove(image_id)
            test_indices[image_id] = test_counter
            test_img_bb[test_counter, :, :] = bboxes
            test_img_features[test_counter, :, :] = np.frombuffer(
                    base64.b64decode(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
            test_spatial_img_features[test_counter, :, :] = spatial_features
            test_counter += 1

    pickle.dump(test_indices, open(test_indices_file, 'wb'))
    h_test.close()
    print("done!")
