import os
import json
import pickle
import utils.common_utils as utils
from pycocotools.coco import COCO

def _create_entry():
    entry = {
        'question_id': [], #question['question_id'],
        'image_id': [], #question['image_id'],
        'image': [], #img,
        'caption': [],
        'question': [], #question['question'],
        'answer': [] #answer
    }
    return entry

dataroot = 'data'
names = ['train','val']
for name in names:

    img_id2idx = pickle.load(open(os.path.join(dataroot, '{}36_imgid2idx.pkl'.format(name), 'rb')))
    question_path = os.path.join(dataroot, 'v2_OpenEnded_mscoco_{}2014_questions.json'.format(name))
    questions = sorted(json.load(open(question_path))['questions'],key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '{}_target.pkl'.format(name))
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    utils.assert_eq(len(questions), len(answers))

    entries = {}
    coco = COCO('data/annotations/captions_{}2014.json'.format(name))

    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        image_id = 'COCO_{}2014_{}'.format(name, str(img_id).zfill(12))
        feat_id = img_id2idx[image_id]
        if feat_id not in entries:
            entries[feat_id] = _create_entry()
        entries[feat_id]['question_id'] = question['question_id']
        entries[feat_id]['image_id'] = question['image_id']
        entries[feat_id]['image'] = feat_id
        entries[feat_id]['question'].append(question)
        entries[feat_id]['answer'].append(answer)

    caption_ids = list(coco.anns.keys())
    length_captions = 0
    for i in range(len(caption_ids)):
        idx = caption_ids[i]
        caption = coco.anns[idx]['caption']
        feat_id = img_id2idx['COCO_{}2014_{}'.format(name, str(coco.anns[idx]['image_id']).zfill(12))]
        entries[feat_id]['caption'].append(caption)


    pickle.dump(entries, open('VQA_caption_{}dataset.pkl'.format(name), 'wb'))

img_id2idx = pickle.load(open(os.path.join(dataroot, 'test36_imgid2idx.pkl', 'rb')))
question_path = os.path.join(dataroot, 'v2_OpenEnded_mscoco_test2015_questions.json')
questions = sorted(json.load(open(question_path))['questions'],key=lambda x: x['question_id'])
entries = {}

for question in questions:
    img_id = question['image_id']
    image_id = 'COCO_test2015_{}'.format(str(img_id).zfill(12))
    feat_id = img_id2idx[image_id]
    if feat_id not in entries:
        entries[feat_id] = _create_entry()
    entries[feat_id]['question_id'] = question['question_id']
    entries[feat_id]['image_id'] = question['image_id']
    entries[feat_id]['image'] = feat_id
    entries[feat_id]['question'].append(question)

pickle.dump(entries, open('VQA_caption_{}dataset.pkl'.format(name), 'wb'))
