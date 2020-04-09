import argparse
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.VQACAPdataset import Dictionary, VQAFeatureDataset
from train import train
from model.weights_init import init_weights
from model.models import build_model
from utils.config import read_config
from test import test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, choices=['train', 'test'],
        help='running mode: train or test', required=False, default='train'
    )
    parser.add_argument('--config', type=str)
    parser.add_argument('--device_ids', type=int, default=[0], nargs='+', required=False)
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--checkpoint', type=str, required=False)
    args = parser.parse_args()
    return args


def output_dir(out_dir, model_config):
    model_name = model_config['type']
    num_hidden = model_config['num_hidden']
    activation = model_config['activation']
    train_config = model_config['train']
    return '{}/{}_{}_{}_{}_D{}_DL{}_DG{}_DW{}_DC{}_W{}_init_{}'.format(
        out_dir, model_name, num_hidden, activation, train_config['optimizer'],
        train_config['dropout'], train_config['dropout_l'], train_config['dropout_g'],
        train_config['dropout_w'], train_config['dropout_c'], train_config['weights_decay'], train_config['initializer']
    )


def main(args):
    config = read_config(args.config)
    mode = args.mode
    seed = config.get(mode, {}).get('seed', 0)
    if seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    output = output_dir(args.output, config)
    dictionary = Dictionary.load_from_file('data/dictionary.pkl') # question dictionary
    caption_dictionary = Dictionary.load_from_file('data/caption_dictionary.pkl')
    if mode == 'train':
        train_dset = VQAFeatureDataset('train', dictionary, caption_dictionary)
        batch_size = config['train']['batch_size']
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
        eval_dset = VQAFeatureDataset('val', dictionary, caption_dictionary)
        eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=0)
        model = build_model(config, train_dset)
        model.cuda()
        if not args.checkpoint:
            init_weights(model, config['train']['initializer'])
            model.w_emb.init_embedding('data/glove6b_init_300d.npy')
        model = nn.DataParallel(model, device_ids= args.device_ids)
        train(model, train_loader, eval_loader, output, config['train'], args.checkpoint)
    else:
        if args.checkpoint is None or not os.path.exists(args.checkpoint):
            raise ValueError('checkpoint for testing is not provided')
        test_dset = VQAFeatureDataset('test', dictionary, caption_dictionary)
        batch_size = config['test'].get('batch_size', 1)
        test_loader = DataLoader(test_dset, batch_size, shuffle=False, num_workers=8)
        model = build_model(config, test_dset)
        model.cuda()
        model = nn.DataParallel(model, device_ids=args.device_ids)
        test(model, test_loader, output, args.checkpoint)

if __name__ == '__main__':
    args = parse_args()
    main(args)

