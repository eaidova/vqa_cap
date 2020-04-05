import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.VQACAPdataset import Dictionary, VQAFeatureDataset
from train import train
from model.weights_init import init_weights
from model.models import build_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='set this to evaluate.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_hid', type=int, default=1280) # they used 1024
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dropout_L', type=float, default=0.1)
    parser.add_argument('--dropout_G', type=float, default=0.2)
    parser.add_argument('--dropout_W', type=float, default=0.4)
    parser.add_argument('--dropout_C', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='PReLU, ReLU, LeakyReLU, Tanh, Hardtanh, Sigmoid, RReLU, ELU, SELU')
    parser.add_argument('--norm', type=str, default='weight', help='weight, batch, layer, none')
    parser.add_argument('--model', type=str, default='A3x2')
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='Adamax', help='Adam, Adamax, Adadelta, RMSprop')
    parser.add_argument('--initializer', type=str, default='kaiming_normal')
    parser.add_argument('--seed', type=int, default=9731, help='random seed')
    parser.add_argument('--device_ids', type=int, default=[0], nargs='+', required=False)
    args = parser.parse_args()
    return args


def output_dir(
        out_dir, model_name, num_hidden, activation, optimizer,
        dropout, dropout_l, dropout_g, dropout_w, dropout_c,
        weights_decay, init
):
    return '{}/{}_{}_{}_{}_D{}_DL{}_DG{}_DW{}_DC{}_W{}_init_{}'.format(
        out_dir, model_name, num_hidden, activation, optimizer,
        dropout, dropout_l, dropout_g, dropout_w, dropout_c, weights_decay, init
    )


if __name__ == '__main__':
    args = parse_args()

    seed = 0
    if args.seed == 0:
        seed = random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(args.seed)
    else:
        seed = args.seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    output = output_dir(
        args.output, args.model, args.num_hid, args.activation, args.optimizer,
        args.dropout, args.dropout_L, args.dropout_G, args.dropout_W, args.dropout_C,
        args.weights_decay, args.initializer
    )
    dictionary = Dictionary.load_from_file('data/dictionary.pkl') # question dictionary

    caption_dictionary = Dictionary.load_from_file('data/caption_dictionary.pkl') 
    train_dset = VQAFeatureDataset('train', dictionary, caption_dictionary)
    batch_size = args.batch_size
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_dset = VQAFeatureDataset('val', dictionary, caption_dictionary)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=0)
    model = build_model(args.model,
                        train_dset,
                        num_hid=args.num_hid,
                        dropout= args.dropout,
                        norm=args.norm,
                        activation=args.activation,
                        dropL=args.dropout_L,
                        dropG=args.dropout_G,
                        dropW=args.dropout_W,
                        dropC=args.dropout_C)
    model = model.cuda()
    init_weights(model, args.initializer)
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model,device_ids = args.device_ids)

    train(model, train_loader, eval_loader, args.epochs, output, args.optimizer, args.weight_decay)

