# Author: David Harwath
import argparse
import os
import pickle
import time
import torch

import dataloaders
import models
from steps import train_classifier

print("I am process %s, running on %s: starting (%s)" % (
    os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='',
                    help="training data json")
parser.add_argument("--data-val", type=str, default='',
                    help="validation data json")
parser.add_argument("--exp-dir", type=str, default="",
                    help="directory to dump experiments")
parser.add_argument("--resume", action="store_true", dest="resume",
                    help="load from exp_dir if True")
parser.add_argument("--reparam-model", type=str, default="",
                    help="load the best model from given directory")
parser.add_argument("--optim", type=str, default="sgd",
                    help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--max-lr', default=10, type=float,
                    help='maximum learning rate')
parser.add_argument('--lr-decay', default=5, type=int, metavar='LRDECAY',
                    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--num-iter', default=2000, type=int,
                    help='Number of iterations to run the learning rate search')
parser.add_argument("--n_epochs", type=int, default=50,
                    help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=100,
                    help="number of steps to print statistics")
parser.add_argument("--audio-model", type=str, default="DaveClassifier",
                    help="audio model architecture", choices=["Davenet", "DaveClassifier"])
parser.add_argument("--pretrained-image-model", action="store_true",
                    dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--simtype", type=str, default="MISA",
                    help="matchmap similarity function", choices=["SISA", "MISA", "SIMA"])
parser.add_argument("--input-length", "-L", type=int, default=2048,
                    help="number of input frames", choices=[1024, 2048])

args = parser.parse_args()

resume = args.resume

if args.resume:
    assert (bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        args = pickle.load(f)
args.resume = resume

print(args)

train_loader = torch.utils.data.DataLoader(
    dataloaders.LabelCaptionDataset(args.data_train, audio_conf={'target_length': args.input_length}),
    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloaders.LabelCaptionDataset(args.data_val, audio_conf={'target_length': args.input_length}),
    batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

audio_model = models.DavenetClassifier()

if not bool(args.exp_dir):
    print("exp_dir not specified, automatically creating one...")
    args.exp_dir = "exp/Data-%s/AudioModel-%s_Optim-%s_LR-%s_Epochs-%s" % (
        os.path.basename(args.data_train), args.audio_model, args.optim,
        args.lr, args.n_epochs)

if not args.resume:
    print("\nexp_dir: %s" % args.exp_dir)
    os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

train_classifier(audio_model, train_loader, val_loader, args)
