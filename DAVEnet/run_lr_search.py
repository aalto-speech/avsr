# Author: David Harwath
import argparse
import os
import json
import time
import torch

import dataloaders
import models
from steps.util import *
from steps.lr_finder import *

print("I am process %s, running on %s: starting (%s)" % (
    os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='',
                    help="training data json")
parser.add_argument("--data-val", type=str, default='',
                    help="validation data json")
parser.add_argument("--exp-dir", type=str, default="",
                    help="directory to dump experiments")
parser.add_argument("--optim", type=str, default="sgd",
                    help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.0000001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--max-lr', default=10, type=float,
                    help='maximum learning rate')
parser.add_argument('--lr-decay', default=5, type=int, metavar='LRDECAY',
                    help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--num-iter', default=2000, type=int,
                    help='Number of iterations to run the learning rate search')
parser.add_argument('--mode', default="exp", type=str,
                    help='LR search mode', choices=["exp", "linear"])
parser.add_argument("--n_epochs", type=int, default=50,
                    help="number of maximum training epochs")
parser.add_argument("--n_print_steps", type=int, default=100,
                    help="number of steps to print statistics")
parser.add_argument("--audio-model", type=str, default="Davenet",
                    help="audio model architecture", choices=["Davenet"])
parser.add_argument("--image-model", type=str, default="VGG16",
                    help="image model architecture", choices=["VGG16"])
parser.add_argument("--pretrained-image-model", action="store_true",
                    dest="pretrained_image_model", help="Use an image network pretrained on ImageNet")
parser.add_argument("--margin", type=float, default=1.0, help="Margin paramater for triplet loss")
parser.add_argument("--input-length", "-L", type=int, default=2048,
                    help="number of input frames", choices=[1024, 2048])

args = parser.parse_args()

print(args)

# Setup loaders, models and loss
train_loader = torch.utils.data.DataLoader(
    dataloaders.ImageCaptionDataset(args.data_train, audio_conf={'target_length': args.input_length}),
    batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloaders.ImageCaptionDataset(args.data_val, audio_conf={'target_length': args.input_length}, image_conf={'center_crop': True}),
    batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

audio_model = models.Davenet(embedding_dim=args.input_length)
image_model = models.VGG16(embedding_dim=args.input_length, pretrained=args.pretrained_image_model)


criterion = MatchMapLoss()

# Set up the optimizer
audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
image_trainables = [p for p in image_model.parameters() if p.requires_grad]
trainables = audio_trainables + image_trainables
if args.optim == 'sgd':
    optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
elif args.optim == 'adam':
    optimizer = torch.optim.Adam(trainables, args.lr,
                                 weight_decay=args.weight_decay,
                                 betas=(0.95, 0.999))
else:
    raise ValueError('Optimizer %s is not supported' % args.optim)

# Run learning rate search and plot result
lr_finder = LRFinder(image_model, audio_model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=args.max_lr, 
                     num_iter=args.num_iter, step_mode=args.mode)

# Save results
save_file = "LRmin-{}_LRmax-{}_NumIter-{}_Mode-{}_Optim-{}_history.txt".format(
            args.lr, args.max_lr, args.num_iter, args.mode, args.optim)
best_loss = {"best_loss": lr_finder.best_loss}
best_acc = {"best_acc": lr_finder.best_acc}
with open(save_file, 'w') as file:
    file.write(json.dumps(lr_finder.history))
    file.write(json.dumps(best_loss))
    file.write(json.dumps(best_acc))

# Plot
loss_name = "LRmin-{}_LRmax-{}_NumIter-{}_Mode-{}_Optim-{}_loss.png".format(
            args.lr, args.max_lr, args.num_iter, args.mode, args.optim)
acc_name = "LRmin-{}_LRmax-{}_NumIter-{}_Mode-{}_Optim-{}_acc.png".format(
            args.lr, args.max_lr, args.num_iter, args.mode, args.optim)
if args.mode == "exp":
    lr_finder.plot(loss_name=loss_name, acc_name=acc_name)
elif args.mode == "linear":
    lr_finder.plot(log_lr=False, loss_name=loss_name, acc_name=acc_name)
else:
    raise ValueError("Expected mode to be one of (exp, linear), got {}".format(args.mode))

"""
if not bool(args.exp_dir):
    print("exp_dir not specified, automatically creating one...")
    args.exp_dir = "exp/Data-%s/AudioModel-%s_ImageModel-%s_Optim-%s_LR-%s_Epochs-%s" % (
        os.path.basename(args.data_train), args.audio_model, args.image_model, args.optim,
        args.lr, args.n_epochs)

print("\nexp_dir: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
pickle.dump(args, f)
"""

