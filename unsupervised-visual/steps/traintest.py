import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from .util import *


def train(audio_model, image_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    dot_loss = DotLoss()
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc,
                         time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)

        # create/load exp
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)
    elif bool(args.reparam_model):
        progress_pkl = "%s/progress.pkl" % args.reparam_model
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % args.reparam_model))
        image_model.load_state_dict(torch.load("%s/models/best_image_model.pth" % args.reparam_model))
        epoch = best_epoch
        print("loaded parameters of model %s from epoch %d" % (args.reparam_model, epoch))

    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.num_iter,
                                                           eta_min=args.lr_limit)
    """
    Default Torch on Triton is old and doesn't yet have the CyclicLR optimizer
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  base_lr=args.lr,
                                                  max_lr=args.lr_limit,
                                                  step_size_up=args.num_iter,
                                                  mode="triangular2")
    """

    if args.resume:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)
    elif bool(args.reparam_model):
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (args.reparam_model, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()
    image_model.train()
    # for e in range(epoch, args.n_epochs + 1):
    while True:
        #adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model.train()
        image_model.train()
        for i, (image_input, audio_input, nframes) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)

            audio_input = audio_input.to(device)
            image_input = image_input.to(device)

            optimizer.zero_grad()

            audio_output = audio_model(audio_input)
            image_output = image_model(image_input)

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            nframes.div_(pooling_ratio)

            loss = dot_loss(image_output, audio_output)

            loss.backward()
            # Triton has torch 1.0.0 where scheduler step is executed first
            scheduler.step()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        recalls = validate(audio_model, image_model, test_loader, args)

        avg_acc = (recalls['A_r10'] + recalls['I_r10']) / 2

        torch.save(audio_model.state_dict(),
                   "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        torch.save(image_model.state_dict(),
                   "%s/models/image_model.%d.pth" % (exp_dir, epoch))
        torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        if avg_acc > best_acc:
            best_epoch = epoch
            best_acc = avg_acc
            shutil.copyfile("%s/models/audio_model.%d.pth" % (exp_dir, epoch),
                            "%s/models/best_audio_model.pth" % exp_dir)
            shutil.copyfile("%s/models/image_model.%d.pth" % (exp_dir, epoch),
                            "%s/models/best_image_model.pth" % exp_dir)
        _save_progress()
        epoch += 1


def validate(audio_model, image_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    # switch to evaluate mode
    image_model.eval()
    audio_model.eval()

    end = time.time()
    N_examples = len(val_loader.dataset)
    I_embeddings = []
    A_embeddings = []
    frame_counts = []
    with torch.no_grad():
        for i, (image_input, audio_input, nframes) in enumerate(val_loader):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)

            # compute output
            image_output = image_model(image_input)
            audio_output = audio_model(audio_input)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            nframes.div_(pooling_ratio)

            frame_counts.append(nframes.cpu())

            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        nframes = torch.cat(frame_counts)

        recalls = calc_recalls(image_output, audio_output)
        A_r10 = recalls['A_r10']
        I_r10 = recalls['I_r10']
        A_r5 = recalls['A_r5']
        I_r5 = recalls['I_r5']
        A_r1 = recalls['A_r1']
        I_r1 = recalls['I_r1']

    print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
          .format(A_r10=A_r10, I_r10=I_r10, N=N_examples), flush=True)
    print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
          .format(A_r5=A_r5, I_r5=I_r5, N=N_examples), flush=True)
    print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
          .format(A_r1=A_r1, I_r1=I_r1, N=N_examples), flush=True)

    return recalls


def train_classifier(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    cross_entropy_loss = nn.CrossEntropyLoss()
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc,
                         time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    # create/load exp
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)
    elif bool(args.reparam_model):
        progress_pkl = "%s/progress.pkl" % args.reparam_model
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % args.reparam_model))
        epoch = best_epoch
        print("loaded parameters of model %s from epoch %d" % (args.reparam_model, epoch))

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.num_iter,
                                                           eta_min=args.max_lr)
    """
    Default Torch on Triton is old and doesn't yet have the CyclicLR optimizer
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  base_lr=args.lr,
                                                  max_lr=args.max_lr,
                                                  step_size_up=args.num_iter,
                                                  mode="triangular2")
    """
    if args.resume:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)
    elif bool(args.reparam_model):
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (args.reparam_model, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()
    # for e in range(epoch, args.n_epochs + 1):
    while True:
        end_time = time.time()
        audio_model.train()
        for i, (labels, audio_input, nframes) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)

            audio_input = audio_input.to(device)
            labels = labels.to(device)
            print(labels)
            optimizer.zero_grad()

            audio_output = audio_model(audio_input)
            print(audio_output.shape)
            print(labels.shape)
            loss = cross_entropy_loss(audio_output, labels)

            loss.backward()
            # Triton has torch 1.0.0 where scheduler step is executed first
            scheduler.step()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        acc = validate_classifier(audio_model, test_loader, args)

        torch.save(audio_model.state_dict(),
                   "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            shutil.copyfile("%s/models/audio_model.%d.pth" % (exp_dir, epoch),
                            "%s/models/best_audio_model.pth" % (exp_dir))
        _save_progress()
        epoch += 1


def validate_classifier(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (labels, audio_input, nframes) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input)
            audio_output = audio_output.to('cpu').detach()
            _, predictions = torch.max(audio_output.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            batch_time.update(time.time() - end)
            end = time.time()

    accuracy = correct / total
    print("Accuracy of the network on the 1000 validation samples: {:.2f} %".format(100 * accuracy))
    return accuracy
