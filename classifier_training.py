import argparse
from pathlib import Path
import time
import wandb
import torch
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from mixup import mixup_data, mixup_criterion
from utils import accuracy, multitask_accuracy, save_checkpoint, AverageMeter, text_feature_generator
from video_encoder import video_encoder
from generalized_model import final_model
from data_sets import EpicKitchens
import os

parser = argparse.ArgumentParser(description=('Train Multimodal Audio Text Encoder Decoder on Sequence ' +
                                              'of actions from untrimmed video'))

# ------------------------------ Dataset -------------------------------
parser.add_argument('--train_data', type=Path)
parser.add_argument('--train_labels', type=Path)
parser.add_argument('--val_data', type=Path)
parser.add_argument('--val_labels', type=Path)
parser.add_argument('--clip_mode', choices=['random', 'all'], default='random')
parser.add_argument('--clip_number', type=int, default=10)
parser.add_argument('--seg_number', type=int, default=1)
parser.add_argument('--swapping', action='store_true')

# ------------------------------ Model ---------------------------------

parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--seq_change', type=int, default=-1)
parser.add_argument('--visual_input_dim', type=int, default=1024)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--dim_feedforward', type=int, default=2048)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--classification_mode', choices=['summary', 'all'], default='summary')
parser.add_argument('--dropout', type=float, default=0.1)

# ------------------------------ Train ----------------------------------

parser.add_argument('--general_model', type=Path)

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--cuda_core', type=int, default=0)

# ------------------------------ Optimizer ------------------------------
parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[25, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--weight_loss', default=None, type=float,
                    metavar='W', help='loss for all weight of sequence integration')
parser.add_argument('--loss_impacts', default=[1, 1, 1], type=int, nargs='+',
                    help=' loss impacts of heads, [visual decoder, text decoder, classifier]')
# ------------------------------ Misc ------------------------------------
parser.add_argument('--output_dir', type=Path)
parser.add_argument('--disable_wandb_log', action='store_true')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--project_name', type=str, default='Thesis')
parser.add_argument('--running_name', type=str, default='general pipeline training')

args = parser.parse_args()

if not args.output_dir.exists():
    args.output_dir.mkdir(parents=True)

torch.cuda.set_device(args.cuda_core)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.cuda_core)
device = torch.device(device)

_DATASETS = EpicKitchens
_NUM_CLASSES = [97, 300]


def main():
    global args, device
    weight = None
    if not args.weight_loss == None:
        weight = args.weight_loss

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    if args.seq_change == -1:
        args.seq_change = args.seq_len
    general_model = video_encoder(_NUM_CLASSES,
                                  seq_len=args.seq_change,
                                  num_clips=args.seg_number,
                                  visual_input_dim=args.visual_input_dim,
                                  d_model=args.d_model,
                                  dim_feedforward=args.dim_feedforward,
                                  nhead=args.nhead,
                                  num_layers=args.num_layers,
                                  dropout=args.dropout)

    general_model = general_model.to(device)

    # loading general model
    checkpoint = torch.load(os.path.join(args.general_model, 'checkpoint.pyth'))  # 'model_best.pyth'))
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    general_model.load_state_dict(checkpoint['state_dict'])


    # loading the classifier model
    model = final_model(_NUM_CLASSES,
                        seq_len=args.seq_len,
                        num_clips=args.seg_number,
                        visual_input_dim=args.visual_input_dim,
                        d_model=args.d_model,
                        dim_feedforward=args.dim_feedforward,
                        nhead=args.nhead,
                        num_layers=args.num_layers,
                        dropout=args.dropout)

    model = model.to(device)

    model.set_representation(general_model.aggregation)  # , general_model.representation_layer1,
    # general_model.dropout_layer, general_model.representation_layer2, general_model.relu)

    if not args.disable_wandb_log:
        wandb.init(project=str(args.project_name),
                   name=str(args.running_name),
                   config=args)
        wandb.watch(model)

    dataset = _DATASETS
    train_loader = torch.utils.data.DataLoader(
        dataset(args.seg_number,
                args.train_data,
                args.train_labels,
                visual_feature_dim=args.visual_input_dim,
                window_len=args.seq_len,
                num_clips=args.clip_number,
                clips_mode=args.clip_mode,
                swapping=False,
                ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        dataset(args.seg_number,
                args.val_data,
                args.val_labels,
                visual_feature_dim=args.visual_input_dim,
                window_len=args.seq_len,
                num_clips=args.clip_number,
                clips_mode=args.clip_mode,
                ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion_classifier = torch.nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, args.lr_steps, gamma=0.1)

    best_message = None
    best_prec1 = 0

    for epoch in range(args.epochs):
        train(train_loader, model, criterion_classifier, optimizer, epoch, device)
        # evaluate on validation set
        prec1, temp_message = validation(val_loader, model, criterion_classifier, device)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_message = temp_message
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, args.output_dir)
        scheduler.step()
    print(best_message)
    print(args)


def train(train_loader, model, criterion_classifier, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    verb_losses = AverageMeter()
    noun_losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, label, _, narration_id) in enumerate(train_loader):

        batch_size = data.shape[0]
        data_time.update(time.time() - end)

        data = data.to(device)
        data, target_a, target_b, lam = mixup_data(data, label, alpha=0.2)

        # compute output
        output_classifier = model(data)

        target_a = {k: v.to(device) for k, v in target_a.items()}
        target_b = {k: v.to(device) for k, v in target_b.items()}

        loss_verb = mixup_criterion(criterion_classifier, output_classifier[0], target_a['verb'], target_b['verb'], lam,
                                    weights=None)
        loss_noun = mixup_criterion(criterion_classifier, output_classifier[1], target_a['noun'], target_b['noun'], lam,
                                    weights=None)
        loss = 0.5 * (loss_verb + loss_noun)

        verb_losses.update(loss_verb.item(), batch_size)
        noun_losses.update(loss_noun.item(), batch_size)
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not args.disable_wandb_log:
            wandb.log(
                {
                    "Train/loss": losses.avg,
                    "Train/epochs": epoch,
                    "Train/lr": optimizer.param_groups[-1]['lr'],
                    "Train/verb/loss": verb_losses.avg,
                    "Train/noun/loss": noun_losses.avg,
                },
            )

        if i % args.print_freq == 0:
            message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' +
                       'Time {batch_time.avg:.3f} ({batch_time.avg:.3f})\t' +
                       'Data {data_time.avg:.3f} ({data_time.avg:.3f})\t' +
                       'Loss {loss.avg:.4f} ({loss.avg:.4f})\t' +
                       'Verb Loss {verb_loss.avg:.4f} ({verb_loss.avg:.4f})\t' +
                       'Noun Loss {noun_loss.avg:.4f} ({noun_loss.avg:.4f})\t'  # +
                       ).format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, verb_loss=verb_losses,
                noun_loss=noun_losses,
                lr=optimizer.param_groups[-1]['lr'])

            print(message)


def validation(val_loader, model, criterion_classifier, device):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        verb_losses = AverageMeter()
        noun_losses = AverageMeter()
        verb_top1 = AverageMeter()
        verb_top5 = AverageMeter()
        noun_top1 = AverageMeter()
        noun_top5 = AverageMeter()

        model.eval()
        end = time.time()
        for i, (data, label, _, narration_id) in enumerate(val_loader):
            batch_size = len(narration_id)

            data = data.to(device)
            output_classifier = model(data)

            label = {k: v.to(device) for k, v in label.items()}
            loss_classifier_verb = criterion_classifier(output_classifier[0], label['verb'])
            loss_classifier_noun = criterion_classifier(output_classifier[1], label['noun'])
            loss_classifier = 0.5 * (loss_classifier_noun + loss_classifier_verb)

            verb_losses.update(loss_classifier_verb.item(), batch_size)
            noun_losses.update(loss_classifier_noun.item(), batch_size)

            verb_output = output_classifier[0]
            noun_output = output_classifier[1]

            verb_target = label['verb']
            noun_target = label['noun']

            verb_prec1, verb_prec5 = accuracy(verb_output, verb_target, topk=(1, 5))

            verb_top1.update(verb_prec1, batch_size)
            verb_top5.update(verb_prec5, batch_size)

            noun_prec1, noun_prec5 = accuracy(noun_output, noun_target, topk=(1, 5))

            noun_top1.update(noun_prec1, batch_size)
            noun_top5.update(noun_prec5, batch_size)

            prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                              (verb_target, noun_target),
                                              topk=(1, 5))

            losses.update(loss_classifier.item(), batch_size)
            top1.update(prec1, batch_size)
            top5.update(prec5, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if not args.disable_wandb_log:
            wandb.log(
                {
                    "Val/loss": losses.avg,
                    "Val/Top1_acc": top1.avg,
                    "Val/Top5_acc": top5.avg,
                    "Val/verb/loss": verb_losses.avg,
                    "Val/verb/Top1_acc": verb_top1.avg,
                    "Val/verb/Top5_acc": verb_top5.avg,
                    "Val/noun/loss": noun_losses.avg,
                    "Val/noun/Top1_acc": noun_top1.avg,
                    "Val/noun/Top5_acc": noun_top5.avg,
                },
            )

        message = ("Testing Results: "
                   " Verb: Prec@1 {verb_top1.avg:.3f} Prec@5 {verb_top5.avg:.3f} "
                   " Noun: Prec@1 {noun_top1.avg:.3f} Prec@5 {noun_top5.avg:.3f} "
                   " Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} "
                   " Verb Loss {verb_loss.avg:.5f} "
                   " Noun Loss {noun_loss.avg:.5f} "
                   " Loss {loss.avg:.5f}").format(verb_top1=verb_top1, verb_top5=verb_top5,
                                                  noun_top1=noun_top1, noun_top5=noun_top5,
                                                  top1=top1, top5=top5,
                                                  verb_loss=verb_losses,
                                                  noun_loss=noun_losses,
                                                  loss=losses)
        print(message)

    return top1.avg, message


if __name__ == '__main__':
    main()
