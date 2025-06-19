import argparse
import os
from pathlib import Path
import time
import wandb
import torch
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
# from mixup import mixup_data, mixup_criterion
from mixup import mixup_data, mixup_criterion, mixup_data_pipeline
from utils import accuracy, multitask_accuracy, save_checkpoint, AverageMeter, text_feature_generator
from general_pipeline import General_Encoder_Decoder
from data_sets import EpicKitchens, egtea

parser = argparse.ArgumentParser(description=('Train Multimodal Audio Text Encoder Decoder on Sequence ' +
                                              'of actions from untrimmed video'))

# ------------------------------ Dataset -------------------------------
parser.add_argument('--train_data', type=Path)
parser.add_argument('--train_labels', type=Path)
parser.add_argument('--val_data', type=Path)
parser.add_argument('--val_labels', type=Path)
parser.add_argument('--clip_mode', choices=['random', 'all'], default='random')
parser.add_argument('--dataset', choices=['epic', 'egtea', 'epic55'], default='epic')
parser.add_argument('--clip_number', type=int, default=10)
parser.add_argument('--seg_number', type=int, default=1)
parser.add_argument('--swapping', action='store_true')
parser.add_argument('--feature_path', type=Path)
parser.add_argument('--random_swapping', action='store_true')

# ------------------------------ Model ---------------------------------

parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--visual_input_dim', type=int, default=1024)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--dim_feedforward', type=int, default=2048)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--classification_mode', choices=['summary', 'all'], default='summary')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--modalities',  nargs='+', default=['rgb'])


# ------------------------------ Train ----------------------------------
parser.add_argument('--path_to_encoder', type=Path)

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--cuda_core', type=int, default=0)
parser.add_argument('--mix_up', action='store_true')

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
parser.add_argument('--loss_impacts', default=[1, 1, 1], type=float, nargs='+',
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



# Loading the dataset
if args.dataset == 'epic':
    _NUM_CLASSES = [97, 300]
    _DATASETS = EpicKitchens
elif args.dataset == 'egtea':
    _NUM_CLASSES = 106
    _DATASETS = egtea
# elif args.dataset == 'epic55':
#     _NUM_CLASSES = 8
#     _DATASETS = EpicKitchens55


torch.autograd.set_detect_anomaly(True)


def main():
    global args, device

    np.random.seed(0)
    torch.manual_seed(0)


    # Loading the main model 
    model = General_Encoder_Decoder(_NUM_CLASSES,
                                    seq_len=args.seq_len,
                                    num_clips=args.seg_number,
                                    visual_input_dim=args.visual_input_dim,
                                    d_model=args.d_model,
                                    dim_feedforward=args.dim_feedforward,
                                    nhead=args.nhead,
                                    num_layers=args.num_layers,
                                    dropout=args.dropout,
                                    classification_mode=args.classification_mode,
                                    dataset=args.dataset,
                                    modalities=args.modalities
                                    )

    model = model.to(device)


    # wandb setup
    if not args.disable_wandb_log:
        wandb.init(project=str(args.project_name),
                   name=str(args.running_name),
                   config=args)
        wandb.watch(model)
        wandb.watch(model.video_encoder)

    
    if args.classification_mode == 'all':
        center_classification = False
    else:
        center_classification = True
    
    #loading the dataload and dataset
    dataset = _DATASETS
    train_loader = torch.utils.data.DataLoader(
        dataset(args.seg_number,
                args.train_data,
                args.train_labels,
                text_feature_path=args.feature_path,
                visual_feature_dim=args.visual_input_dim,
                window_len=args.seq_len,
                num_clips=args.clip_number,
                clips_mode=args.clip_mode,
                swapping=args.swapping,
                center_classification=center_classification,
                text_feature_availablity=True,
                random_swapping=args.random_swapping,
                modalities=args.modalities,
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
                modalities=args.modalities,
                ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion_text = torch.nn.CrossEntropyLoss()
    criterion_classifier = torch.nn.CrossEntropyLoss()
    criterion_vision = torch.nn.MSELoss()

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
    proc_start_time = time.time()
    for epoch in range(args.epochs):
        train(train_loader, model, criterion_text, criterion_classifier, criterion_vision, optimizer, epoch, device)
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
        cnt_time = time.time() - proc_start_time
        print('The average training time per epoch is {} sec/video'.format(
            float(cnt_time) / (epoch + 1)))
    print(best_message)
    print(args)



def train(train_loader, model, criterion_text, criterion_classifier, criterion_vision, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    text_losses = AverageMeter()
    classifier_losses = AverageMeter()
    vision_losses = AverageMeter()
    if args.dataset == 'epic':
        verb_losses = AverageMeter()
        noun_losses = AverageMeter()
    else:
        action_losses = AverageMeter()

    if args.classification_mode == 'all':
        weights = torch.tensor(args.seq_len * [0.1] + [0.9]).unsqueeze(0).cuda(device=device)
    else:
        weights = None
    model.train()
    end = time.time()
    for i, (data, label, narrations_input, narration_id) in enumerate(train_loader):
       # import pdb;
       # pdb.set_trace()

        batch_size = data.shape[0]
        data_time.update(time.time() - end)

        text_input = narrations_input.to(device)

        data = data.to(device)
       # import pdb; pdb.set_trace()
        if args.mix_up:
            data, text_input, target_a, target_b, lam = mixup_data_pipeline(data, text_input, label, alpha=0.2,
                                                                            dataset=args.dataset)

        with torch.no_grad():
            model.eval()
            cloned_input = data.clone().detach()
            presentation_labels = model.video_encoder(cloned_input, True)

        model.train()
        output_text, output_video, output_classifier = model(data, text_input)
        if args.mix_up:

            if args.dataset == 'epic':
                target_a = {k: v.to(device) for k, v in target_a.items()}
                target_b = {k: v.to(device) for k, v in target_b.items()}
                if args.classification_mode == 'all':

                    loss_text_verb = mixup_criterion(criterion_text, output_text[:, 300:], target_a['verb'][:, -1],
                                                     target_b['verb'][:, -1],
                                                     lam,
                                                     weights=None)
                    loss_text_noun = mixup_criterion(criterion_text, output_text[:, :300], target_a['noun'][:, -1],
                                                     target_b['noun'][:, -1],
                                                     lam,
                                                     weights=None)

                else:

                    loss_text_verb = mixup_criterion(criterion_text, output_text[:, 300:], target_a['verb'],
                                                     target_b['verb'],
                                                     lam,
                                                     weights=None)
                    loss_text_noun = mixup_criterion(criterion_text, output_text[:, :300], target_a['noun'],
                                                     target_b['noun'],
                                                     lam,
                                                     weights=None)

                loss_text = (0.5 * (loss_text_noun + loss_text_verb))

                loss_classifier_verb = mixup_criterion(criterion_classifier, output_classifier[0], target_a['verb'],
                                                       target_b['verb'], lam,
                                                       weights=weights)
                loss_classifier_noun = mixup_criterion(criterion_classifier, output_classifier[1], target_a['noun'],
                                                       target_b['noun'], lam,
                                                       weights=weights)

                loss_classifier = (0.5 * (loss_classifier_noun + loss_classifier_verb))
            else:
                target_a = target_a.to(device)
                target_b = target_b.to(device)
                if args.classification_mode == 'all':

                    loss_text = mixup_criterion(criterion_text, output_text, target_a[:, -1],
                                                target_b[:, -1],
                                                lam,
                                                weights=None)

                else:

                    loss_text = mixup_criterion(criterion_text, output_text, target_a,
                                                target_b,
                                                lam,
                                                weights=None)

                loss_classifier = mixup_criterion(criterion_classifier, output_classifier, target_a,
                                                  target_b, lam,
                                                  weights=weights)


        else:
            if args.dataset == 'epic':
                label = {k: v.to(device) for k, v in label.items()}
                if args.classification_mode == 'all':
                    loss_text_verb = criterion_text(output_text[:, 300:], label['verb'][:, -1])
                    loss_text_noun = criterion_text(output_text[:, :300], label['noun'][:, -1])
                else:
                    loss_text_verb = criterion_text(output_text[:, 300:], label['verb'])
                    loss_text_noun = criterion_text(output_text[:, :300], label['noun'])
                loss_text = (0.5 * (loss_text_noun + loss_text_verb))

                loss_classifier_verb = criterion_classifier(output_classifier[0], label['verb'])
                loss_classifier_noun = criterion_classifier(output_classifier[1], label['noun'])

                loss_classifier = (0.5 * (loss_classifier_noun + loss_classifier_verb))
            else:
                label = label.to(device)
                if args.classification_mode == 'all':
                    loss_text = criterion_text(output_text, label[:, -1])
                else:
                    loss_text = criterion_text(output_text, label)

                loss_classifier = criterion_classifier(output_classifier, label)


        loss_vision = criterion_vision(output_video,
                                       presentation_labels.reshape(batch_size, presentation_labels.shape[2]))

        loss = args.loss_impacts[0] * loss_vision + args.loss_impacts[1] * loss_text + args.loss_impacts[
            2] * loss_classifier

        text_losses.update(loss_text.item(), batch_size)
        classifier_losses.update(loss_classifier.item(), batch_size)
        vision_losses.update(loss_vision.item(), batch_size)
        if args.dataset == 'epic':
            verb_losses.update(loss_classifier_verb.item(), batch_size)
            noun_losses.update(loss_classifier_noun.item(), batch_size)
        else:
            action_losses.update(loss_classifier.item(), batch_size)
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if args.dataset == 'epic':
            if not args.disable_wandb_log:
                wandb.log(
                    {
                        "Train/loss_total": losses.avg,
                        "Train/loss_text": text_losses.avg,
                        "Train/loss_classifier": classifier_losses.avg,
                        "Train/loss_vision": vision_losses.avg,
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
                           'Vision Loss {vision_losses.avg:.4f} ({vision_losses.avg:.4f})\t' +
                           'Classifier Loss {classifier_losses.avg:.4f} ({classifier_losses.avg:.4f})\t' +
                           'Text Loss {text_losses.avg:.4f} ({text_losses.avg:.4f})\t'
                           ).format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, vision_losses=vision_losses,
                    classifier_losses=classifier_losses,
                    text_losses=text_losses,
                    lr=optimizer.param_groups[-1]['lr'])
                print(message)

        else:
            if not args.disable_wandb_log:
                wandb.log(
                    {
                        "Train/loss_total": losses.avg,
                        "Train/loss_text": text_losses.avg,
                        "Train/loss_classifier": classifier_losses.avg,
                        "Train/loss_vision": vision_losses.avg,
                        "Train/epochs": epoch,
                        "Train/lr": optimizer.param_groups[-1]['lr'],
                    },
                )
            if i % args.print_freq == 0:
                message = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t' +
                           'Time {batch_time.avg:.3f} ({batch_time.avg:.3f})\t' +
                           'Data {data_time.avg:.3f} ({data_time.avg:.3f})\t' +
                           'Loss {loss.avg:.4f} ({loss.avg:.4f})\t' +
                           'Vision Loss {vision_losses.avg:.4f} ({vision_losses.avg:.4f})\t' +
                           'Classifier Loss {classifier_losses.avg:.4f} ({classifier_losses.avg:.4f})\t' +
                           'Text Loss {text_losses.avg:.4f} ({text_losses.avg:.4f})\t'
                           ).format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, vision_losses=vision_losses,
                    classifier_losses=classifier_losses,
                    text_losses=text_losses,
                    lr=optimizer.param_groups[-1]['lr'])
                print(message)


def validation(val_loader, model, criterion_classifier, device):
    with torch.no_grad():
        batch_time = AverageMeter()

        if args.dataset == 'epic':
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            verb_losses = AverageMeter()
            noun_losses = AverageMeter()
            verb_losses = AverageMeter()
            noun_losses = AverageMeter()
            verb_top1 = AverageMeter()
            verb_top5 = AverageMeter()
            noun_top1 = AverageMeter()
            noun_top5 = AverageMeter()
        else:
            action_top1 = AverageMeter()
            action_top5 = AverageMeter()
            action_losses = AverageMeter()

        model.eval()
        end = time.time()
        for i, (data, label, narrations_input, narration_id) in enumerate(val_loader):
            batch_size = len(narration_id)

            data = data.to(device)
            output_classifier = model(data, narrations_input)
            if args.dataset == 'epic':
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
            else:
                label = label.to(device)
                loss_classifier = criterion_classifier(output_classifier, label)
                action_losses.update(loss_classifier.item(), batch_size)

                action_prec1, action_prec5 = accuracy(output_classifier, label, topk=(1, 5))

                action_top1.update(action_prec1, batch_size)
                action_top5.update(action_prec5, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        if args.dataset == 'epic':
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
        else:
            if not args.disable_wandb_log:
                wandb.log(
                    {
                        "Val/loss": action_losses.avg,
                        "Val/Top1_acc": action_top1.avg,
                        "Val/Top5_acc": action_top5.avg,
                    },
                )

            message = ("Testing Results: "
                       " action: Prec@1 {verb_top1.avg:.3f} Prec@5 {verb_top5.avg:.3f} "
                       " Loss {loss.avg:.5f}").format(verb_top1=action_top1, verb_top5=action_top5,
                                                      loss=action_losses)

        print(message)
    if args.dataset == 'epic':
        return top1.avg, message
    else:
        return action_top1.avg, message


if __name__ == '__main__':
    main()
    wandb.finish()


