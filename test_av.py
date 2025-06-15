import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix, accuracy_score
from utils import accuracy, multitask_accuracy, save_checkpoint, AverageMeter, text_feature_generator
from data_sets import egtea, EpicKitchens
from general_pipeline import General_Encoder_Decoder
import pickle

_DATASETS = {'epic': EpicKitchens, 'egtea': egtea}
np.random.seed(0)
torch.manual_seed(0)


def evaluate_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.cuda_core)
    torch.cuda.set_device(args.cuda_core)
    _NUM_CLASSES = [97, 300] if args.dataset == 'epic' else 106

    net = General_Encoder_Decoder(_NUM_CLASSES,
                        seq_len=args.seq_len,
                        num_clips=args.seg_number,
                        visual_input_dim=args.visual_input_dim,
                        d_model=args.d_model,
                        dim_feedforward=args.dim_feedforward,
                        nhead=args.nhead,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        classification_mode='summary',
                        dataset=args.dataset,
                        modalities=args.modalities,
                        )

   # import pdb;
    #pdb.set_trace()
    checkpoint = torch.load(args.checkpoint)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    net.load_state_dict(checkpoint['state_dict'])

    dataset = EpicKitchens
    test_loader = torch.utils.data.DataLoader(
        dataset(args.seg_number,
                args.test_data,
                args.test_pickle,
                visual_feature_dim=args.visual_input_dim,
                window_len=args.seq_len,
                num_clips=args.clip_number,
                clips_mode=args.clip_mode,
                modalities=args.modalities,
                ),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.dataset == 'epic':
        top1 = AverageMeter()
        top5 = AverageMeter()
        verb_top1 = AverageMeter()
        verb_top5 = AverageMeter()
        noun_top1 = AverageMeter()
        noun_top5 = AverageMeter()
    else:
        action_top1 = AverageMeter()
        action_top5 = AverageMeter()
    net = net.to(device)
    with torch.no_grad():
        net.eval()
        results = []
        total_num = len(test_loader.dataset)

        proc_start_time = time.time()
        end = time.time()
        for i, (data, label, narrations_input, narration_id) in enumerate(test_loader):
            data = data.to(device)
            rst = net(data, narrations_input)


            if args.dataset == 'egtea':
                label = label.to(device)

                action_prec1, action_prec5 = accuracy(rst, label, topk=(1, 5))

                action_top1.update(action_prec1, 1)
                action_top5.update(action_prec5, 1)
                test_result = 1 if rst.argmax() == label[0] else 0
            else:
                label = {k: v.to(device) for k, v in label.items()}
                verb_output = rst[0]
                noun_output = rst[1]

                verb_target = label['verb']
                noun_target = label['noun']

                verb_prec1, verb_prec5 = accuracy(verb_output, verb_target, topk=(1, 5))

                verb_top1.update(verb_prec1, 1)
                verb_top5.update(verb_prec5, 1)

                noun_prec1, noun_prec5 = accuracy(noun_output, noun_target, topk=(1, 5))

                noun_top1.update(noun_prec1, 1)
                noun_top5.update(noun_prec5, 1)

                prec1, prec5 = multitask_accuracy((verb_output, noun_output),
                                                  (verb_target, noun_target),
                                                  topk=(1, 5))
                top1.update(prec1, 1)
                top5.update(prec5, 1)
                verb_test = 1 if verb_output.argmax() == verb_target[0] else 0
                noun_test = 1 if noun_output.argmax() == noun_target[0] else 0
                action_test = 1 if verb_test == 1 and noun_test == 1 else 0
                test_result = [verb_test, noun_test, action_test]
            if args.dataset == 'egtea':
                rst = rst.cpu().numpy().squeeze()
            else:

                rst = {'verb': rst[0].cpu().numpy().squeeze(),
                        'noun': rst[1].cpu().numpy().squeeze()}

            if args.dataset == 'egtea':
                label_ = label.item()
            else:
                label_ = {k: v.item() for k, v in label.items()}
            results.append((rst, label_, narration_id, test_result))

            cnt_time = time.time() - proc_start_time
            print('video {} done, total {}/{}, average {} sec/video'.format(
                i, i + 1, total_num, float(cnt_time) / (i + 1)))
        if args.dataset == 'epic':
            message = ("Testing Results: "
                       " Verb: Prec@1 {verb_top1.avg:.3f} Prec@5 {verb_top5.avg:.3f} "
                       " Noun: Prec@1 {noun_top1.avg:.3f} Prec@5 {noun_top5.avg:.3f} "
                       " Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} ").format(verb_top1=verb_top1, verb_top5=verb_top5,
                                                      noun_top1=noun_top1, noun_top5=noun_top5,
                                                      top1=top1, top5=top5)
        else:
            message = ("Testing Results: "
                       " action: Prec@1 {verb_top1.avg:.3f} Prec@5 {verb_top5.avg:.3f} ").format(verb_top1=action_top1, verb_top5=action_top5)
        print(message)
        return results


def print_accuracy(scores, labels):

    video_pred = [np.argmax(score) for score in scores]
    cf = confusion_matrix(labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_cnt[cls_hit == 0] = 1  # to avoid divisions by zero
    cls_acc = cls_hit / cls_cnt

    acc = accuracy_score(labels, video_pred)

    print('Accuracy {:.02f}%'.format(acc * 100))
    print('Average Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))


def save_scores(results, output):
    save_dict = {}
    if args.dataset == 'egtea':
        scores = np.array([result[0] for result in results])
        labels = np.array([result[1] for result in results])
        save_dict['scores'] = scores
        save_dict['labels'] = labels
        save_dict['correctness'] = np.array([result[2] for result in results])
    else:
        keys = results[0][0].keys()
        save_dict = {k + '_output': np.array([result[0][k] for result in results]) for k in keys}
        save_dict['narration_id'] = np.array([result[2] for result in results])
        save_dict['correctness'] = np.array([result[3] for result in results])


    with open(output, 'wb') as f:
        pickle.dump(save_dict, f)


def main():
    parser = argparse.ArgumentParser(description=('Test Audio-Visual Transformer on Sequence ' +
                                                  'of actions from untrimmed video'))
    parser.add_argument('--cuda_core', type=int, default=0)
    parser.add_argument('--test_data', type=Path)
    parser.add_argument('--test_pickle', type=Path)
    parser.add_argument('--dataset', choices=['epic', 'egtea'])
    parser.add_argument('--modalities', nargs='+', default=['rgb'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--visual_input_dim', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--window_len', type=int, default=60)
    parser.add_argument('--extract_attn_weights', action='store_true')
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--clip_mode', choices=['random', 'all'], default='random')
    parser.add_argument('--clip_number', type=int, default=10)
    parser.add_argument('--seg_number', type=int, default=1)
    parser.add_argument('--split')
    parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--base_line', action='store_true')


    global args
    args = parser.parse_args()


    results = evaluate_model()
    if ('test' not in args.split and 'epic' in args.dataset) or 'epic' not in args.dataset:
        if args.dataset == 'epic':
            keys = results[0][0].keys()
            for task in keys:
                print('Evaluation of {}'.format(task.upper()))
                print_accuracy([result[0][task] for result in results],
                               [result[1][task] for result in results])
        else:
            print_accuracy([result[0] for result in results],
                           [result[1] for result in results])

    output_dir = args.output_dir / Path('scores')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    save_scores(results, output_dir / Path(args.split+'.pkl'))


if __name__ == '__main__':
    main()
