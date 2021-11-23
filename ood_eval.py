from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
import models.densenet as dn
import util.svhn_loader as svhn
import numpy as np
import time
from util.metrics import compute_traditional_ood


from util.score import get_score
from models.resnet import resnet18_cifar, resnet50_cifar, resnet101_cifar
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--in-dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
parser.add_argument('--name', default="densenet", type=str,
                    help='neural network name and training set')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')
parser.add_argument('--p', default=None, type=int, help='sparsity level')

parser.add_argument('--gpu', default = '0', type = str,
		    help='gpu index')

parser.add_argument('--in-dist-only', help='only evaluate in-distribution', action='store_true')
parser.add_argument('--out-dist-only', help='only evaluate out-distribution', action='store_true')

parser.add_argument('--method', default='energy', type=str, help='odin mahalanobis')
parser.add_argument('--cal-metric', help='calculatse metric directly', action='store_true')

parser.add_argument('--epsilon', default=8.0, type=float, help='epsilon')
parser.add_argument('--iters', default=40, type=int,
                    help='attack iterations')
parser.add_argument('--iter-size', default=1.0, type=float, help='attack step size')

parser.add_argument('--severity-level', default=5, type=int, help='severity level')

parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=50, type=int,
                    help='mini-batch size')

parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')

parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--depth', default=40, type=int,
                    help='depth of resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')

parser.set_defaults(argument=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def eval_ood_detector(args, mode_args):
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    out_datasets = args.out_datasets
    batch_size = args.batch_size
    method = args.method
    method_args = args.method_args
    name = args.name
    epochs = args.epochs

    in_save_dir = os.path.join(base_dir, in_dataset, method, name, 'nat')
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)


    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if in_dataset == "CIFAR-10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        num_classes = 10

    elif in_dataset == "CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        num_classes = 100

    method_args['num_classes'] = num_classes

    if args.model_arch == 'densenet':
        info = np.load(f"cache/{args.in_dataset}_{args.model_arch}_wa_stat.npy")
        model = dn.DenseNet3(args.layers, num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None, k=args.p,info=info)
    elif args.model_arch == 'resnet18':
        model = resnet18_cifar(num_classes=num_classes, k=args.p, info=None)
    elif args.model_arch == 'resnet50':
        model = resnet50_cifar(num_classes=num_classes, k=args.p)
    elif args.model_arch == 'resnet101':
        model = resnet101_cifar(num_classes=num_classes, k=args.p)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=in_dataset, name=name, epochs=epochs))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.cuda()

    if method == "mahalanobis":
        temp_x = torch.rand(2,3,32,32)
        temp_x = Variable(temp_x).cuda()
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        method_args['num_output'] = num_output


    if not mode_args['out_dist_only']:
        t0 = time.time()

        f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

    ########################################In-distribution###########################################
        # print("Processing in-distribution images")

        N = len(testloaderIn.dataset)
        count = 0
        for j, data in enumerate(testloaderIn):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images

            scores = get_score(inputs, model, method, method_args)

            for score in scores:
                f1.write("{}\n".format(score))

            if method == "rowl":
                outputs = F.softmax(model(inputs), dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)
            else:
                outputs = F.softmax(model(inputs)[:, :num_classes], dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)

            for k in range(preds.shape[0]):
                g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            count += curr_batch_size
            # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f1.close()
        g1.close()

    if mode_args['in_dist_only']:
        return

    for out_dataset in out_datasets:

        out_save_dir = os.path.join(in_save_dir, out_dataset)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        if out_dataset == 'SVHN':
            testsetout = svhn.SVHN('datasets/ood_datasets/svhn/', split='test', transform=transform_test, download=False)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=2)
        elif out_dataset == 'dtd':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images", transform=transform_test)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'places365':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365/", transform=transform_test)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'CIFAR-100':
            testsetout = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'celebA':
            testsetout = torchvision.datasets.ImageFolder(root="/media/sunyiyou/ubuntu-hdd1/dataset/celebA/small", transform=transform_test)
            # testsetout = torchvision.datasets.CelebA(root='datasets/ood_datasets/', split='test', download=True, transform=transform_test)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=True, num_workers=2)
        else:
            testsetout = torchvision.datasets.ImageFolder("./datasets/ood_datasets/{}".format(out_dataset), transform=transform_test)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=2)

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        # print("Processing out-of-distribution images")

        N = len(testloaderOut.dataset)
        count = 0
        for j, data in enumerate(testloaderOut):

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            curr_batch_size = images.shape[0]

            inputs = images

            scores = get_score(inputs, model, method, method_args)

            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time()-t0))
            t0 = time.time()

        f2.close()
    return

if __name__ == '__main__':
    args.method_args = dict()
    mode_args = dict()

    mode_args['in_dist_only'] = args.in_dist_only
    mode_args['out_dist_only'] = args.out_dist_only

    args.out_datasets = ['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']  #['SVHN', 'CIFAR-100', 'celebA']

    if args.method == 'msp':
        eval_ood_detector(args, mode_args)
    elif args.method == "odin":
        args.method_args['temperature'] = 1000.0
        param_dict = {
            "CIFAR-10": {
                "wideresnet": 0.0,
                "densenet": 0.018,
                "resnet50": 0.04,
            },
            "CIFAR-100": {
                "wideresnet": 0.024,
                "densenet": 0.024,
                "resnet50": 0.034,
            }
        }
        args.method_args['magnitude'] = param_dict[args.in_dataset][args.model_arch]
        eval_ood_detector(args, mode_args)
    elif args.method == 'mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'results.npy'), allow_pickle=True)
        regressor = LogisticRegressionCV(cv=2).fit([[0,0,0,0],[0,0,0,0],[1,1,1,1],[1,1,1,1]], [0,0,1,1])
        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias
        args.method_args['sample_mean'] = sample_mean
        args.method_args['precision'] = precision
        args.method_args['magnitude'] = magnitude
        args.method_args['regressor'] = regressor
        eval_ood_detector(args, mode_args)
    elif args.method == 'sofl':
        eval_ood_detector(args, mode_args)
    elif args.method == 'rowl':
        eval_ood_detector(args, mode_args)
    elif args.method == 'atom':
        eval_ood_detector(args, mode_args)
    elif args.method == 'energy':
        args.method_args['temperature'] = 1000.0
        eval_ood_detector(args, mode_args)

    compute_traditional_ood(args.base_dir, args.in_dataset, args.out_datasets, args.method, args.name)
