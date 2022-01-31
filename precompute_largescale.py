from __future__ import print_function

from os.path import join as pjoin
import models.resnetv2 as resnetv2_models
import models.resnet as resnet_models
import torch
import time
import torchvision as tv
from util import hyperrule
from util import lbtoolbox as lb
import numpy as np
from util.tool_largescale import arg_parser, mk_id_ood, load_model, get_measures, setup_logger


def iterate_data(data_loader, chrono, model, device, args):
    confs, feat_list = [], []
    end = time.time()

    labels = []
    preds = []
    model = model.module
    for b, (x, y) in enumerate(data_loader):
        print(f"feat extraction {b}/{len(data_loader)}")
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            labels.append(y.data.cpu().numpy())

            # measure data loading time
            chrono._done("eval load", time.time() - end)

            # compute output, measure accuracy and record loss.
            with chrono.measure("eval fprop"):
                feat = model.before_head(model.body(model.root(x)))
                logits = model.head(feat).squeeze()

                m = torch.nn.Softmax(dim=-1)
                conf, pred = torch.max(m(logits), dim=-1)
                confs.append(conf)
                preds.append(pred.data.cpu().numpy())
                feat_list.append(feat.squeeze())


    feat_log = torch.cat(feat_list, dim=0)
    nat_correct = (np.concatenate(preds, axis=0) == np.concatenate(labels, axis=0)).astype(np.float32)
    print(nat_correct.sum() / len(data_loader.dataset))
    return confs, feat_log


def run_eval(model, in_loader, out_loader, device, chrono, logger, args):
    # switch to evaluate mode
    model.eval()

    logger.info("Running validation...")
    logger.flush()

    in_confs, feat_log = iterate_data(in_loader, chrono, model, device, args)

    w = model.module.head[0].weight.data.squeeze().cpu().numpy()
    v_in = feat_log.mean(0).data.cpu().numpy()[None, :] * w

    np.save(f"cache/{args.in_dataset}_{args.model}_wa_stat.npy", (v_in, ))
    print("done")

def main(args):
    logger = setup_logger(args)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Going to train on {device}")

    in_set, out_set, in_loader, out_loader = mk_id_ood(args, logger)

    model = load_model(args.model_type, args.model, args.model_path, logger, len(in_set.classes), args)

    # logger.info("Moving model onto all GPUs")
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    chrono = lb.Chrono()
    start_time = time.time()
    run_eval(model, in_loader, out_loader, device, chrono, logger, args)
    end_time = time.time()
    logger.info(f"Timings:\n{chrono}")

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')
    parser.add_argument('--inference_k', default=None, type=int, help='')
    main(parser.parse_args())

