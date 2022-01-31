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
    confs, odin_confs, energy_confs = [], [], []
    end = time.time()

    labels = []
    preds = []
    for b, (x, y) in enumerate(data_loader):

        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            labels.append(y.data.cpu().numpy())

            # measure data loading time
            chrono._done("eval load", time.time() - end)

            # compute output, measure accuracy and record loss.
            with chrono.measure("eval fprop"):
                logits = model(x)
                m = torch.nn.Softmax(dim=-1)
                conf, pred = torch.max(m(logits), dim=-1)
                confs.append(conf)
                preds.append(pred.data.cpu().numpy())

                odin_conf, _ = torch.max(m(logits / args.temperature_odin), dim=-1)
                odin_confs.append(odin_conf) 

                energy_conf = args.temperature_energy * torch.logsumexp(logits / args.temperature_energy, dim=1)
                energy_confs.append(energy_conf)
        if b % 10 == 0:
            print(f"run {b}/{len(data_loader)}")
            # break
    nat_correct = (np.concatenate(preds, axis=0) == np.concatenate(labels, axis=0)).astype(np.float32)
    print(nat_correct.sum() / len(data_loader.dataset))
    return confs, odin_confs, energy_confs


def run_eval(model, in_loader, out_loader, device, chrono, logger, args):
    # switch to evaluate mode
    model.eval()

    logger.info("Running validation...")
    logger.flush()

    _, _, in_energy_confs = iterate_data(in_loader, chrono, model, device, args)
    _, _, out_energy_confs = iterate_data(out_loader, chrono, model, device, args)

    # test energy
    in_confs = torch.cat(in_energy_confs, dim=-1)
    out_confs = torch.cat(out_energy_confs, dim=-1)

    in_examples = np.array(in_confs.cpu()).reshape((-1, 1))
    out_examples = np.array(out_confs.cpu()).reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    logger.info('============Results for Energy============')
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()


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
    parser.add_argument('--inference_k', default=90, type=int, help='')
    main(parser.parse_args())
