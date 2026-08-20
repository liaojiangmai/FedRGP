import argparse
import torch
import numpy as np
from Dassl.dassl.utils import setup_logger, set_random_seed, collect_env_info
from Dassl.dassl.config import get_cfg_default
import os
import gc
from federated_core.factory import FederatedLearnerFactory

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed is not None:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.num_shots:
        cfg.DATASET.NUM_SHOTS = args.num_shots

    if args.subsample:
        cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample
        
    if args.debug_mode is not None:
        cfg.DEBUG_MODE = args.debug_mode


def extend_cfg(cfg):
    """Add FedRGP-specific configuration defaults."""
    from yacs.config import CfgNode as CN

    cfg.DEBUG_MODE = False

    cfg.TRAINER.FEDRGP = CN()
    cfg.TRAINER.FEDRGP.N_CTX_VISION = 8
    cfg.TRAINER.FEDRGP.N_CTX_TEXT = 8
    cfg.TRAINER.FEDRGP.NUM_PROMPTS_VISION = 3
    cfg.TRAINER.FEDRGP.NUM_PROMPTS_TEXT = 3
    cfg.TRAINER.FEDRGP.CTX_INIT = ""
    cfg.TRAINER.FEDRGP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.FEDRGP.PROMPT_DEPTH_VISION = 1  # 1 for shallow prompting
    cfg.TRAINER.FEDRGP.PROMPT_DEPTH_TEXT = 1   # 1 for shallow prompting
    cfg.TRAINER.FEDRGP.USE_DIVERGENT_LOSS = False
    cfg.TRAINER.FEDRGP.DIVERGENT_LOSS_WEIGHT = 0.1
    cfg.TRAINER.FEDRGP.DIVERGENT_LOSS_TYPE = "l2"  # cos, l1, l2
    cfg.TRAINER.FEDRGP.TOPK = 2  # top-k prompts for aggregation
    cfg.TRAINER.FEDRGP.AGGREGATE_HIGHEST_SIM = False  # True: highest similarity, False: lowest
    cfg.TRAINER.FEDRGP.SELECT_MODE = "similarity"  # similarity, random, top_fixed, all
    cfg.TRAINER.FEDRGP.INFERENCE_MODE = "average"  # average, selected_group, max_logits, feature_average
    cfg.TRAINER.FEDRGP.SELECTED_PROMPT_GROUP = 0  # only used when INFERENCE_MODE=selected_group
    cfg.TRAINER.FEDRGP.RANDOM_SEED = 42
    cfg.TRAINER.FEDRGP.PERTURBATION_ENABLED = False  # add perturbation if same prompt selected 3 rounds
    cfg.TRAINER.FEDRGP.PROBABILISTIC_SELECTION = False  # probability sampling based on similarity
    cfg.TRAINER.FEDRGP.TEMPERATURE = 1.0  # softmax temperature for probability distribution
    # FedGVP additions
    cfg.TRAINER.FEDRGP.APG_ENABLED = False
    cfg.TRAINER.FEDRGP.APG_INIT_PROMPTS = 1
    cfg.TRAINER.FEDRGP.APG_MAX_PROMPTS = 5
    cfg.TRAINER.FEDRGP.APG_THRESHOLD = 0.3
    cfg.TRAINER.FEDRGP.GVG_ENABLED = False
    cfg.TRAINER.FEDRGP.GVG_LAMBDA = 0.5
    cfg.TRAINER.FEDRGP.ORTHO_WEIGHT = 0.0
    cfg.TRAINER.FEDRGP.RGP_ENABLED = True
    cfg.TRAINER.FEDRGP.RGP_TAU = 1.0
    cfg.TRAINER.FEDRGP.RGP_CONSIST_WEIGHT = 0.5
    cfg.TRAINER.FEDRGP.RGP_GLOBAL_MIX = 1.0

    cfg.DATASET.SUBSAMPLE_CLASSES = "base"  # all, base or new
    cfg.DATASET.USERS = 10  # number of clients
    cfg.DATASET.IID = False  # is iid
    cfg.DATASET.FRAC = 1.0
    cfg.DATASET.PARTITION = "noniid-labeldir"
    cfg.DATASET.USEALL = False # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = 4
    cfg.DATASET.BETA = 0.3
    cfg.DATASET.REPEATRATE = 0.0 # repeat rate on each client
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = 4 # number of domain
    cfg.DATASET.IMBALANCE_TRAIN = False # is adding label skew to feature skew datasets
    cfg.DATASET.SPLIT_CLIENT = False # is adding label skew to feature skew datasets and split one domain to multi clients
    # Noisy label settings (federated train only)
    cfg.DATASET.LABEL_NOISE_RATIO = 0.0
    cfg.DATASET.LABEL_NOISE_TYPE = "symmetric"  # symmetric, pairflip
    cfg.DATASET.LABEL_NOISE_SEED = 0
    cfg.DATASET.LABEL_NOISE_PER_CLIENT = True
    cfg.OPTIM.ROUND = 10 # global round
    cfg.OPTIM.MAX_EPOCH = 2 # local epoch
    cfg.OPTIM.GAMMA = 1 # gamma of single-step
    cfg.OPTIM.LR = 0.001 #learning rate

    cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    federated_learner = FederatedLearnerFactory.create_learner(args.model, cfg, args)

    if args.eval_only:
        if cfg.DATASET.SUBSAMPLE_CLASSES not in ['base', 'new', 'all']:
            print(f"Warning: SUBSAMPLE_CLASSES='{cfg.DATASET.SUBSAMPLE_CLASSES}', should be 'base', 'new', or 'all'")
            if args.subsample:
                print(f"Using subsample from args: '{args.subsample}'")
                cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample
            else:
                print("Defaulting to 'new' for test classes")
                cfg.DATASET.SUBSAMPLE_CLASSES = 'new'

        print(f"Testing on {cfg.DATASET.SUBSAMPLE_CLASSES} classes")
        federated_learner.test()
    else:
        federated_learner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ================================== model and trainer ==================================
    parser.add_argument("--model", type=str, default="FedRGP", help="federated model (FedRGP)")
    parser.add_argument("--trainer", type=str, default="FedRGP", help="trainer name (FedRGP)")

    # ================================== param for vision backbone ================================== 
    parser.add_argument("--config-file", type=str, default="configs/trainers/FedRGP/base2novel_vit_b16.yaml", help="path to config file")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")

    # ================================== param for dataset ================================== 
    parser.add_argument("--root", type=str, default="DATA/", help="path to dataset")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/oxford_pets.yaml", help="path to config file for dataset setup")

    # ================================== feaderal experimental settings ================================== 
    parser.add_argument("--seed", type=int, default=0, help="only positive value enables a fixed seed")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--output-dir", type=str, default="outputtest/", help="output directory")
    parser.add_argument("--debug-mode", type=lambda x: (str(x).lower() == 'true'), default=None, help="Run test only in last round to speed up training")

    # ================================== data partition ================================== 
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")
    parser.add_argument('--num_shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--subsample', type=str, default='base', help="all,base,new")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")

    # ================================== evaluation settings ================================== 
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    
    args = parser.parse_args()
    main(args)




