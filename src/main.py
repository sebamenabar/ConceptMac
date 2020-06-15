import os
import sys
import os.path as osp
from pprint import PrettyPrinter as PP

from easydict import EasyDict as edict
from dotenv import load_dotenv

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pl_model import PLModel
from base_pl_model import Logger, ErrLogger
from config import __C, parse_args_and_set_config

load_dotenv()


if __name__ == "__main__":
    args, cfg = parse_args_and_set_config(__C, blacklist=["gradient_clip_val"])
    pp = PP(indent=4)

    cfg.train.accumulate_grad_batches = args.accumulate_grad_batches
    model = PLModel(cfg)

    # Prints should be done after the init log
    model.init_log(vars(args))
    try:
        _stdout = sys.stdout
        _stderr = sys.stderr
        with open(osp.join(model.exp_dir, "logfile.log"), "a") as log:
            sys.stdout = Logger(log)
            sys.stderr = ErrLogger(log)
            pl._logger.addHandler(pl.python_logging.StreamHandler(sys.stdout))

            if torch.cuda.is_available():
                print("GPUS:", os.environ["CUDA_VISIBLE_DEVICES"])
                print(torch.cuda.get_device_name())
            print(pp.pformat(vars(args)))
            print(pp.pformat(cfg))

            loggers = model.make_lightning_loggers()
            default_ckpt_callback_kwargs = {
                "filepath": osp.join(model.exp_dir, "checkpoints/"),
                "monitor": "val_uni_acc",
                "verbose": True,
                "save_top_k": 2,
            }
            ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
                **default_ckpt_callback_kwargs,
            )
            trainer = pl.Trainer.from_argparse_args(
                args,
                logger=loggers,
                checkpoint_callback=ckpt_callback,
                max_epochs=cfg.train.epochs,
                default_root_dir=model.exp_dir,
                gradient_clip_val=cfg.train.gradient_clip_val,
            )
            if args.eval:
                pass
            elif args.test:
                pass
            else:
                pass
                trainer.fit(model)
    finally:
        sys.stdout = _stdout
        sys.stderr = _stderr
