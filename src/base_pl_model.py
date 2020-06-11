import os
import os.path as osp
import sys
import yaml
import json
import shutil
import random
from dateutil import tz
from datetime import datetime as dt

from pytorch_lightning.utilities.distributed import rank_zero_only
import pytorch_lightning as pl

from config import mkdir_p

_plt = None


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class BasePLModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()

        # self.net = Net()
        self.hparams = hparams
        self.cfg = hparams
        # if hparams.resume:
        #     self.load_from_checkpoint(hparams.resume)
        self.cfg = hparams
        self.comet_set = False

    @property
    def use_cuda(self):
        return self.cfg.gpus is not None and self.cfg.gpus != 0

    def on_epoch_start(self):
        if not self.comet_set:
            if self.cfg.logcomet:
                self.set_comet_graph()
                self.log_src_comet()
                self.log_cfg_comet()
                self.print("\nSent src and config to comet")
            self.comet_set = True  # Avoiding calling each time epoch starts

        return super().on_epoch_start()

    @property
    def comet_logger(self):
        comet_logger = getattr(self, "_comet_logger", None)
        if comet_logger:
            return comet_logger
        for logger in self.logger:
            if isinstance(logger, pl.loggers.CometLogger):
                self._comet_logger = logger
                return logger

    @rank_zero_only
    def set_comet_graph(self):
        if self.cfg.logcomet:
            self.comet_logger.experiment.set_model_graph(str(self))

    @rank_zero_only
    def log_src_comet(self):
        if self.cfg.logcomet:
            self.comet_logger.experiment.log_asset_folder(
                osp.join(self.work_dir, "src"), recursive=True, log_file_name=True
            )

    @rank_zero_only
    def log_cfg_comet(self):
        if self.cfg.logcomet:
            self.comet_logger.experiment.log_asset(osp.join(self.exp_dir, "cfg.json"))
            self.comet_logger.experiment.log_asset(osp.join(self.exp_dir, "cfg.yml"))

    @rank_zero_only
    def log_figure(self, figure, name, step=None, close=True):
        if self.logger:
            for logger in self.logger:
                if type(logger) is pl.loggers.TensorBoardLogger:
                    logger.experiment.add_figure(
                        name, figure, global_step=step, close=False
                    )
                elif type(logger) is pl.loggers.CometLogger:
                    logger.experiment.log_figure(name, figure, step=step)

        if close:
            global _plt
            if _plt is None:
                import matplotlib.pyplot as plt

                _plt = plt
            plt = _plt
            plt.close(figure)

    def make_lightning_loggers_ckpt(self, ckpt_callback_kwargs=dict(
            # filepath=osp.join(self.exp_dir, "checkpoints/"),
            monitor="val_loss",
            verbose=True,
            save_top_k=2,

    )):
        loggers = [pl.loggers.TensorBoardLogger(save_dir=self.exp_dir, name="",)]
        if self.cfg.logcomet:
            comet_logger = pl.loggers.CometLogger(
                api_key=os.environ["COMET_API_KEY"],
                workspace=self.cfg.comet_workspace,
                project_name=self.cfg.comet_project_name,
                experiment_name=self.run_name,
            )
            loggers.append(comet_logger)

        default_ckpt_callback_kwargs = {
            'filepath': osp.join(self.exp_dir, "checkpoints/"),
        }
        default_ckpt_callback_kwargs.update(ckpt_callback_kwargs)
        ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            **default_ckpt_callback_kwargs,
        )
        return loggers, ckpt_callback


    @rank_zero_only
    def init_log(self):
        now = dt.now(tz.tzlocal())
        now = now.strftime("%m-%d-%Y-%H-%M-%S")
        log_dir = self.cfg.exp_name
        run_name = self.cfg.run_name
        if run_name == "" or run_name is None:
            run_name = now
        self.run_name = run_name
        work_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
        self.work_dir = work_dir
        exp_dir = osp.join(work_dir, "experiments", log_dir, run_name)
        _exp_dir = exp_dir
        i = 2
        while os.path.exists(_exp_dir):
            _exp_dir = f"{exp_dir}-{i}"
            i += 1
        exp_dir = _exp_dir

        self.exp_dir = exp_dir
        shutil.copytree(
            osp.join(work_dir, "src"),
            osp.join(exp_dir, "src"),
            ignore=shutil.ignore_patterns(".*", "__pycache__", ".DS_Store"),
        )

        self.logfile = osp.join(exp_dir, "logfile.log")
        sys.stdout = Logger(self.logfile)

        with open(osp.join(self.exp_dir, "cfg.json"), "w") as f:
            json.dump(self.cfg, f, indent=4)
        with open(osp.join(self.exp_dir, "cfg.yml"), "w") as f:
            yaml.dump(json.loads(json.dumps(self.cfg)), f, indent=4)

        # print(exp_dir)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
