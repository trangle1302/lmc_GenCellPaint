import argparse, os, sys, datetime, glob, socket, subprocess
from contextlib import redirect_stderr, redirect_stdout

#from memory_profiler import profile
from omegaconf import OmegaConf
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from ldm.parse import get_parser, separate_args
from ldm.util import instantiate_from_config


# @profile
def main(opt, logdir, nowname):
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = trainer_config.get("accelerator", "ddp")
    nondefault_trainer_args, non_trainer_args = separate_args(opt)
    for k in nondefault_trainer_args:
        trainer_config[k] = getattr(opt, k)
    config_to_log = dict()
    for k in nondefault_trainer_args + non_trainer_args:
        config_to_log[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        assert (
            "," in trainer_config["gpus"]
        ), "Please specify GPUs as comma-separated list."
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    if hasattr(trainer_opt, "profiler"):
        if trainer_opt.profiler == "simple":
            trainer_opt.profiler = pl.profiler.SimpleProfiler(
                dirpath=logdir, filename="perf_logs"
            )
        elif trainer_opt.profiler == "advanced":
            trainer_opt.profiler = pl.profiler.AdvancedProfiler(
                dirpath=logdir, filename="perf_logs"
            )
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                # "offline": False,
                "id": nowname,
                "project": "super-multiplex-cell",
                "config": config_to_log,
                "resume": "allow",
            },
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            },
        },
    }
    default_logger_cfg = default_logger_cfgs["wandb"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        },
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse("1.4.0"):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "ldm.callbacks.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            },
        },
        "image_logger": {
            "target": "ldm.callbacks.ImageLogger",
            "params": {"batch_frequency": 750, "max_images": 4, "clamp": True},
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            },
        },
        "cuda_callback": {"target": "ldm.callbacks.CUDACallback"},
        "cpu_mem_monitor": {"target": "ldm.callbacks.CPUMemoryMonitor"},
    }
    if version.parse(pl.__version__) >= version.parse("1.4.0"):
        default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
        print(
            "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 10000,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if "ignore_keys_callback" in callbacks_cfg and hasattr(
        trainer_opt, "resume_from_checkpoint"
    ):
        callbacks_cfg.ignore_keys_callback.params[
            "ckpt_path"
        ] = trainer_opt.resume_from_checkpoint
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]

    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###
    print("Streaming bool:", opt.streaming)  # False, No streaming for this project

    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(str(lightning_config.trainer.gpus).strip(",").split(","))
    else:
        ngpu = 1
    if "accumulate_grad_batches" in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
            )
        )
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    if opt.train:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            print(
                    "Oops, the diffusion model training process has stopped unexpectedly"
                )
            raise
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)

    if trainer.global_rank == 0:
        print(trainer.profiler.summary())


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value
    __spec__ = None

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when running as `python main.py` (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    opt.now = now
    opt.hostname = socket.gethostname()
    opt.pid = os.getpid()
    opt.screen = subprocess.check_output("echo $STY", shell=True).decode("utf").strip()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(
            opt.logdir, "debug_logs" if opt.debug else "logs", nowname
        )
        os.makedirs(logdir)
    import wandb

    wandb.init(
        project="super-multiplex-cell",
        config=opt,
        resume="allow",
        settings=wandb.Settings(start_method="fork"),
        name=nowname,
        mode="offline" if opt.debug else "online",
        id=nowname,
    )
    print(logdir)
    print(opt)
    if opt.debug:
        main(opt, logdir, nowname)
    else:
        log_filename = os.path.join(logdir, f"{now}-log.txt")
        with open(log_filename, "w") as f:
            with redirect_stdout(f):
                with redirect_stderr(f):
                    print(f"Redirecting stdout and stderr to {log_filename}...")
                    print("A message to stdout...")
                    print("A message to stderr...", file=sys.stderr)
                    main(opt, logdir, nowname)
