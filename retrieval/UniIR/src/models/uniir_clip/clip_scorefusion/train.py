"""
Training Code for CLIP-SF
"""

# Standard library
import argparse
import logging
import os
import random

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
from dotenv import load_dotenv
import wandb

# Local modules or packages
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
    build_distributed_sampler_list,
    build_dataloader_list,
)
from models.uniir_clip.engine import train_one_epoch, eval_engine
from models.uniir_clip import utils
from clip_sf import CLIPScoreFusion

# Set up logger
logger = logging.getLogger()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def filter_parameters(model, condition_fn):
    named_parameters = model.named_parameters()
    return [p for n, p in named_parameters if condition_fn(n, p) and p.requires_grad]


def create_optimizer(gain_or_bias_params, rest_params, config):
    return optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": 0.2},
        ],
        lr=config.trainer_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1.0e-6,
    )


def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config):
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
    save_obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "scaler": scaler.state_dict(),
    }
    checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def log_results(train_stats, val_stats, test_stats, epoch=None, best_epoch=None):
    log_stats = {}
    if train_stats:
        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
    if val_stats:
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
    if test_stats:
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
    if epoch is not None:
        log_stats["epoch"] = epoch
    if best_epoch is not None:
        log_stats["best_epoch"] = best_epoch
    return log_stats


def train(
    train_loader,
    val_loader,
    model,
    model_without_ddp,
    optimizer,
    scheduler,
    scaler,
    config,
    epoch,
):
    gpu_id = config.dist_config.gpu_id
    is_distributed_mode = config.dist_config.distributed_mode
    global_step, total_loss, best_inbatch_accuracy = (
        0,
        0.0,
        0.0,
    )  # TODO: global_step is not used.
    best_epoch = 0
    model.zero_grad()

    if epoch != 0:
        print(f"Resuming training from epoch {epoch}")
    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        # Set different seed for different epoch
        if is_distributed_mode:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            gpu_id,
            scheduler,
            global_step,
            scaler,
            config,
        )

        eval_freq = config.evaluator.eval_freq
        if val_loader is None or epoch % eval_freq != 0:
            log_stats = log_results(train_stats, None, None, epoch, best_epoch)
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
        else:
            val_status = eval_engine(model, val_loader, gpu_id, config)
            try:
                inbatch_accuracy = float(val_status["inbatch_accuracy"])
            except ValueError:
                print(f"Error: Expected a number but got '{val_status['inbatch_accuracy']}'")
                inbatch_accuracy = 100.0
            # Note: still save the model even if the in-batch accuracy is not the best
            if utils.is_main_process():
                save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
            if inbatch_accuracy >= best_inbatch_accuracy:
                # if utils.is_main_process():
                #     save_checkpoint(model_without_ddp, optimizer, scheduler, epoch, scaler, config)
                best_inbatch_accuracy = inbatch_accuracy
                best_epoch = epoch
            log_stats = log_results(train_stats, val_status, None, epoch, best_epoch)

        if utils.is_main_process():
            # logger_out_dir = os.path.join(config.uniir_dir, config.logger_config.logger_out_dir)
            # logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
            # with open(logger_out_path, "a") as f:
            #     f.write(json.dumps(log_stats) + "\n")
            if config.wandb_config.enabled:
                wandb.log(log_stats)

        dist.barrier()  # Wait for the master process to finish writing the log file
        torch.cuda.empty_cache()


def main(config):
    is_distributed_mode = config.dist_config.distributed_mode

    # Set up seed for reproducibility
    seed = config.seed + utils.get_rank()
    set_seed(seed)

    cudnn.benchmark = True

    # Initialize and load model
    print("Creating CLIP-SF model...")
    model_config = config.model
    pretrained_clip_model_dir = os.path.join(config.uniir_dir, model_config.pretrained_clip_model_dir)
    logger.info(f"Downloading CLIP model to {pretrained_clip_model_dir}...")
    model = CLIPScoreFusion(
        model_name=model_config.clip_vision_model_name,
        download_root=pretrained_clip_model_dir,
        config=config,
    )
    model.float()  # The origial CLIP was in fp16 so we need to convert it to fp32

    # Set up optimizer, and scaler
    # Apply different optimization strategies to different parameters
    # This is adapted from the UniVL-DR codebase
    exclude_condition = lambda n, p: p.ndim < 2 or any(sub in n for sub in ["bn", "ln", "bias", "logit_scale"])
    include_condition = lambda n, p: not exclude_condition(n, p)
    gain_or_bias_params = filter_parameters(model, exclude_condition)
    rest_params = filter_parameters(model, include_condition)
    optimizer = create_optimizer(gain_or_bias_params, rest_params, config)
    scaler = GradScaler()  # Initialize the GradScaler

    # If resume training, load the checkpoint
    ckpt_config = model_config.ckpt_config
    if ckpt_config.resume_training:
        checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        logger.info(f"loading CLIPScoreFusion checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    # Move model to GPUs
    model.train()
    model = model.to(config.dist_config.gpu_id)
    model_without_ddp = model
    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id])
        model_without_ddp = model.module

    # Prepare datasets and dataloaders
    logger.info("Preparing dataset ...")  # Note printing only available in the main process
    logger.info(f"Loading dataset from {config.mbeir_data_dir}{config.data_config.train_query_data_path}...")

    img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()
    tokenizer = model_without_ddp.get_tokenizer()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_dataset, train_collector = build_mbeir_dataset_from_config(
        config=config,
        tokenizer=tokenizer,
        img_preprocess_fn=img_preprocess_fn,
        dataset_type=DatasetType.MAIN_TRAIN,
    )
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader_config.train_batch_size,
        num_workers=config.dataloader_config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,  # Note: since we use sampler, shuffle should be False
        collate_fn=train_collector,
        drop_last=True,
    )

    enable_eval = config.evaluator.enable_eval
    valid_loader = None
    if enable_eval:
        in_batch_val_dataset, in_batch_val_collector = build_mbeir_dataset_from_config(
            config=config,
            tokenizer=tokenizer,
            img_preprocess_fn=img_preprocess_fn,
            dataset_type=DatasetType.IN_BATCH_VAL,
        )
        in_batch_val_sampler = DistributedSampler(
            dataset=in_batch_val_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset=in_batch_val_dataset,
            batch_size=config.dataloader_config.valid_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=True,
            sampler=in_batch_val_sampler,
            shuffle=False,  # Note: since we use sampler, shuffle should be False
            collate_fn=in_batch_val_collector,
            drop_last=True,
        )
    else:
        print("In-batch validation is disabled.")

    # Initializing the scheduler
    t_total = (
        len(train_loader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_total, eta_min=0)

    epoch = 0
    if ckpt_config.resume_training:
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"] + 1

    # Training loop
    dist.barrier()
    train(
        train_loader,
        valid_loader,
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        config,
        epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--uniir_dir",
        type=str,
        default="/data/UniIR",
        help="Path to UniIR directory to save checkpoints, embeddings, etc.",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data",
        help="Path to mbeir dataset directory",
    )
    args = parser.parse_args()
    print(f"Loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    # Initialize distributed training
    args.dist_url = config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    # Set up wandb
    if config.wandb_config.enabled and utils.is_main_process():
        load_dotenv()  # Load .env and get WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY
        wandb_key = os.environ.get("WANDB_API_KEY")
        wandb_project = os.environ.get("WANDB_PROJECT")
        wandb_entity = os.environ.get("WANDB_ENTITY")

        if not wandb_key:
            raise ValueError("WANDB_API_KEY not found. Ensure it's set in the .env file.")

        wandb.login(key=wandb_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=config.wandb_config.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    # Set up logger
    if utils.is_main_process():
        logger_out_dir = os.path.join(config.uniir_dir, config.logger_config.logger_out_dir)
        logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
        if not os.path.exists(logger_out_dir):
            os.makedirs(logger_out_dir, exist_ok=True)
        handlers = [logging.FileHandler(logger_out_path), logging.StreamHandler()]
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s",
            level=logging.DEBUG,
            datefmt="%d-%m-%Y %H:%M:%S",
            handlers=handlers,
        )
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logger = logging.getLogger(__name__)
        logger.info(config)

    main(config)

    # Close wandb
    if config.wandb_config.enabled and utils.is_main_process():
        wandb.finish()

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
