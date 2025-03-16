"""
This module generates embeddings for MBEIR with multiple GPUs.
"""

import os
import argparse
from omegaconf import OmegaConf
import tqdm
import gc
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.cuda.amp import autocast
import transformers

import dist_utils
from dist_utils import ContiguousDistributedSampler
from utils import build_model_from_config, set_seed
import sys
sys.path.append("/home/du/pan1/new_knowledge/real_new/mmrag/UniIR/src")
from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolDataset,
    MBEIRCandidatePoolCollator,
    Mode,
)


@torch.no_grad()
def generate_embeds_and_ids_for_dataset_with_gather(model, data_loader, device, use_fp16=True):
    embedding_tensors = []
    id_list = []

    total_cores = os.cpu_count()
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        initial_threads_per_process = total_cores // world_size
        torch.set_num_threads(initial_threads_per_process)
        data_loader = tqdm.tqdm(data_loader, desc=f"Rank {rank}")
    for batch in data_loader:
        # Used in combination with pin_memory=True
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    batch[key][k] = v.to(device)
        # Enable autocast to FP16
        with autocast(enabled=use_fp16):
            embeddings_batched, ids_list_batched = model(batch, encode_mbeir_batch=True)

        embedding_tensors.append(embeddings_batched.half())  # We only save FP16 embeddings to save space.
        id_list.extend(ids_list_batched)

    # Convert list of tensors to a single tensor
    embedding_tensor = torch.cat(embedding_tensors, dim=0)
    embedding_list = None

    if dist.is_initialized():
        # First, share the sizes of tensors across processes
        size_tensor = torch.tensor([embedding_tensor.size(0)], dtype=torch.long, device=device)

        # Allocate tensors to gather results on rank 0
        if dist.get_rank() == 0:
            # Allocate tensors with the correct sizes on rank 0
            gathered_embeddings = [torch.empty_like(embedding_tensor) for _ in range(dist.get_world_size())]
            id_list_gathered = [list() for _ in range(dist.get_world_size())]
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
        else:
            gathered_embeddings = None
            id_list_gathered = None
            sizes = None

        # Synchronize all processes before gathering data
        dist.barrier()

        # Gather embeddings from all processes
        dist.gather(embedding_tensor, gather_list=gathered_embeddings, dst=0)
        # Gather ids from all processes
        dist.gather_object(id_list, object_gather_list=id_list_gathered, dst=0)
        # Gather sizes from all processes
        dist.gather(size_tensor, gather_list=sizes, dst=0)

        # Synchronize all processes after gathering data
        dist.barrier()

        # On the main process, concatenate the results
        if dist.get_rank() == 0:
            print(f"Embedder Log: Gathered embeddings and ids on rank 0, starting to process them...")

            # Increase number of threads for rank 0 during conversion
            torch.set_num_threads(total_cores)

            # Trim the last tensors based on the gathered sizes
            gathered_embeddings[-1] = gathered_embeddings[-1][: sizes[-1][0]]
            embedding_list = torch.cat(gathered_embeddings, dim=0)

            # Flatten gathered ids
            id_list = [id for sublist in id_list_gathered for id in sublist]
            assert len(id_list) == embedding_list.size(0)
            # Check unique ids
            assert len(id_list) == len(set(id_list)), "Hashed IDs should be unique"
            print(f"Embedder Log: Finished processing embeddings and ids on rank 0.")

            # Note: we are using float16 to save space, and the precision loss is negligible.
            embedding_list = embedding_list.half().cpu().numpy()
            print(f"Embedder Log: Converted embedding_list to cpu numpy array of type {embedding_list.dtype}.")

            # Reset number of threads to initial value after conversion
            torch.set_num_threads(initial_threads_per_process)

        dist.barrier()  # Wait for rank 0 to process the embeddings and ids.
    else:
        embedding_list = embedding_tensor.half().cpu().numpy()

    return embedding_list, id_list


@torch.no_grad()
def generate_embeds_and_ids_for_dataset_with_tmp_files(
    model, data_loader, embed_dir, dataset_name, split_name, device, use_fp16=True
):
    embedding_tensors = []
    hashed_id_list = []

    if dist.is_initialized():
        rank = dist.get_rank()
        data_loader = tqdm.tqdm(data_loader, desc=f"Rank {rank}")

    for batch in data_loader:
        # Used in combination with pin_memory=True
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Enable autocast to FP16
        with autocast(enabled=use_fp16):
            embeddings_batched, hashed_ids_list_batched = model(batch, encode_mbeir_batch=True)

        embedding_tensors.append(embeddings_batched)
        hashed_id_list.extend(hashed_ids_list_batched)

    # Convert list of tensors to a single tensor
    embedding_tensor = torch.cat(embedding_tensors, dim=0)
    assert embedding_tensor.size(0) == len(hashed_id_list), "embeddings and ids must have the same size."

    # Save the embeddings and ids to .npy on each GPU
    all_embeddings = []
    all_ids = []
    if dist.is_initialized():
        gpu_id = device
        embed_tmp_data_name = f"mbeir_{dataset_name}_{split_name}_embed_gpu_{gpu_id}.npy"
        embed_tmp_id_name = f"mbeir_{dataset_name}_{split_name}_ids_gpu_{gpu_id}.npy"
        embed_tmp_data_path = os.path.join(embed_dir, embed_tmp_data_name)
        embed_tmp_id_path = os.path.join(embed_dir, embed_tmp_id_name)

        # Note: we are using float16 to save space, and the precision loss is negligible.
        np.save(embed_tmp_data_path, embedding_tensor.half().cpu().numpy())
        np.save(embed_tmp_id_path, hashed_id_list)

        print(f"Saved embeddings to {embed_tmp_data_path} and ids to {embed_tmp_id_path}.")
        dist.barrier()  # Ensure every process has finished saving.

        if dist.get_rank() == 0:
            print("Loading tmp embeddings and ids on rank 0...")

            for gpu_id in range(dist.get_world_size()):
                embed_data_name = f"mbeir_{dataset_name}_{split_name}_embed_gpu_{gpu_id}.npy"
                embed_id_name = f"mbeir_{dataset_name}_{split_name}_ids_gpu_{gpu_id}.npy"  # Loading IDs using .npy

                embed_path = os.path.join(embed_dir, embed_data_name)
                id_path = os.path.join(embed_dir, embed_id_name)

                embeddings = np.load(embed_path)
                all_embeddings.append(embeddings)

                ids = np.load(id_path)
                all_ids.extend(ids)

            all_embeddings = np.concatenate(all_embeddings, axis=0)
            assert len(all_embeddings) == len(all_ids), "Mismatch between embeddings and IDs length."
            print(f"Finished processing tmp embeddings and ids on rank 0.")

        dist.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.
    else:
        all_embeddings = embedding_tensor.half().cpu().numpy()
        all_ids = hashed_id_list

    return all_embeddings, all_ids


@torch.no_grad()
def generate_embeds_for_config(model, img_preprocess_fn, tokenizer, config):
    """This script generates embeddings for the queries and candidates(doc)"""
    uniir_dir = config.uniir_dir
    mbeir_data_dir = config.mbeir_data_dir
    embed_config = config.embed_config
    embed_dir_name = embed_config.embed_dir_name
    expt_dir_name = config.experiment.path_suffix

    # Config for dataset
    data_config = config.data_config
    query_instruct_path = data_config.query_instruct_path
    cand_pool_dir = data_config.cand_pool_dir_name
    image_size = tuple(map(int, data_config.image_size.split(",")))

    splits = []
    # Load the dataset splits to embed
    dataset_types = ["train", "val", "test"]
    for split_name in dataset_types:
        split_dir_name = getattr(data_config, f"{split_name}_dir_name")
        embed_dataset_config = getattr(embed_config, f"{split_name}_datasets_config", None)
        if embed_dataset_config and embed_dataset_config.enable_embed:
            dataset_name_list = getattr(embed_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(embed_dataset_config, "correspond_cand_pools_name", None)
            splits.append((split_name, split_dir_name, dataset_name_list, cand_pool_name_list))
            assert len(dataset_name_list) == len(cand_pool_name_list), "Mismatch between datasets and candidate pools."

    # Load the candidate pool to embed
    embed_cand_pool_config = embed_config.cand_pools_config
    if embed_cand_pool_config and embed_cand_pool_config.enable_embed:
        split_name = "cand_pool"
        split_dir_name = data_config.cand_pool_dir_name
        cand_pool_name_list = embed_cand_pool_config.cand_pools_name_to_embed
        splits.append(
            (
                split_name,
                split_dir_name,
                [None] * len(cand_pool_name_list),
                cand_pool_name_list,
            )
        )

    # Pretty Print dataset and candidate pool to embed
    if dist_utils.is_main_process():
        print("-" * 30)
        for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
            if split_name == "cand_pool":
                print(f"Split: {split_name}, Split dir: {split_dir}, Candidate pools to embed: {cand_pool_name_list}")
            else:
                print(f"Split: {split_name}, Split dir: {split_dir}, Datasets to embed: {dataset_name_list}")
            print("-" * 30)

    # Generate embeddings
    for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
        for dataset_name, cand_pool_name in zip(dataset_name_list, cand_pool_name_list):
            if split_name == "cand_pool":
                cand_pool_name = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{cand_pool_name}_{split_name}.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"\nEmbedder Log: Generating embeddings for {cand_pool_data_path}...")
                    print_config = True
                # TODO: refactor this and use build dataset from Config.
                dataset = MBEIRCandidatePoolDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    cand_pool_data_path=cand_pool_data_path,
                    img_preprocess_fn=img_preprocess_fn,
                    print_config=print_config,
                )
                collator = MBEIRCandidatePoolCollator(
                    tokenizer=tokenizer,
                    image_size=image_size,
                )
            else:  # "train" or "val" or "test"
                # Construct query data path
                dataset_name = dataset_name.lower()
                query_data_name = f"mbeir_{dataset_name}_{split_name}.jsonl"
                query_data_path = os.path.join(split_dir, query_data_name)

                # Construct the candidate pool path
                cand_pool_name = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{cand_pool_name}_cand_pool.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"\nEmbedder Log: Generating embeddings for {query_data_path} with {cand_pool_data_path}...")
                    print_config = True
                mode = Mode.EVAL
                dataset = MBEIRMainDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    query_data_path=query_data_path,
                    cand_pool_path=cand_pool_data_path,
                    query_instruct_path=query_instruct_path,
                    img_preprocess_fn=img_preprocess_fn,
                    mode=mode,
                    enable_query_instruct=data_config.enable_query_instruct,
                    shuffle_cand=data_config.shuffle_cand,
                    print_config=print_config,
                )
                collator = MBEIRMainCollator(
                    tokenizer=tokenizer,
                    image_size=image_size,
                    mode=mode,
                )

            # Config for data loader
            batch_size = config.dataloader_config.batch_size
            num_workers = config.dataloader_config.num_workers

            # Set up distributed data parallel
            num_tasks = dist_utils.get_world_size()
            global_rank = dist_utils.get_rank()
            sampler = ContiguousDistributedSampler(
                dataset,
                num_replicas=num_tasks,
                rank=global_rank,
            )  # Note: assume the dataset is in sorted order.
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=False,  # Since we have distributed sampler, we don't need to shuffle the data here.
                collate_fn=collator,
                drop_last=False,
            )
            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.
            if dist_utils.is_main_process():
                if split_name == "cand_pool":
                    print(f"Embedder Log: Data loader for {cand_pool_data_path} is set up.")
                    print(f"Embedder Log: Generating embeddings for {cand_pool_data_path}...")
                else:
                    print(
                        f"Embedder Log: Data loader for {query_data_path} with candidate pool {cand_pool_data_path} is set up."
                    )
                    print(f"Embedder Log: Generating embeddings for {query_data_path} ...")
                print(f"Inference with half precision: {config.embed_config.use_fp16}")

            # Generate embeddings and ids
            embedding_list, id_list = generate_embeds_and_ids_for_dataset_with_gather(
                model,
                data_loader,
                device=config.dist_config.gpu_id,
                use_fp16=config.embed_config.use_fp16,
            )

            # Save the embeddings and ids to .npy
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Embedder Log: Embedding list length: {len(embedding_list)}")
                print(f"Embedder Log: ID list length: {len(id_list)}")

                mid_name = cand_pool_name if split_name == "cand_pool" else dataset_name
                # Save the embeddings to .npy
                embed_data_name = f"mbeir_{mid_name}_{split_name}_embed.npy"
                embed_path = os.path.join(
                    uniir_dir,
                    embed_dir_name,
                    expt_dir_name,
                    split_name,
                    embed_data_name,
                )
                os.makedirs(os.path.dirname(embed_path), exist_ok=True)
                np.save(embed_path, embedding_list)
                print(f"Embedder Log: Saved embeddings to {embed_path}.")

                # Save the IDs to .npy
                id_data_name = f"mbeir_{mid_name}_{split_name}_ids.npy"
                id_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, id_data_name)
                os.makedirs(os.path.dirname(id_path), exist_ok=True)
                np.save(id_path, id_list)
                print(f"Embedder Log: Saved ids to {id_path}.")

            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.

            # Delete the embeddings and IDs to free up memory
            del embedding_list
            del id_list
            del data_loader
            del dataset
            del collator
            del sampler

            # Explicitly call the garbage collector
            gc.collect()
            torch.cuda.empty_cache()

        # Union pool embeddings
        if split_name == "cand_pool" and embed_cand_pool_config.embed_union_pool:
            # To efficiently generate embeddings for the union(global) pool,
            # We concat previously saved embeddings and ids from single(local) pool
            # Instead of embed the union pool directly.
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\nEmbedder Log: Generating embeddings for union pool...")

                # Increase number of threads for rank 0
                world_size = dist.get_world_size()
                total_cores = os.cpu_count()
                initial_threads_per_process = total_cores // world_size
                torch.set_num_threads(total_cores)

                all_embeddings = []
                all_ids = []
                for cand_pool_name in cand_pool_name_list:
                    cand_pool_name = cand_pool_name.lower()
                    cand_pool_name = f"mbeir_{cand_pool_name}_{split_name}"
                    embed_data_name = f"{cand_pool_name}_embed.npy"
                    id_data_name = f"{cand_pool_name}_ids.npy"
                    embed_path = os.path.join(
                        uniir_dir,
                        embed_dir_name,
                        expt_dir_name,
                        split_name,
                        embed_data_name,
                    )
                    id_path = os.path.join(
                        uniir_dir,
                        embed_dir_name,
                        expt_dir_name,
                        split_name,
                        id_data_name,
                    )
                    all_embeddings.append(np.load(embed_path))
                    all_ids.append(np.load(id_path))
                    print(f"Embedder Log: Concatenating embeddings from {embed_path} and ids from {id_path}.")

                all_embeddings = np.concatenate(all_embeddings, axis=0)
                all_ids = np.concatenate(all_ids, axis=0)
                assert len(all_embeddings) == len(all_ids), "Mismatch between embeddings and IDs length."
                print(f"Embedder Log: all_embeddings length: {len(all_embeddings)} and all_ids length: {len(all_ids)}.")

                # Save the embeddings to .npy
                embed_data_name = f"mbeir_union_{split_name}_embed.npy"
                embed_path = os.path.join(
                    uniir_dir,
                    embed_dir_name,
                    expt_dir_name,
                    split_name,
                    embed_data_name,
                )
                os.makedirs(os.path.dirname(embed_path), exist_ok=True)
                np.save(embed_path, all_embeddings)
                print(f"Embedder Log: Saved embeddings to {embed_path}.")

                # Save the IDs to .npy
                id_data_name = f"mbeir_union_{split_name}_ids.npy"
                id_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, id_data_name)
                os.makedirs(os.path.dirname(id_path), exist_ok=True)
                np.save(id_path, all_ids)
                print(f"Embedder Log: Saved ids to {id_path}.")

                # Delete the embeddings and IDs to free up memory
                del all_embeddings
                del all_ids

                # Explicitly call the garbage collector
                gc.collect()

                # Reset number of threads to initial value after conversion
                torch.set_num_threads(initial_threads_per_process)

            if dist.is_initialized():
                dist.barrier()  # Wait for rank 0 to finish saving the embeddings and ids.


def main(config):
    # Set up seed for reproducibility
    seed = config.seed + dist_utils.get_rank()
    set_seed(seed)

    # Initialize and load model
    model = build_model_from_config(config)
    model.eval()

    # Ensure the model has an 'encode' method before generating embeddings
    if not callable(getattr(model, "encode_mbeir_batch")):
        raise AttributeError("The provided model does not have a callable 'encode' method.")
    if not callable(getattr(model, "get_img_preprocess_fn")):
        raise AttributeError("The provided model does not have an 'img_preprocess_fn' attribute.")
    if not callable(getattr(model, "get_tokenizer")):
        raise AttributeError("The provided model does not have a 'tokenizer' attribute.")
    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()

    # Enable distributed data parallel
    model = model.to(config.dist_config.gpu_id)
    if config.dist_config.distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id])
    print(f"Model is set up on GPU {config.dist_config.gpu_id}.")

    # Generate embeddings
    generate_embeds_for_config(
        model=model,
        img_preprocess_fn=img_preprocess_fn,
        tokenizer=tokenizer,
        config=config,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Embeddings for MBEIR")
    parser.add_argument("--uniir_dir", type=str, default="retrieval/UniIR")
    parser.add_argument("--mbeir_data_dir", type=str, default="retrieval/TIGER-Lab/M-BEIR")
    parser.add_argument("--config_path", default="src/models/uniir_clip/clip_scorefusion/configs_scripts/large/eval/unirag/embed.yaml", help="Path to the config file.")
    return parser.parse_args()

## env: unirr step 1
if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    # Initialize distributed mode
    args.dist_url = config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
    dist_utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    if dist_utils.is_main_process():
        print(OmegaConf.to_yaml(config, sort_keys=False))

    main(config)

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
