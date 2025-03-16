import torch
from enum import Enum
from torch.utils.data import DataLoader

from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRCandidatePoolDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolCollator,
    Mode,
)


class DatasetType(Enum):
    MAIN_TRAIN = "main_train"
    IN_BATCH_VAL = "in_batch_val"
    CAND = "cand"


def build_mbeir_dataset_from_config(config, img_preprocess_fn, tokenizer, dataset_type):
    data_config = config.data_config
    if dataset_type == DatasetType.CAND:
        cand_pool_dataset = MBEIRCandidatePoolDataset(
            mbeir_data_dir=config.mbeir_data_dir,
            cand_pool_data_path=data_config.cand_pool_path,
            img_preprocess_fn=img_preprocess_fn,
        )
        cand_pool_collator = MBEIRCandidatePoolCollator(
            tokenizer=tokenizer,
            image_size=tuple(map(int, data_config.image_size.split(","))),
        )
        return cand_pool_dataset, cand_pool_collator

    if dataset_type == DatasetType.MAIN_TRAIN:
        query_data_path = data_config.train_query_data_path
        cand_pool_path = data_config.train_cand_pool_path
        mode = Mode.TRAIN
        hard_neg_num = data_config.hard_neg_num
    elif dataset_type == DatasetType.IN_BATCH_VAL:
        # Note: This validation dataset is used for in-batch validation.
        query_data_path = data_config.val_query_data_path
        cand_pool_path = data_config.val_cand_pool_path
        mode = Mode.TRAIN
        hard_neg_num = 0
        print("MBeir in-batch validation dataset is built.")
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    dataset = MBEIRMainDataset(
        mbeir_data_dir=config.mbeir_data_dir,
        query_data_path=query_data_path,
        cand_pool_path=cand_pool_path,
        query_instruct_path=data_config.query_instruct_path,
        img_preprocess_fn=img_preprocess_fn,
        mode=mode,
        enable_query_instruct=data_config.enable_query_instruct,
        shuffle_cand=data_config.shuffle_cand,
        hard_neg_num=hard_neg_num,
        returns=data_config.returns,
    )
    collector = MBEIRMainCollator(
        tokenizer=tokenizer,
        image_size=tuple(map(int, data_config.image_size.split(","))),
        mode=mode,
    )
    return dataset, collector


def build_distributed_sampler_list(dataset_list, shuffle_list, num_tasks_list, global_rank_list):
    samplers = []
    for dataset, shuffle, num_tasks, global_rank in zip(dataset_list, shuffle_list, num_tasks_list, global_rank_list):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def build_dataloader_list(datasets, samplers, batch_size_list, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size_list, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
