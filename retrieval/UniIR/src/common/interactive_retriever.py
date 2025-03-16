"""
Retrieves candidates for a given set of queries after embedding them.
"""

from enum import Enum
import gc
import json
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from data.mbeir_dataset import (
    MBEIRInferenceOnlyDataset,
    MBEIRInferenceOnlyCollator,
)
import dist_utils
from dist_utils import ContiguousDistributedSampler
from mbeir_embedder import generate_embeds_and_ids_for_dataset_with_gather
from utils import build_model_from_config, set_seed
from data.preprocessing.utils import unhash_did, DATASET_IDS, MBEIR_TASK


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    IMAGE_TEXT = "image,text"


class InteractiveRetriever:
    def __init__(self, cand_index_path: str, candidates_path: str, dataset_name, config):
        # Set up seed for reproducibility
        seed = config.seed + dist_utils.get_rank()
        set_seed(seed)
        self.dataset_id = DATASET_IDS[dataset_name]
        # Setup query embedder
        model = build_model_from_config(config)
        model.eval()

        # Ensure the model has an 'encode' method before generating embeddings
        if not callable(getattr(model, "encode_mbeir_batch")):
            raise AttributeError("The provided model does not have a callable 'encode' method.")
        if not callable(getattr(model, "get_img_preprocess_fn")):
            raise AttributeError("The provided model does not have an 'img_preprocess_fn' attribute.")
        if not callable(getattr(model, "get_tokenizer")):
            raise AttributeError("The provided model does not have a 'tokenizer' attribute.")
        self.img_preprocess_fn = model.get_img_preprocess_fn()
        self.tokenizer = model.get_tokenizer()

        # Enable distributed data parallel
        model = model.to(config.dist_config.gpu_id)
        if config.dist_config.distributed_mode:
            model = DDP(model, device_ids=[config.dist_config.gpu_id])
        self.model = model
        print(f"Model is set up on GPU {config.dist_config.gpu_id}.")

        self.cand_index_path = cand_index_path
        self.config = config
        self.queries = []

        # Load did_to_candidates
        self.did_to_candidates = {}
        with open(candidates_path, "r") as f:
            for l in f:
                c = json.loads(l.strip())
                assert c["did"] not in self.did_to_candidates, "dids must be unique"
                self.did_to_candidates[c["did"]] = c

    def add_queries(self, queries):
        for query_modality, query_txt, query_img_path, candidate_modality in queries:
            if query_modality == Modality.TEXT.value:
                assert query_txt, "Query with 'text' modality must have non-null 'query_txt'"
                assert query_img_path is None, "Query with 'text' modality must have null 'query_img_path'"
            elif query_modality == Modality.IMAGE.value:
                assert query_txt is None, "Query with 'image' modality must have null 'query_txt'"
                assert query_img_path, "Query with 'image' modality must have non-null 'query_img_path'"
            elif query_modality == Modality.IMAGE_TEXT.value:
                assert query_txt, "Query with 'image' modality must have non-null 'query_txt'"
                assert query_img_path, "Query with 'image' modality must have non-null 'query_img_path'"
            else:
                raise ValueError("Only 'text', 'image' and 'image,text' query modalities are supported.")
            task_id = MBEIR_TASK[" -> ".join([query_modality, candidate_modality])]
            self.queries.append(
                {
                    # Hardcoded qid in format of dataset_id:query_num.
                    "qid": ":".join([str(self.dataset_id), str(len(self.queries) + 1)]),
                    "query_modality": query_modality,
                    "query_txt": query_txt,
                    "query_img_path": query_img_path,
                    "task_id": task_id,
                    "candidate_modality": candidate_modality,
                }
            )

    def _embed_queries(self):
        mbeir_data_dir = self.config.mbeir_data_dir
        embed_config = self.config.embed_config

        # Config for dataset
        data_config = self.config.data_config
        query_instruct_path = data_config.query_instruct_path
        image_size = tuple(map(int, data_config.image_size.split(",")))

        print_config = False
        if dist_utils.is_main_process():
            print(f"\nEmbedder Log: Generating embeddings for {len(self.queries)} queries.")
            print_config = True

        dataset = MBEIRInferenceOnlyDataset(
            mbeir_data_dir,
            self.queries,
            query_instruct_path,
            self.img_preprocess_fn,
            enable_query_instruct=data_config.enable_query_instruct,
            print_config=print_config,
        )
        collator = MBEIRInferenceOnlyCollator(
            tokenizer=self.tokenizer,
            image_size=image_size,
        )

        # Config for data loader
        batch_size = self.config.dataloader_config.batch_size
        num_workers = self.config.dataloader_config.num_workers

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
            print(f"Embedder Log: Data loader is set up.")
            print(f"Embedder Log: Generating embeddings for {len(self.queries)} queries ...")
            print(f"Inference with half precision: {embed_config.use_fp16}")

        # Generate embeddings and ids
        embedding_list, id_list = generate_embeds_and_ids_for_dataset_with_gather(
            self.model,
            data_loader,
            device=self.config.dist_config.gpu_id,
            use_fp16=embed_config.use_fp16,
        )

        # Save the embeddings to a temprary .npy
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Embedder Log: Embedding list length: {len(embedding_list)}")
            print(f"Embedder Log: ID list length: {len(id_list)}")

            # Save the embeddings to .npy
            self.embed_file = "interactive_queries_embed.npy"
            np.save(self.embed_file, embedding_list)
            print(f"Embedder Log: Saved embeddings to {self.embed_file}.")

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

    def retrieve(self, k: int = 1, batch_size: int = 100):
        results = []
        self._embed_queries()
        # retrieve skipping the eval
        from mbeir_retriever import search_index

        print(f"Retriever: Searching with k={k}")
        _, retrieved_indices = search_index(
            self.embed_file,
            self.cand_index_path,
            batch_size=batch_size,
            num_cand_to_retrieve=k,
        )

        for indices in retrieved_indices:
            retrieved_cands = []
            for hashed_doc_id in indices:
                doc_id = unhash_did(hashed_doc_id)
                retrieved_cands.append(self.did_to_candidates[doc_id])
            results.append(retrieved_cands)

        # Remove the temprarily stored embeddings
        os.remove(self.embed_file)

        return results
