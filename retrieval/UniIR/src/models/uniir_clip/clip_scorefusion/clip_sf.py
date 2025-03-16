"""
Score level fusion model using CLIP
Code adapted from OpenAI's CLIP codebase
"""

import torch
from torch import nn
import torch.nn.functional as F
import clip
import torch.distributed.nn


class CLIPScoreFusion(nn.Module):
    def __init__(
        self,
        model_name="ViT-B/32",
        device="cuda",
        jit=False,
        download_root=None,
        config=None,
    ):
        super().__init__()

        # Load pre-trained CLIP model
        self.clip_model, self.img_preprocess_fn = clip.load(model_name, device, jit, download_root=download_root)
        self.tokenizer = clip.tokenize
        self.loss_function = nn.CrossEntropyLoss()
        if config is not None:
            self.gather_embeddings = config.model.gather_embeddings
            self.in_batch_neg_num = config.data_config.in_batch_neg_num

    def get_img_preprocess_fn(self):
        return self.img_preprocess_fn

    def get_tokenizer(self):
        def tokenizer_wrapper(txt):
            tokenizer = self.tokenizer
            txt_tensor = tokenizer(txt, context_length=77, truncate=True)
            return txt_tensor

        return tokenizer_wrapper

    def encode_text(self, text_tensor):
        return self.clip_model.encode_text(text_tensor)

    def encode_image(self, image_tensor):
        return self.clip_model.encode_image(image_tensor)

    def fuse_embeddings(self, img_emb, txt_emb):
        fused_emb = img_emb + txt_emb
        return fused_emb

    def encode_multimodal_input(self, txt_tensor, img_tensor, txt_mask, img_mask):
        """
        :param txt_tensor:
        :param img_tensor:
        :param txt_mask:  expected shape: [batch_size, 1]
        :param img_mask:  expected shape: [batch_size, 1]
        :return:
        """
        txt_emb = self.encode_text(txt_tensor) * txt_mask.unsqueeze(-1)
        img_emb = self.encode_image(img_tensor) * img_mask.unsqueeze(-1)
        return self.fuse_embeddings(txt_emb, img_emb)  # shape: [batch_size, embed_dim]

    def get_logit_scale(self):
        return self.clip_model.logit_scale.exp()

    def compute_inbatch_contrastive_loss(self, batch):
        """
         adapted from the CLIP codebase and UniVL-DR codebase

        :param model:
        :param batch:
        :param loss_function:
        :return:
        """
        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask_batched = batch["txt_mask_batched"]
        image_mask_batched = batch["image_mask_batched"]
        index_mapping = batch["index_mapping"]
        enable_hard_neg = "neg_cand_list" in index_mapping

        # Compute embeddings
        embeddings = self.encode_multimodal_input(txt_batched, image_batched, txt_mask_batched, image_mask_batched)

        # Extract embeddings
        q_embeds = embeddings[torch.tensor(index_mapping["query"]).flatten()]  # shape: [bs, embed_dim]
        p_embeds = embeddings[torch.tensor(index_mapping["pos_cand"]).flatten()]  # shape: [bs, embed_dim]
        n_embeds = None
        if enable_hard_neg:
            n_embeds = embeddings[torch.tensor(index_mapping["neg_cand_list"])]  # [bs, neg_num, embed_dim]
        bs = q_embeds.size(0)

        # Normalized features
        q_embeds = F.normalize(q_embeds, dim=-1)
        p_embeds = F.normalize(p_embeds, dim=-1)

        logit_scale = self.get_logit_scale()

        # We gather tensors from all gpus
        if self.gather_embeddings:
            all_p_embeds = torch.cat(torch.distributed.nn.all_gather(p_embeds), dim=0)  # [bs * num_gpus, embed_dim]

        if enable_hard_neg:
            # Normalize the negative embeddings
            n_embeds = F.normalize(n_embeds, dim=-1)

            # Number of in-batch positives to add as negatives
            in_batch_neg_num = min(bs - 1, self.in_batch_neg_num)

            # Augment neg_cand_embeddings with a subset of in-batch positive candidates from other queries
            mask = torch.eye(bs).to(n_embeds.device) == 0
            in_batch_negs = p_embeds.unsqueeze(1).expand(-1, bs, -1)[mask].reshape(bs, bs - 1, -1)
            in_batch_negs = in_batch_negs[:, :in_batch_neg_num, :]
            aug_n_embeds = torch.cat([n_embeds, in_batch_negs], dim=1)  # [bs, neg_num + in_batch_neg_num, embed_dim]

            # Compute similarity scores for positives and negatives
            pos_scores = (q_embeds * p_embeds).sum(-1) * logit_scale  # [bs]
            neg_scores = (q_embeds.unsqueeze(1) * aug_n_embeds).sum(-1) * logit_scale  # [bs, neg_num +in_batch_neg_num]
            logit_matrix = torch.cat([pos_scores.unsqueeze(-1), neg_scores], 1)  # [bs, neg_num + in_batch_neg_num + 1]

            # Compute log softmax over the matrix
            lsm = F.log_softmax(logit_matrix, dim=1)

            # The NNL loss for the positive candidate
            loss = torch.mean(-1.0 * lsm[:, 0])

            # Compute accuracy by checking which instances have the positive candidate as the most similar one
            _max_score, max_idxs = torch.max(logit_matrix, 1)
            accuracy = (max_idxs == 0).sum() / bs
        else:
            if self.gather_embeddings:
                score = torch.matmul(q_embeds, all_p_embeds.t()) * logit_scale  # [bs, bs * num_gpus]
                gpu_id = torch.distributed.get_rank()
                sim_targets = (gpu_id * bs + torch.arange(bs)).to(score.device)  # [bs]
            else:
                score = torch.matmul(q_embeds, p_embeds.t()) * logit_scale  # [bs, bs]
                sim_targets = torch.arange(bs).to(score.device)  # [bs]

            # compute loss
            loss = self.loss_function(score, sim_targets)
            _max_score, max_idxs = torch.max(score, 1)
            accuracy = (max_idxs == sim_targets).sum() / bs

        outputs = {"loss": loss, "accuracy": accuracy}
        return outputs

    def forward(self, batch, encode_mbeir_batch=False):
        if encode_mbeir_batch:
            return self.encode_mbeir_batch(batch)
        return self.compute_inbatch_contrastive_loss(batch)

    def encode_mbeir_batch(self, batch):
        # Get hashed id_list
        id_list = batch.get("did_list") or batch.get("qid_list")
        assert id_list is not None, "id_list must be provided."
        assert isinstance(id_list[0], int), "id_list must be hashed to int."

        # Compute embeddings
        embeddings = self.encode_multimodal_input(
            batch["txt_batched"],
            batch["image_batched"],
            batch["txt_mask_batched"],
            batch["image_mask_batched"],
        )
        assert embeddings.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."
        return embeddings, id_list
