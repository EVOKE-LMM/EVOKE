"""
Feature level fusion model using CLIP
Code adapted from OpenAI's CLIP codebase
"""

from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass

# Import from third library
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed.nn

from transformers.models.t5.modeling_t5 import T5Block, T5Stack

import clip
from clip.model import VisionTransformer

from transformers.models.t5 import T5Config


class VisionTransformerWithoutPooling(VisionTransformer):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIPFeatureFusion(nn.Module):
    """CLIP Feature Fusion model implemented using OpenAI's CLIP and HuggingFace Transformers' T5 models"""

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

        # T5 layers for feature fusion
        if model_name == "ViT-B/32":
            conf_t5 = T5Config()
            conf_t5.num_layers = 2
            conf_t5.num_decoder_layers = 2
            conf_t5.num_heads = 12
            conf_t5.d_model = 512
            conf_t5.d_kv = 64
            self.t5_layers = T5Stack(conf_t5)
        elif model_name == "ViT-L/14":
            conf_t5_vit_large = T5Config()
            conf_t5_vit_large.num_layers = 2
            conf_t5_vit_large.num_decoder_layers = 2
            conf_t5_vit_large.num_heads = 12
            conf_t5_vit_large.d_model = 768
            conf_t5_vit_large.d_kv = 64
            self.t5_layers = T5Stack(conf_t5_vit_large)
        else:
            raise NotImplementedError("Only ViT-B/32 and ViT-L/14 are supported.")

        if config is not None:
            self.gather_embeddings = config.model.gather_embeddings
            self.in_batch_neg_num = config.data_config.in_batch_neg_num
        else:
            self.gather_embeddings = None
            self.in_batch_neg_num = None
        # to avoid unused parameters warning when doing distributed training
        del self.clip_model.text_projection
        state_dict = self.clip_model.visual.state_dict()
        self.clip_model.visual = self.get_vision_transformer(model_name=model_name)
        self.clip_model.visual.load_state_dict(state_dict)
        self.clip_model.float()

    def get_vision_transformer(self, model_name="VIT-B/32"):
        if model_name == "ViT-B/32":
            return VisionTransformerWithoutPooling(
                input_resolution=224,
                patch_size=32,
                width=768,
                layers=12,
                heads=12,
                output_dim=512,
            )
        elif model_name == "ViT-L/14":
            return VisionTransformerWithoutPooling(
                input_resolution=224,
                patch_size=14,
                width=1024,
                layers=24,
                heads=16,
                output_dim=768,
            )
        else:
            raise NotImplementedError("Only ViT-B/32 and ViT-L/14 are supported.")

    def get_img_preprocess_fn(self):
        return self.img_preprocess_fn

    def get_tokenizer(self):
        """Get the tokenize function used by the CLIP model"""

        def tokenizer_wrapper(txt):
            tokenizer = self.tokenizer
            txt_tensor = tokenizer(txt, context_length=77, truncate=True)
            return txt_tensor

        return tokenizer_wrapper

    def encode_text(self, text_tensor):
        x = self.clip_model.token_embedding(text_tensor).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        return x

    def encode_image(self, image_tensor):
        return self.clip_model.encode_image(image_tensor)  # [batch_size, seq_len, embed_dim]

    def encode_multimodal_input(self, txt_tensor, img_tensor, txt_mask, img_mask):
        """Encode multimodal input using CLIP and T5 models

        Args:
            txt_tensor (_type_): text tensor [batch_size, seq_len]
            img_tensor (_type_): image tensor [batch_size, 3, img_size, img_size]
            txt_mask (_type_): text mask [batch_size]
            img_mask (_type_): image mask [batch_size]

        Returns:
            multimodal_emb (_type_): _description_ multimodal embeddings [batch_size, embed_dim]
        """
        txt_feat = self.encode_text(txt_tensor)
        # txt_feat = txt_feat * txt_mask.unsqueeze(-1).unsqueeze(-1).expand_as(txt_feat)
        img_feat = self.encode_image(img_tensor)
        # img_feat = img_feat * img_mask.unsqueeze(-1).unsqueeze(-1).expand_as(img_feat)
        combined_features = torch.cat([txt_feat, img_feat], dim=1)  # shape: [batch_size, seq_len, embed_dim]

        # combined_features = self.dense_clip_to_t5(combined_features)
        transformer_output = self.t5_layers(
            inputs_embeds=combined_features,
            attention_mask=None,
            use_cache=False,
            return_dict=True,
        )

        def mean_pooling(embeddings):
            return torch.mean(embeddings, dim=1)

        # Pool the output of the T5 transformer to get the final features
        multimodal_emb = mean_pooling(transformer_output.last_hidden_state)
        return multimodal_emb  # shape: [batch_size, embed_dim]

    def get_logit_scale(self):
        return self.clip_model.logit_scale.exp()

    def compute_inbatch_contrastive_loss(self, batch):
        """Compute the in-batch contrastive loss

        Args:
            batch (dict): batch dictionary consists of "txt_batch", "image_batch",
            "txt_mask_batch", "image_mask_batch", "index_mapping"

        Returns:
            outputs (dict): dictionary of loss and accuracy
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
