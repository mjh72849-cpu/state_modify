import logging
import math

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geomloss import SamplesLoss
from typing import Dict, Optional, Tuple

from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder, PanelFreeGeneDecoder
from .utils import build_mlp, get_activation_class, get_transformer_backbone, apply_lora


logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """Combined Sinkhorn + Energy loss."""

    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)

    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val


class ConfidenceToken(nn.Module):
    """
    Learnable confidence token that gets appended to the input sequence
    and learns to predict the expected loss value.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Learnable confidence token embedding
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Projection head to map confidence token output to scalar loss prediction
        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),  # Ensure positive loss prediction
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append confidence token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        # Expand confidence token to batch size
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        # Concatenate along sequence dimension
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(self, transformer_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and confidence prediction from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E]
            confidence_pred: Tensor of shape [B, 1]
        """
        # Split the output
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        confidence_output = transformer_output[:, -1:, :]  # [B, 1, E]

        # Project confidence token output to scalar
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)  # [B, 1]

        return main_output, confidence_pred


class StateTransitionPerturbationModel(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        basal_mapping_strategy: str = "random",
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        num_trainable_perturbations: int = 0,
        trainable_perturbation_names: list[str] | None = None,
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        # Call the parent PerturbationModel constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            num_trainable_perturbations=num_trainable_perturbations,
            trainable_perturbation_names=trainable_perturbation_names,
            **kwargs,
        )

        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.output_space = output_space
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim
        self.mmd_num_chunks = max(int(kwargs.get("mmd_num_chunks", 1)), 1)
        self.randomize_mmd_chunks = bool(kwargs.get("randomize_mmd_chunks", False))

        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            sinkhorn_weight = kwargs.get("sinkhorn_weight", 0.01)
            energy_weight = kwargs.get("energy_weight", 1.0)
            self.loss_fn = CombinedLoss(sinkhorn_weight=sinkhorn_weight, energy_weight=energy_weight, blur=blur)
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        self.use_basal_projection = kwargs.get("use_basal_projection", True)

        # Build the underlying neural OT network
        self._build_networks(lora_cfg=kwargs.get("lora", None))

        # Preserve the semantic protein prior while giving every observed
        # genetic perturbation its own trainable identity. Zero initialization
        # makes checkpoint transfer behavior identical before the first update.
        self.num_trainable_perturbations = int(num_trainable_perturbations)
        if self.num_trainable_perturbations < 0:
            raise ValueError("num_trainable_perturbations cannot be negative")
        self.trainable_perturbation_names = list(trainable_perturbation_names or [])
        if self.trainable_perturbation_names and (
            len(self.trainable_perturbation_names) != self.num_trainable_perturbations
        ):
            raise ValueError(
                "trainable_perturbation_names length must equal num_trainable_perturbations"
            )
        if len(self.trainable_perturbation_names) != len(set(self.trainable_perturbation_names)):
            raise ValueError("trainable_perturbation_names contains duplicates")
        self.trainable_perturbation_to_id = {
            name: index for index, name in enumerate(self.trainable_perturbation_names)
        }
        self.perturbation_residual = None
        if self.num_trainable_perturbations:
            self.perturbation_residual = nn.Embedding(
                self.num_trainable_perturbations, self.hidden_dim
            )
            nn.init.zeros_(self.perturbation_residual.weight)

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # Optional batch predictor ablation: learns a single batch token added to every position,
        # and adds an auxiliary per-token batch classification head + CE loss.
        self.batch_predictor = bool(kwargs.get("batch_predictor", False))
        # If batch_encoder is enabled, disable batch_predictor per request
        if self.batch_encoder is not None and self.batch_predictor:
            logger.warning(
                "Both model.kwargs.batch_encoder and model.kwargs.batch_predictor are True. "
                "Disabling batch_predictor and proceeding with batch_encoder."
            )
            self.batch_predictor = False
            try:
                # Keep hparams in sync if available
                self.hparams["batch_predictor"] = False  # type: ignore[index]
            except Exception:
                pass

        self.batch_predictor_weight = float(kwargs.get("batch_predictor_weight", 0.1))
        self.batch_predictor_num_classes: Optional[int] = batch_dim if self.batch_predictor else None
        if self.batch_predictor:
            if self.batch_predictor_num_classes is None:
                raise ValueError("batch_predictor=True requires a valid `batch_dim` (number of batch classes).")
            # A single learnable batch token that is added to each position
            self.batch_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            # Simple per-token classifier from transformer hidden to batch classes
            self.batch_classifier = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.batch_predictor_num_classes,
                hidden_dim=self.hidden_dim,
                n_layers=4,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.batch_token = None
            self.batch_classifier = None
        # Internal cache for last token features (B, S, H) from transformer for aux loss
        self._token_features: Optional[torch.Tensor] = None

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            self.relu = torch.nn.ReLU()

        self.use_batch_token = kwargs.get("use_batch_token", False)
        self.basal_mapping_strategy = basal_mapping_strategy
        # Disable batch token only for truly incompatible cases
        disable_reasons = []
        if self.batch_encoder and self.use_batch_token:
            disable_reasons.append("batch encoder is used")
        if basal_mapping_strategy == "random" and self.use_batch_token:
            disable_reasons.append("basal mapping strategy is random")

        if disable_reasons:
            self.use_batch_token = False
            logger.warning(
                f"Batch token is not supported when {' or '.join(disable_reasons)}, setting use_batch_token to False"
            )
            try:
                self.hparams["use_batch_token"] = False
            except Exception:
                pass

        self.batch_token_weight = kwargs.get("batch_token_weight", 0.1)
        self.batch_token_num_classes: Optional[int] = batch_dim if self.use_batch_token else None

        if self.use_batch_token:
            if self.batch_token_num_classes is None:
                raise ValueError("batch_token_num_classes must be set when use_batch_token is True")
            self.batch_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            self.batch_classifier = build_mlp(
                in_dim=self.hidden_dim,
                out_dim=self.batch_token_num_classes,
                hidden_dim=self.hidden_dim,
                n_layers=1,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.batch_token = None
            self.batch_classifier = None

        # Internal cache for last token features (B, S, H) from transformer for aux loss
        self._batch_token_cache: Optional[torch.Tensor] = None

        # initialize a confidence token
        self.confidence_token = None
        self.confidence_loss_fn = None
        if kwargs.get("confidence_token", False):
            self.confidence_token = ConfidenceToken(hidden_dim=self.hidden_dim, dropout=self.dropout)
            self.confidence_loss_fn = nn.MSELoss()
            self.confidence_target_scale = float(kwargs.get("confidence_target_scale", 10.0))
            self.confidence_weight = float(kwargs.get("confidence_weight", 0.01))
        else:
            self.confidence_target_scale = None
            self.confidence_weight = 0.0

        # Backward-compat: accept legacy key `freeze_pert`
        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", kwargs.get("freeze_pert", False))
        if self.freeze_pert_backbone:
            # Freeze backbone base weights but keep LoRA adapter weights (if present) trainable
            for name, param in self.transformer_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # Freeze projection head as before
            for param in self.project_out.parameters():
                param.requires_grad = False

        # Warm-up mode for transfer learning: train only the newly initialized
        # target-gene encoder, panel-free decoder, and fallback parameters.
        self.freeze_pretrained_backbone = bool(kwargs.get("freeze_pretrained_backbone", False))
        if self.freeze_pretrained_backbone:
            for module in (self.basal_encoder, self.transformer_backbone, self.project_out):
                for param in module.parameters():
                    param.requires_grad = False

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False) and self.panel_free_decoder_cfg is not None:
            raise ValueError("finetune_vci_decoder and panel_free_decoder_cfg are mutually exclusive")
        if kwargs.get("finetune_vci_decoder", False):  # TODO: This will go very soon
            # Prefer the gene names supplied by the data module (aligned to training output)
            gene_names = self.gene_names
            if gene_names is None:
                raise ValueError(
                    "finetune_vci_decoder=True but model.gene_names is None. "
                    "Please provide gene_names via data module var_dims."
                )

            n_genes = len(gene_names)
            logger.info(
                f"Initializing FinetuneVCICountsDecoder with {n_genes} genes (output_space={output_space}; "
                + ("HVG subset" if output_space == "gene" else "all genes")
                + ")"
            )
            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
            )
        print(self)

    def _build_networks(self, lora_cfg=None):
        """
        Here we instantiate the actual GPT2-based model.
        """
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Simple linear layer that maintains the input dimension
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Optionally wrap backbone with LoRA adapters
        if lora_cfg and lora_cfg.get("enable", False):
            self.transformer_backbone = apply_lora(
                self.transformer_backbone,
                self.transformer_backbone_key,
                lora_cfg,
            )

        # Project from input_dim to hidden_dim for transformer input
        # self.project_to_hidden = nn.Linear(self.input_dim, self.hidden_dim)

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.output_space == "all":
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )

    def encode_perturbation(
        self,
        pert: torch.Tensor,
        perturbation_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine the semantic encoding with an optional learned target residual.

        ``-1`` is reserved for control and contributes exactly zero. Non-control
        IDs are stable rows in the registry emitted by the data preparation
        pipeline, so targets never collapse merely because names differ.
        """
        encoded = self.pert_encoder(pert)
        residual_table = getattr(self, "perturbation_residual", None)
        if residual_table is None:
            return encoded
        if perturbation_ids is None:
            raise KeyError(
                "This model has a trainable perturbation table but batch['perturbation_ids'] is missing"
            )
        perturbation_ids = perturbation_ids.to(device=encoded.device, dtype=torch.long)
        if perturbation_ids.shape != encoded.shape[:-1]:
            raise ValueError(
                f"perturbation_ids shape {tuple(perturbation_ids.shape)} does not match "
                f"perturbation tokens {tuple(encoded.shape[:-1])}"
            )
        active = perturbation_ids >= 0
        if active.any() and perturbation_ids[active].max() >= residual_table.num_embeddings:
            raise IndexError("perturbation ID is outside the trainable perturbation registry")
        safe_ids = perturbation_ids.clamp_min(0)
        residual = residual_table(safe_ids)
        residual = residual * active.unsqueeze(-1).to(residual.dtype)
        return encoded + residual

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def forward(self, batch: dict, padded=True) -> torch.Tensor:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
            perturbation_ids = batch.get("perturbation_ids")
            if perturbation_ids is not None:
                perturbation_ids = perturbation_ids.reshape(-1, self.cell_sentence_len)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)
            perturbation_ids = batch.get("perturbation_ids")
            if perturbation_ids is not None:
                perturbation_ids = perturbation_ids.reshape(1, -1)

        # Shape: [B, S, input_dim]
        pert_embedding = self.encode_perturbation(pert, perturbation_ids)
        control_cells = self.encode_basal_expression(basal)

        # Add encodings in input_dim space, then project to hidden_dim
        combined_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        seq_input = combined_input  # Shape: [B, S, hidden_dim]

        if self.batch_encoder is not None:
            # Extract batch indices (assume they are integers or convert from one-hot)
            batch_indices = batch["batch"]

            # Handle one-hot encoded batch indices
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            # Reshape batch indices to match sequence structure
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)

            # Get batch embeddings and add to sequence input
            batch_embeddings = self.batch_encoder(batch_indices.long())  # Shape: [B, S, hidden_dim]
            seq_input = seq_input + batch_embeddings

        if self.use_batch_token and self.batch_token is not None:
            batch_size, _, _ = seq_input.shape
            # Prepend the batch token to the sequence along the sequence dimension
            # [B, S, H] -> [B, S+1, H], batch token at position 0
            seq_input = torch.cat([self.batch_token.expand(batch_size, -1, -1), seq_input], dim=1)

        confidence_pred = None
        if self.confidence_token is not None:
            # Append confidence token: [B, S, E] -> [B, S+1, E] (might be one more if we have the batch token)
            seq_input = self.confidence_token.append_confidence_token(seq_input)

        # forward pass + extract CLS last hidden state
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device
            self.transformer_backbone._attn_implementation = "eager"  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

            # create a [1,1,S,S] mask (now S+1 if confidence token is used)
            base = torch.eye(seq_length, device=device, dtype=torch.bool).view(1, 1, seq_length, seq_length)

            # Get number of attention heads from model config
            num_heads = self.transformer_backbone.config.num_attention_heads

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, num_heads, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            outputs = self.transformer_backbone(inputs_embeds=seq_input)
            transformer_output = outputs.last_hidden_state

        # Extract outputs accounting for optional prepended batch token and optional confidence token at the end
        if self.confidence_token is not None and self.use_batch_token and self.batch_token is not None:
            # transformer_output: [B, 1 + S + 1, H] -> batch token at 0, cells 1..S, confidence at -1
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(
                transformer_output[:, 1:, :]
            )
            # res_pred currently excludes the confidence token and starts from former index 1
            self._batch_token_cache = batch_token_pred
        elif self.confidence_token is not None:
            # Only confidence token appended at the end
            res_pred, confidence_pred = self.confidence_token.extract_confidence_prediction(transformer_output)
            self._batch_token_cache = None
        elif self.use_batch_token and self.batch_token is not None:
            # Only batch token prepended at the beginning
            batch_token_pred = transformer_output[:, :1, :]  # [B, 1, H]
            res_pred = transformer_output[:, 1:, :]  # [B, S, H]
            self._batch_token_cache = batch_token_pred
        else:
            # Neither special token used
            res_pred = transformer_output
            self._batch_token_cache = None

        # Cache token features for auxiliary batch prediction loss (B, S, H)
        self._token_features = res_pred

        # add to basal if predicting residual
        if (
            self.predict_residual
            and self.output_space == "all"
            and not isinstance(self.gene_decoder, PanelFreeGeneDecoder)
        ):
            # Project control_cells to hidden_dim space to match res_pred
            # control_cells_hidden = self.project_to_hidden(control_cells)
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        # apply relu if specified and we output to HVG space
        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            out_pred = self.relu(out_pred)

        output = out_pred.reshape(-1, self.output_dim)

        if confidence_pred is not None:
            return output, confidence_pred
        else:
            return output

    def _compute_distribution_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply the primary distributional loss, optionally chunking feature dimensions for SamplesLoss."""

        if isinstance(self.loss_fn, SamplesLoss) and self.mmd_num_chunks > 1:
            feature_dim = pred.shape[-1]
            num_chunks = min(self.mmd_num_chunks, feature_dim)
            if num_chunks > 1 and feature_dim > 0:
                if self.randomize_mmd_chunks and self.training:
                    perm = torch.randperm(feature_dim, device=pred.device)
                    pred = pred.index_select(-1, perm)
                    target = target.index_select(-1, perm)
                pred_chunks = torch.chunk(pred, num_chunks, dim=-1)
                target_chunks = torch.chunk(target, num_chunks, dim=-1)
                chunk_losses = [self.loss_fn(p_chunk, t_chunk) for p_chunk, t_chunk in zip(pred_chunks, target_chunks)]
                return torch.stack(chunk_losses, dim=0).nanmean(dim=0)

        return self.loss_fn(pred, target)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        batch = self._normalize_count_keys(batch)
        # Get model predictions (in latent space)
        confidence_pred = None
        if self.confidence_token is not None:
            pred, confidence_pred = self.forward(batch, padded=padded)
        else:
            pred = self.forward(batch, padded=padded)

        target = batch["pert_cell_emb"]

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        per_set_main_losses = self._compute_distribution_loss(pred, target)
        main_loss = torch.nanmean(per_set_main_losses)
        self.log("train_loss", main_loss)

        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).nanmean()
            energy_component = self.loss_fn.energy_loss(pred, target).nanmean()
            self.log("train/sinkhorn_loss", sinkhorn_component)
            self.log("train/energy_loss", energy_component)

        # Process decoder if available
        decoder_loss = None
        total_loss = main_loss

        if self.use_batch_token and self.batch_classifier is not None and self._batch_token_cache is not None:
            logits = self.batch_classifier(self._batch_token_cache)  # [B, 1, C]
            batch_token_targets = batch["batch"]

            B = logits.shape[0]
            C = logits.size(-1)

            # Prepare one label per sequence (all S cells share the same batch)
            if batch_token_targets.dim() > 1 and batch_token_targets.size(-1) == C:
                # One-hot labels; reshape to [B, S, C]
                if padded:
                    target_oh = batch_token_targets.reshape(-1, self.cell_sentence_len, C)
                else:
                    target_oh = batch_token_targets.reshape(1, -1, C)
                sentence_batch_labels = target_oh.argmax(-1)
            else:
                # Integer labels; reshape to [B, S]
                if padded:
                    sentence_batch_labels = batch_token_targets.reshape(-1, self.cell_sentence_len)
                else:
                    sentence_batch_labels = batch_token_targets.reshape(1, -1)

            if sentence_batch_labels.shape[0] != B:
                sentence_batch_labels = sentence_batch_labels.reshape(B, -1)

            if self.basal_mapping_strategy == "batch":
                uniform_mask = sentence_batch_labels.eq(sentence_batch_labels[:, :1]).all(dim=1)
                if not torch.all(uniform_mask):
                    bad_indices = torch.where(~uniform_mask)[0]
                    label_strings = []
                    for idx in bad_indices:
                        labels = sentence_batch_labels[idx].detach().cpu().tolist()
                        logger.error("Batch labels for sentence %d: %s", idx.item(), labels)
                        label_strings.append(f"sentence {idx.item()}: {labels}")
                    raise ValueError(
                        "Expected all cells in a sentence to share the same batch when "
                        "basal_mapping_strategy is 'batch'. "
                        f"Found mixed batch labels: {', '.join(label_strings)}"
                    )

            target_idx = sentence_batch_labels[:, 0]

            # Safety: ensure exactly one target per sequence
            if target_idx.numel() != B:
                target_idx = target_idx.reshape(-1)[:B]

            ce_loss = F.cross_entropy(logits.reshape(B, -1, C).squeeze(1), target_idx.long())
            self.log("train/batch_token_loss", ce_loss)
            total_loss = total_loss + self.batch_token_weight * ce_loss

        # Auxiliary batch prediction loss (per token), if enabled
        if isinstance(self.gene_decoder, PanelFreeGeneDecoder) and "gene_targets" not in batch:
            raise KeyError(
                "PanelFreeGeneDecoder requires gene_targets/gene_embeddings. "
                "Use PanelFreePerturbationDataModule or an equivalent collator."
            )
        has_panel_targets = isinstance(self.gene_decoder, PanelFreeGeneDecoder) and "gene_targets" in batch
        has_fixed_targets = not isinstance(self.gene_decoder, PanelFreeGeneDecoder) and "pert_cell_counts" in batch
        if self.gene_decoder is not None and (has_panel_targets or has_fixed_targets):
            gene_targets = batch["gene_targets"] if has_panel_targets else batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space

            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            if isinstance(self.gene_decoder, PanelFreeGeneDecoder):
                pert_cell_counts_preds = self.gene_decoder(
                    latent_preds,
                    batch["gene_embeddings"],
                    fallback_ids=batch.get("gene_fallback_ids"),
                    gene_baseline=batch.get("gene_baselines"),
                    chunk_size=self.hparams.get("gene_decoder_chunk_size", None),
                )
                gene_targets = gene_targets.reshape_as(pert_cell_counts_preds)
            elif padded:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

            if has_panel_targets and "gene_mask" in batch:
                mask = batch["gene_mask"].reshape_as(gene_targets).to(dtype=gene_targets.dtype)
                squared_error = (pert_cell_counts_preds - gene_targets).square()
                decoder_loss = (squared_error * mask).sum() / mask.sum().clamp_min(1)
            else:
                decoder_per_set = self._compute_distribution_loss(pert_cell_counts_preds, gene_targets)
                decoder_loss = decoder_per_set.mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            confidence_pred_vals = confidence_pred
            if confidence_pred_vals.dim() > 1:
                confidence_pred_vals = confidence_pred_vals.squeeze(-1)
            confidence_targets = per_set_main_losses.detach()
            if self.confidence_target_scale is not None:
                confidence_targets = confidence_targets * self.confidence_target_scale
            confidence_targets = confidence_targets.to(confidence_pred_vals.device)

            confidence_loss = self.confidence_weight * self.confidence_loss_fn(confidence_pred_vals, confidence_targets)
            self.log("train/confidence_loss", confidence_loss)
            self.log("train/actual_loss", confidence_targets.mean())

            total_loss = total_loss + confidence_loss

        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            delta = pred - ctrl_cell_emb

            # compute l1 loss
            l1_loss = torch.abs(delta).mean()

            # Log the regularization loss
            self.log("train/l1_regularization", l1_loss)

            # Add regularization to total loss
            total_loss = total_loss + self.regularization * l1_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        batch = self._normalize_count_keys(batch)
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch), None
        else:
            pred, confidence_pred = self.forward(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        per_set_main_losses = self._compute_distribution_loss(pred, target)
        loss = torch.nanmean(per_set_main_losses)
        self.log("val_loss", loss)

        # Log individual loss components if using combined loss
        if hasattr(self.loss_fn, "sinkhorn_loss") and hasattr(self.loss_fn, "energy_loss"):
            sinkhorn_component = self.loss_fn.sinkhorn_loss(pred, target).mean()
            energy_component = self.loss_fn.energy_loss(pred, target).mean()
            self.log("val/sinkhorn_loss", sinkhorn_component)
            self.log("val/energy_loss", energy_component)

        if isinstance(self.gene_decoder, PanelFreeGeneDecoder) and "gene_targets" not in batch:
            raise KeyError(
                "PanelFreeGeneDecoder requires gene_targets/gene_embeddings. "
                "Use PanelFreePerturbationDataModule or an equivalent collator."
            )
        has_panel_targets = isinstance(self.gene_decoder, PanelFreeGeneDecoder) and "gene_targets" in batch
        has_fixed_targets = not isinstance(self.gene_decoder, PanelFreeGeneDecoder) and "pert_cell_counts" in batch
        if self.gene_decoder is not None and (has_panel_targets or has_fixed_targets):
            gene_targets = batch["gene_targets"] if has_panel_targets else batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            if isinstance(self.gene_decoder, PanelFreeGeneDecoder):
                pert_cell_counts_preds = self.gene_decoder(
                    latent_preds,
                    batch["gene_embeddings"],
                    fallback_ids=batch.get("gene_fallback_ids"),
                    gene_baseline=batch.get("gene_baselines"),
                    chunk_size=self.hparams.get("gene_decoder_chunk_size", None),
                )
                gene_targets = gene_targets.reshape_as(pert_cell_counts_preds)
                if "gene_mask" in batch:
                    mask = batch["gene_mask"].reshape_as(gene_targets).to(dtype=gene_targets.dtype)
                    squared_error = (pert_cell_counts_preds - gene_targets).square()
                    decoder_loss = (squared_error * mask).sum() / mask.sum().clamp_min(1)
                else:
                    decoder_loss = (pert_cell_counts_preds - gene_targets).square().mean()
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                    -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
                )
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                decoder_per_set = self._compute_distribution_loss(pert_cell_counts_preds, gene_targets)
                decoder_loss = decoder_per_set.mean()

            # Log the validation metric
            self.log("val/decoder_loss", decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if confidence_pred is not None:
            confidence_pred_vals = confidence_pred
            if confidence_pred_vals.dim() > 1:
                confidence_pred_vals = confidence_pred_vals.squeeze(-1)
            confidence_targets = per_set_main_losses.detach()
            if self.confidence_target_scale is not None:
                confidence_targets = confidence_targets * self.confidence_target_scale
            confidence_targets = confidence_targets.to(confidence_pred_vals.device)

            confidence_loss = self.confidence_weight * self.confidence_loss_fn(confidence_pred_vals, confidence_targets)
            self.log("val/confidence_loss", confidence_loss)
            self.log("val/actual_loss", confidence_targets.mean())

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        batch = self._normalize_count_keys(batch)
        if self.confidence_token is None:
            pred, confidence_pred = self.forward(batch, padded=False), None
        else:
            pred, confidence_pred = self.forward(batch, padded=False)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        per_set_main_losses = self._compute_distribution_loss(pred, target)
        loss = torch.nanmean(per_set_main_losses)
        self.log("test_loss", loss)

        if confidence_pred is not None:
            confidence_pred_vals = confidence_pred
            if confidence_pred_vals.dim() > 1:
                confidence_pred_vals = confidence_pred_vals.squeeze(-1)
            confidence_targets = per_set_main_losses.detach()
            if self.confidence_target_scale is not None:
                confidence_targets = confidence_targets * self.confidence_target_scale
            confidence_targets = confidence_targets.to(confidence_pred_vals.device)

            confidence_loss = self.confidence_weight * self.confidence_loss_fn(confidence_pred_vals, confidence_targets)
            self.log("test/confidence_loss", confidence_loss)

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:s
         returning 'preds', 'X', 'pert_name', etc.
        """
        batch = self._normalize_count_keys(batch)
        if self.confidence_token is None:
            latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
            confidence_pred = None
        else:
            latent_output, confidence_pred = self.forward(batch, padded=padded)

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        # Add confidence prediction to output if available
        if confidence_pred is not None:
            output_dict["confidence_pred"] = confidence_pred

        if isinstance(self.gene_decoder, PanelFreeGeneDecoder):
            if "gene_embeddings" not in batch:
                raise KeyError(
                    "PanelFreeGeneDecoder prediction requires gene_embeddings. "
                    "Use PanelFreePerturbationDataModule or predict_vcc_counts."
                )
            if padded:
                decoder_latent = latent_output.reshape(-1, self.cell_sentence_len, self.output_dim)
            else:
                decoder_latent = latent_output.reshape(1, -1, self.output_dim)
            pert_cell_counts_preds = self.gene_decoder(
                decoder_latent,
                batch["gene_embeddings"],
                fallback_ids=batch.get("gene_fallback_ids"),
                gene_baseline=batch.get("gene_baselines"),
                chunk_size=self.hparams.get("gene_decoder_chunk_size", None),
            )
            output_dict["gene_names"] = batch.get("gene_names")
            output_dict["gene_mask"] = batch.get("gene_mask")
        elif self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)

        if self.gene_decoder is not None:
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict
