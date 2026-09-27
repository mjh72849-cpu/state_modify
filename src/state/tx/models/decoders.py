import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from omegaconf import OmegaConf

from ...emb.finetune_decoder import Finetune

logger = logging.getLogger(__name__)


class PanelFreeGeneDecoder(nn.Module):
    """Decode arbitrary genes from a cell-state embedding.

    Unlike :class:`LatentToGeneDecoder` and the legacy VCI decoder below, the
    output head does not depend on the size or ordering of the queried panel.
    The optional fallback registry has one residual per explicitly configured
    missing gene. Genes to query are supplied at every forward call.

    ``gene_embeddings`` may be shared by all cells (``[G, E]``), supplied per
    cell set (``[B, G, E]``), or supplied per cell (``[B, S, G, E]``).  The
    last form is useful when a batch contains datasets with different measured
    panels.  Genes can be evaluated in chunks to keep inference over the VCC
    18,533-gene panel memory bounded.
    """

    def __init__(
        self,
        latent_dim: int,
        gene_embedding_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_activation: str = "softplus",
        num_fallback_genes: int = 0,
        fusion_mode: str = "add",
        use_gene_baseline: bool = False,
        predict_residual: bool = False,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if output_activation not in {"identity", "relu", "softplus"}:
            raise ValueError("output_activation must be one of: identity, relu, softplus")
        if fusion_mode not in {"add", "concat"}:
            raise ValueError("fusion_mode must be one of: add, concat")
        if use_gene_baseline and fusion_mode != "concat":
            raise ValueError("use_gene_baseline requires fusion_mode='concat'")
        if predict_residual and not use_gene_baseline:
            raise ValueError("predict_residual requires use_gene_baseline=True")

        self.latent_dim = int(latent_dim)
        self.gene_embedding_dim = int(gene_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_activation = output_activation
        self.num_fallback_genes = int(num_fallback_genes)
        self.fusion_mode = fusion_mode
        self.use_gene_baseline = bool(use_gene_baseline)
        self.predict_residual = bool(predict_residual)
        if self.num_fallback_genes < 0:
            raise ValueError("num_fallback_genes cannot be negative")

        self.cell_projection = nn.Linear(self.latent_dim, self.hidden_dim)
        self.gene_projection = nn.Linear(self.gene_embedding_dim, self.hidden_dim)
        if self.num_fallback_genes:
            # Learn missing genes in projected decoder space instead of
            # fabricating protein-language-model vectors.
            self.fallback_base = nn.Parameter(torch.zeros(self.hidden_dim))
            self.fallback_residual = nn.Embedding(self.num_fallback_genes, self.hidden_dim)
            nn.init.normal_(self.fallback_residual.weight, mean=0.0, std=0.02)
        else:
            self.register_parameter("fallback_base", None)
            self.fallback_residual = None

        head_dim = self.hidden_dim if fusion_mode == "add" else 2 * self.hidden_dim
        if self.use_gene_baseline:
            head_dim += 1
        layers = []
        for _ in range(n_layers - 1):
            layers.extend(
                [
                    nn.LayerNorm(head_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_dim, self.hidden_dim),
                ]
            )
            head_dim = self.hidden_dim
        layers.extend([nn.LayerNorm(head_dim), nn.GELU(), nn.Linear(head_dim, 1)])
        self.shared_head = nn.Sequential(*layers)

    def gene_dim(self):
        """A panel-free decoder has no fixed output dimension."""
        return None

    @staticmethod
    def _expand_gene_embeddings(gene_embeddings: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = latent.shape
        if gene_embeddings.dim() == 2:
            return gene_embeddings[None, None].expand(batch_size, seq_len, -1, -1)
        if gene_embeddings.dim() == 3:
            if gene_embeddings.shape[0] == batch_size:
                return gene_embeddings[:, None].expand(-1, seq_len, -1, -1)
            if gene_embeddings.shape[0] == batch_size * seq_len:
                return gene_embeddings.reshape(batch_size, seq_len, *gene_embeddings.shape[1:])
        if gene_embeddings.dim() == 4 and gene_embeddings.shape[:2] == (batch_size, seq_len):
            return gene_embeddings
        raise ValueError(
            "gene_embeddings must have shape [G,E], [B,G,E], [B*S,G,E], or [B,S,G,E] "
            f"for latent shape {tuple(latent.shape)}; got {tuple(gene_embeddings.shape)}"
        )

    @staticmethod
    def _expand_fallback_ids(
        fallback_ids: torch.Tensor,
        latent: torch.Tensor,
        n_genes: int,
    ) -> torch.Tensor:
        """Broadcast stable fallback IDs to ``[B,S,G]``; ``-1`` means pretrained."""
        batch_size, seq_len, _ = latent.shape
        if fallback_ids.dim() == 1 and fallback_ids.shape[0] == n_genes:
            return fallback_ids[None, None].expand(batch_size, seq_len, -1)
        if fallback_ids.dim() == 2:
            if fallback_ids.shape == (batch_size, n_genes):
                return fallback_ids[:, None].expand(-1, seq_len, -1)
            if fallback_ids.shape == (batch_size * seq_len, n_genes):
                return fallback_ids.reshape(batch_size, seq_len, n_genes)
        if fallback_ids.dim() == 3 and fallback_ids.shape == (batch_size, seq_len, n_genes):
            return fallback_ids
        raise ValueError(
            "fallback_ids must have shape [G], [B,G], [B*S,G], or [B,S,G] "
            f"for latent shape {tuple(latent.shape)} and G={n_genes}; got {tuple(fallback_ids.shape)}"
        )

    def _activate(self, value: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "softplus":
            return torch.nn.functional.softplus(value)
        if self.output_activation == "relu":
            return torch.relu(value)
        return value

    @staticmethod
    def _expand_gene_baseline(
        gene_baseline: torch.Tensor, latent: torch.Tensor, n_genes: int
    ) -> torch.Tensor:
        """Broadcast a control-expression baseline to ``[B,S,G]``."""
        batch_size, seq_len, _ = latent.shape
        if gene_baseline.dim() == 1 and gene_baseline.shape[0] == n_genes:
            return gene_baseline[None, None].expand(batch_size, seq_len, -1)
        if gene_baseline.dim() == 2:
            if gene_baseline.shape == (batch_size, n_genes):
                return gene_baseline[:, None].expand(-1, seq_len, -1)
            if gene_baseline.shape == (batch_size * seq_len, n_genes):
                return gene_baseline.reshape(batch_size, seq_len, n_genes)
        if gene_baseline.dim() == 3 and gene_baseline.shape == (batch_size, seq_len, n_genes):
            return gene_baseline
        raise ValueError(
            "gene_baseline must have shape [G], [B,G], [B*S,G], or [B,S,G] "
            f"for latent shape {tuple(latent.shape)} and G={n_genes}; got {tuple(gene_baseline.shape)}"
        )

    def forward(
        self,
        latent: torch.Tensor,
        gene_embeddings: torch.Tensor,
        *,
        fallback_ids: Optional[torch.Tensor] = None,
        gene_baseline: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Return expression with shape ``[B, S, G]``."""
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
        if latent.dim() != 3:
            raise ValueError(f"latent must have shape [B,S,D] or [S,D], got {tuple(latent.shape)}")
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent.shape[-1]}")

        genes = self._expand_gene_embeddings(gene_embeddings, latent)
        if genes.shape[-1] != self.gene_embedding_dim:
            raise ValueError(f"Expected gene_embedding_dim={self.gene_embedding_dim}, got {genes.shape[-1]}")

        n_genes = genes.shape[-2]
        if n_genes == 0:
            return latent.new_empty((*latent.shape[:2], 0))
        chunk_size = n_genes if chunk_size is None else int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        cell_features = self.cell_projection(latent).unsqueeze(-2)
        expanded_baseline = None
        if self.use_gene_baseline:
            if gene_baseline is None:
                raise ValueError("gene_baseline is required when use_gene_baseline=True")
            expanded_baseline = self._expand_gene_baseline(
                gene_baseline.to(device=latent.device, dtype=latent.dtype), latent, n_genes
            )
        expanded_fallback_ids = None
        if fallback_ids is not None:
            expanded_fallback_ids = self._expand_fallback_ids(
                fallback_ids.to(latent.device), latent, n_genes
            ).long()
            active = expanded_fallback_ids >= 0
            if active.any():
                if self.fallback_residual is None or self.fallback_base is None:
                    raise ValueError("fallback_ids contain active IDs but num_fallback_genes=0")
                largest_id = int(expanded_fallback_ids[active].max().item())
                if largest_id >= self.num_fallback_genes:
                    raise ValueError(
                        f"fallback ID {largest_id} is outside configured range [0, {self.num_fallback_genes})"
                    )
        outputs = []
        for start in range(0, n_genes, chunk_size):
            gene_features = self.gene_projection(genes[..., start : start + chunk_size, :])
            if expanded_fallback_ids is not None:
                ids = expanded_fallback_ids[..., start : start + chunk_size]
                active = ids >= 0
                if active.any():
                    learned = self.fallback_base + self.fallback_residual(ids.clamp_min(0))
                    gene_features = torch.where(active.unsqueeze(-1), learned, gene_features)
            if self.fusion_mode == "add":
                fused = cell_features + gene_features
            else:
                expanded_cells = cell_features.expand(*gene_features.shape[:-1], -1)
                parts = [expanded_cells, gene_features]
                if expanded_baseline is not None:
                    parts.append(expanded_baseline[..., start : start + chunk_size, None])
                fused = torch.cat(parts, dim=-1)
            values = self.shared_head(fused).squeeze(-1)
            if self.predict_residual:
                values = values + expanded_baseline[..., start : start + chunk_size]
            outputs.append(self._activate(values))
        return torch.cat(outputs, dim=-1)


class FinetuneVCICountsDecoder(nn.Module):
    def __init__(
        self,
        genes=None,
        adata=None,
        # checkpoint: Optional[str] = "/large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt",
        # config: Optional[str] = "/large_storage/ctc/userspace/aadduri/SE-600M/config.yaml",
        checkpoint: Optional[str] = "/home/aadduri/vci_pretrain/vci_1.4.4/vci_1.4.4_v7.ckpt",
        config: Optional[str] = "/home/aadduri/vci_pretrain/vci_1.4.4/config.yaml",
        latent_dim: int = 1034,  # total input dim (cell emb + optional ds emb)
        read_depth: float = 4.0,
        ds_emb_dim: int = 10,  # dataset embedding dim at the tail of input
        hidden_dim: int = 512,
        dropout: float = 0.1,
        basal_residual: bool = False,
        train_binary_decoder: bool = True,
    ):
        super().__init__()
        # Initialize finetune helper and model from a single checkpoint
        if config is None:
            raise ValueError(
                "FinetuneVCICountsDecoder requires a VCI/SE config. Set kwargs.vci_config or env STATE_VCI_CONFIG."
            )
        self.finetune = Finetune(cfg=OmegaConf.load(config), train_binary_decoder=train_binary_decoder)
        self.finetune.load_model(checkpoint)
        # Resolve genes: prefer explicit list; else infer from anndata if provided
        if genes is None and adata is not None:
            try:
                genes = self.finetune.genes_from_adata(adata)
            except Exception as e:
                raise ValueError(f"Failed to infer genes from AnnData: {e}")
        if genes is None:
            raise ValueError("FinetuneVCICountsDecoder requires 'genes' or 'adata' to derive gene names")
        self.genes = genes
        # Keep read_depth as a learnable parameter so decoded counts can adapt
        self.read_depth = nn.Parameter(torch.tensor(read_depth, dtype=torch.float), requires_grad=True)
        self.basal_residual = basal_residual
        self.ds_emb_dim = int(ds_emb_dim) if ds_emb_dim is not None else 0
        self.input_total_dim = int(latent_dim)

        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.genes)),
        )

        self.gene_decoder_proj = nn.Sequential(
            nn.Linear(len(self.genes), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(self.genes)),
        )

        self.binary_decoder = self.finetune.model.binary_decoder  # type: ignore

        # Validate that all requested genes exist in the pretrained checkpoint's embeddings
        pe = getattr(self.finetune, "protein_embeds", {})
        self.present_mask = [g in pe for g in self.genes]
        self.missing_positions = [i for i, g in enumerate(self.genes) if g not in pe]
        self.missing_genes = [self.genes[i] for i in self.missing_positions]
        total_req = len(self.genes)
        found = total_req - len(self.missing_positions)
        total_pe = len(pe) if hasattr(pe, "__len__") else -1
        miss_pct = (len(self.missing_positions) / total_req) if total_req > 0 else 0.0
        logger.info(
            f"FinetuneVCICountsDecoder gene check: requested={total_req}, found={found}, missing={len(self.missing_positions)} ({miss_pct:.1%}), all_embeddings_size={total_pe}"
        )

        # Create learnable embeddings for missing genes in the post-ESM gene embedding space
        if len(self.missing_positions) > 0:
            # Infer gene embedding output dimension by a dry-run through gene_embedding_layer
            try:
                sample_vec = next(iter(pe.values())).to(self.finetune.model.device)
                if sample_vec.dim() == 1:
                    sample_vec = sample_vec.unsqueeze(0)
                gene_embed_dim = self.finetune.model.gene_embedding_layer(sample_vec).shape[-1]
            except Exception:
                # Conservative fallback
                gene_embed_dim = 1024

            self.missing_table = nn.Embedding(len(self.missing_positions), gene_embed_dim)
            nn.init.normal_(self.missing_table.weight, mean=0.0, std=0.02)
            # For user visibility
            try:
                self.finetune.missing_genes = self.missing_genes
            except Exception:
                pass
        else:
            # Register a dummy buffer so attributes exist
            self.missing_table = None

        # Ensure the wrapped Finetune helper creates its own missing-table parameters
        # prior to Lightning's checkpoint load. Otherwise the checkpoint will contain
        # weights like `gene_decoder.finetune.missing_table.weight` that are absent
        # from a freshly constructed module, triggering "unexpected key" errors.
        try:
            with torch.no_grad():
                self.finetune.get_gene_embedding(self.genes)
        except Exception as exc:
            logger.debug(f"Deferred Finetune missing-table initialization failed: {exc}")

    def gene_dim(self):
        return len(self.genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, total_dim]
        if x.dim() != 3:
            x = x.unsqueeze(0)
        batch_size, seq_len, total_dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, total_dim)

        # Split cell and dataset embeddings
        if self.ds_emb_dim > 0:
            cell_embeds = x_flat[:, : total_dim - self.ds_emb_dim]
            ds_emb = x_flat[:, total_dim - self.ds_emb_dim : total_dim]
        else:
            cell_embeds = x_flat
            ds_emb = None

        # Prepare gene embeddings (replace any missing with learned vectors)
        gene_embeds = self.finetune.get_gene_embedding(self.genes)
        if self.missing_table is not None and len(self.missing_positions) > 0:
            device = gene_embeds.device
            learned = self.missing_table.weight.to(device)
            idx = torch.tensor(self.missing_positions, device=device, dtype=torch.long)
            gene_embeds = gene_embeds.clone()
            gene_embeds.index_copy_(0, idx, learned)
        # Ensure embeddings live on the same device as cell_embeds
        if gene_embeds.device != cell_embeds.device:
            gene_embeds = gene_embeds.to(cell_embeds.device)

        # RDA read depth vector (if enabled in SE model)
        use_rda = getattr(self.finetune.model.cfg.model, "rda", False)
        task_counts = None
        if use_rda:
            task_counts = self.read_depth.expand(cell_embeds.shape[0])
            if task_counts.device != cell_embeds.device:
                task_counts = task_counts.to(cell_embeds.device)

        # Binary decoder forward with safe dtype handling.
        # - On CUDA: enable bf16 autocast for speed.
        # - On CPU: ensure inputs match decoder weight dtype to avoid BF16/FP32 mismatch.
        device_type = "cuda" if cell_embeds.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
            merged = self.finetune.model.resize_batch(
                cell_embeds=cell_embeds, task_embeds=gene_embeds, task_counts=task_counts, ds_emb=ds_emb
            )

            # Align input dtype with decoder weights when autocast is not active (e.g., CPU path)
            dec_param_dtype = next(self.binary_decoder.parameters()).dtype
            if device_type != "cuda" and merged.dtype != dec_param_dtype:
                merged = merged.to(dec_param_dtype)

            logprobs = self.binary_decoder(merged)
            if logprobs.dim() == 3 and logprobs.size(-1) == 1:
                logprobs = logprobs.squeeze(-1)

        # Reshape back to [B, S, gene_dim]
        decoded_gene = logprobs.view(batch_size, seq_len, len(self.genes))

        # Match dtype for post-decoder projection to avoid mixed-dtype matmul
        proj_param_dtype = next(self.gene_decoder_proj.parameters()).dtype
        if decoded_gene.dtype != proj_param_dtype:
            decoded_gene = decoded_gene.to(proj_param_dtype)
        decoded_gene = decoded_gene + self.gene_decoder_proj(decoded_gene)

        # Optional residual from latent decoder (operates on full input features)
        ld_param_dtype = next(self.latent_decoder.parameters()).dtype
        x_flat_for_ld = x_flat if x_flat.dtype == ld_param_dtype else x_flat.to(ld_param_dtype)
        decoded_x = self.latent_decoder(x_flat_for_ld).view(batch_size, seq_len, len(self.genes))
        return torch.nn.functional.relu(decoded_gene + decoded_x)
