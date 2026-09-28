# Paper-style decoder profile

The repository now has two compatible panel-free decoder modes:

| profile | decoder input | output |
|---|---|---|
| `state_vcc` | `z_pert`, gene embedding, matched-control gene baseline | control-calibrated residual prediction |
| `state_vcc_paper` | one-hot-equivalent perturbation ID, `z_pert`, gene embedding, scalar `r_depth` | direct log1p(CP10K) prediction |

The paper-style profile implements

```text
r_depth = mean(log1p(CP10K(control cell))) over expressed genes
x_hat[g] = f(z_pert, e_gene[g], r_depth)
```

For perturbations, the profile uses `nn.Embedding(D, 768)`. This is the
memory-efficient implementation of a categorical one-hot vector followed by a
linear projection; control uses the `-1` sentinel and contributes a zero
perturbation token.

It does not evaluate the decoder at `z_control` and does not add a measured
control expression vector in gene space. The ST input still contains the
control-cell set, so basal information enters upstream through ST. The VCC
profile remains available for comparison and uses the matched-control baseline
to cancel decoder-wide offsets.

Run the paper profile with:

```bash
external/state-env/bin/python scripts/train_vcc.py h1-loco-joint \
  --data-config vcc_panel_free_paper \
  --model-config state_vcc_paper \
  --gpu 2 --max-steps 1000 --val-freq 100
```

The scalar read-depth feature is emitted by the data module as
`decoder_read_depth: [B, S, 1]`. At VCC inference it is computed from each
donor control cell before the four-cell pooling step. Existing checkpoints and
the original `state_vcc` profile remain loadable; the new profile only changes
the decoder configuration and its required batch key.
