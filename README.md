# PhysJEPA

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black-compatible](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](https://github.com/psf/black)

**Controlled diagnostic for the JEPA physics claim:** does self-supervised *latent* future prediction encode naive physics (object permanence, collisions) better than a matched *pixel* reconstruction baseline?

Evaluation uses **linear probes** and **violation-of-expectation (VoE)** surprise curves in a fully controlled 2D pymunk sandbox (64×64 RGB). Mid-scale run **`paper_mid`**: 1000 train episodes, 80 epochs, JEPA vs pixel on CUDA.

| | JEPA | Pixel |
|---|------|-------|
| xy probe val R² | **0.421** | 0.415 |
| VoE Δ impossible bounce | **+0.028** | (baseline) |
| VoE Δ teleport | −0.018 | (baseline) |

**Headline:** JEPA shows selective surprise on **impossible bounce**; teleport-under-occlusion remains weak — a **nuanced partial** result, not a clean JEPA win.

---

## Links

| Resource | Path |
|----------|------|
| Technical report | [`docs/report.md`](docs/report.md) |
| Demo / SOP script | [`docs/DEMO.md`](docs/DEMO.md) |
| JSON schema (React) | [`viz/SCHEMA.md`](viz/SCHEMA.md) |
| Research brief | [`plan.md`](plan.md) |
| Static figures | [`docs/figures/`](docs/figures/) |

<p align="center">
  <img src="docs/figures/fig_comparison_delta.png" alt="VoE spike delta JEPA minus pixel" width="48%" />
  <img src="docs/figures/fig_voe_bounce.png" alt="VoE impossible bounce curves" width="48%" />
</p>

---

## Quick start

```bash
git clone https://github.com/AvoCahDoe/PhysJEPA.git
cd PhysJEPA

python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# source .venv/bin/activate     # Linux / macOS

pip install -e .
# GPU (recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Smoke test (minutes)

```bash
python scripts/smoke_jepa.py
python scripts/smoke_pixel.py
```

### Full mid-scale pipeline (`paper_mid`)

```bash
python scripts/run_paper_mid.py --device cuda
```

Or step-by-step — see [Phases](#phases) below.

### Live demo dashboard

```bash
python scripts/sync_viz_fixtures.py --jepa-run paper_mid --pixel-run paper_mid
cd viz/dashboard && npm install && npm run dev
# → http://localhost:5173
```

Tabs: Comparison · VoE · Probes · Ablations · Training · Episode replay.

---

## Method (one paragraph)

Procedural pymunk rollouts train a small **JEPA** (CNN encoder + EMA target + MLP/GRU/Transformer predictor) with smooth-L1 latent prediction only — no physics labels. A **pixel baseline** matches capacity but reconstructs frames. Held-out VoE pairs (possible vs impossible) test teleport, wall pass-through, stop-without-collision, and impossible bounce. **Linear probes** on frozen latents read out position, velocity, mass, and visibility.

---

## Phases

<details>
<summary><strong>Phase 1 — Data</strong></summary>

```bash
python scripts/generate_train.py --n 1000 --out data/train
python scripts/generate_voe.py --out data/voe
python scripts/preview_rollout.py data/train/episode_000000
```

</details>

<details>
<summary><strong>Phase 2 — JEPA train</strong></summary>

```bash
python scripts/train_jepa.py --config configs/jepa_paper.yaml --run-id paper_mid --device cuda
```

Artifacts: `runs/jepa/<run_id>/{ckpt_last.pt,summary.json,metrics.jsonl}`

</details>

<details>
<summary><strong>Phase 3 — Probes + VoE</strong></summary>

```bash
python scripts/eval_probes.py --ckpt runs/jepa/paper_mid/ckpt_last.pt --device cuda
python scripts/eval_voe.py --ckpt runs/jepa/paper_mid/ckpt_last.pt --device cuda
```

</details>

<details>
<summary><strong>Phase 4 — Pixel baseline + comparison</strong></summary>

```bash
python scripts/train_pixel.py --config configs/pixel_paper.yaml --run-id paper_mid --device cuda
python scripts/eval_compare.py \
  --jepa-ckpt runs/jepa/paper_mid/ckpt_last.pt \
  --pixel-ckpt runs/pixel/paper_mid/ckpt_last.pt \
  --device cuda
```

</details>

<details>
<summary><strong>Phase 5 — Dashboard</strong></summary>

```bash
python scripts/sync_viz_fixtures.py --jepa-run paper_mid --pixel-run paper_mid
cd viz/dashboard && npm run dev
```

</details>

<details>
<summary><strong>Phase 6 — Ablations</strong></summary>

```bash
python scripts/run_ablations.py --config configs/ablations_paper.yaml --epochs 80 --device cuda
```

Occlusion duration + MLP / GRU / Transformer predictors → `runs/ablations.json`

</details>

<details>
<summary><strong>Phase 7 — Write-up & figures</strong></summary>

```bash
python scripts/export_report_figures.py
```

Report: [`docs/report.md`](docs/report.md)

</details>

---

## Repository layout

```
PhysJEPA/
├── src/physjepa/     # sim, gen, voe, models, train, eval, data
├── scripts/          # CLI entrypoints + run_paper_mid.py
├── configs/          # YAML (jepa_paper, pixel_paper, ablations_paper, …)
├── docs/             # report, DEMO, figures/
├── viz/
│   ├── SCHEMA.md     # JSON contract for dashboard
│   ├── fixtures/     # committed demo JSON (paper_mid metrics)
│   └── dashboard/    # Vite + React + Recharts
└── runs/             # gitignored — train outputs locally
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{physjepa2026,
  author = {El Boubkraoui, Farid},
  title = {PhysJEPA: A Controlled JEPA Diagnostic for Naive Physics},
  year = {2026},
  url = {https://github.com/AvoCahDoe/PhysJEPA}
}
```

See also [`CITATION.cff`](CITATION.cff).

---

## Author

**Farid El Boubkraoui** — [farid.elboubkraoui@w-ays.de](mailto:farid.elboubkraoui@w-ays.de)

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Related work

LeCun JEPA / I-JEPA / V-JEPA · Baillargeon VoE · Piloto et al. intuitive physics · Physion / IntPhys / CATER benchmarks. Full discussion in [`docs/report.md`](docs/report.md).
