# Episode export schema (React contract)

`schema_version: 1`

This document is the stable contract between Python generators and a future React results viewer. Do not break fields without bumping `schema_version`.

## Episode directory

```
episode_000042/
  frames/
    000000.png
    000001.png
    ...
  meta.json
```

Frames are **64×64 RGB PNG**. Indices are zero-padded to 6 digits and match `trajectory[t].t`.

## `meta.json`

| Field | Type | Notes |
|-------|------|--------|
| `schema_version` | int | Currently `1` |
| `split` | `"train"` \| `"voe"` | |
| `seed` | int \| null | |
| `T` | int | Number of frames |
| `resolution` | `[W, H]` | e.g. `[64, 64]` |
| `fps` | int | Display / timing hint |
| `violation` | object \| null | Null for train and VoE **possible** rollouts |
| `objects` | array | Static object descriptors |
| `occluders` | array | Visual occluders (may be empty) |
| `trajectory` | array | Per-frame body states |
| `pair_role` | string (optional) | `"possible"` \| `"impossible"` on VoE episodes |
| `pair_id` | string (optional) | VoE pair id |

### `violation`

```json
{
  "type": "teleport_occlusion",
  "t_star": 12,
  "pair_id": "teleport_occlusion_s9001"
}
```

Types: `teleport_occlusion`, `pass_through_wall`, `stop_without_collision`, `impossible_bounce`.

### `objects[]`

```json
{ "id": 0, "shape": "disk", "mass": 1.0, "color": [220, 80, 70], "radius": 0.06 }
```

Boxes use `width` / `height` instead of `radius`.

### `trajectory[]`

```json
{
  "t": 0,
  "bodies": [
    {
      "id": 0,
      "x": 0.2,
      "y": 0.5,
      "vx": 0.9,
      "vy": 0.0,
      "angle": 0.0,
      "visible": true,
      "mass": 1.0
    }
  ]
}
```

`visible: false` means the body **center** lies inside a visual occluder — the body is **omitted from RGB** but still present in physics / metadata (object permanence).

World coordinates: origin bottom-left, arena size from sim config (default 1.0 × 1.0).

## Split indexes

### `data/train/index.json`

```json
{
  "schema_version": 1,
  "episodes": [
    { "seed": 0, "path": "episode_000000", "T": 24, "n_objects": 2 }
  ]
}
```

Paths are relative to the index file’s directory.

### `data/voe/index.json`

```json
{
  "schema_version": 1,
  "pairs": [
    {
      "pair_id": "teleport_occlusion_s9001",
      "violation_type": "teleport_occlusion",
      "t_star": 12,
      "matched_seed": 9001,
      "possible": "teleport_occlusion_s9001/possible",
      "impossible": "teleport_occlusion_s9001/impossible"
    }
  ]
}
```

Each pair folder also has `pair_meta.json`.

## React usage (later)

1. Fetch `index.json` for a split.
2. Load `meta.json` + scrub `frames/{t}.png` on a canvas.
3. Overlay trajectory / surprise curves (Recharts) keyed by `t`, using `violation.t_star` as a marker.

Fixture for UI development without regenerating data: [`fixtures/sample_episode/`](fixtures/sample_episode/).

## Training run (`runs/jepa/<run_id>/`)

Written by `scripts/train_jepa.py`. React can plot curves without re-running Python.

### `summary.json`

```json
{
  "schema_version": 1,
  "run_id": "20260101T120000Z",
  "curves": {
    "step": [1, 10, 20],
    "loss": [0.5, 0.3, 0.2],
    "latent_std": [0.4, 0.35, 0.33],
    "latent_norm": [8.0, 7.5, 7.2],
    "val_loss": [0.4, 0.28]
  },
  "best_val_loss": 0.28,
  "config": {},
  "epochs": 20,
  "steps": 200
}
```

| Field | Notes |
|-------|--------|
| `curves.loss` | Train smooth-L1 JEPA loss at logged steps |
| `curves.latent_std` | Mean std of context latents (collapse alarm if ~0) |
| `curves.val_loss` | Present when validation ran at a log step (may be shorter than `step`) |
| `best_val_loss` | Best validation loss seen |

### `metrics.jsonl`

One JSON object per log line (same fields as curve points, plus `epoch`, `ema_momentum`).

### Checkpoints

- `ckpt_last.pt` — last epoch
- `ckpt_best.pt` — best `val_loss` (when val split non-empty)

## Eval results (`runs/jepa/<run_id>/eval/`)

### `probes.json`

Frozen linear probes on encoder latents (body id 0).

```json
{
  "schema_version": 1,
  "run_id": "paper_mid",
  "ckpt": "ckpt_last.pt",
  "targets": {
    "xy": {"val_mse": 0.01, "val_r2": 0.8, "train_mse": 0.008, "train_r2": 0.85},
    "vxvy": {"val_mse": 0.02, "val_r2": 0.5},
    "mass": {"val_mse": 0.05, "val_r2": 0.1},
    "visible": {"val_acc": 0.9, "train_acc": 0.92, "val_bce": 0.3}
  }
}
```

### `voe_surprise.json`

Per-timestep JEPA smooth-L1 prediction error on matched possible/impossible pairs.

```json
{
  "schema_version": 1,
  "run_id": "paper_mid",
  "context_len": 4,
  "by_type": {
    "teleport_occlusion": {
      "t_star": 12,
      "t": [0, 1],
      "possible_mean": [null, 0.01],
      "impossible_mean": [null, 0.02],
      "possible_std": [null, 0.001],
      "impossible_std": [null, 0.002],
      "spike_score": 0.05,
      "pre_tstar_abs_gap": 0.0001,
      "n_pairs": 5
    }
  },
  "pairs": []
}
```

`spike_score` = mean impossible surprise on `[t_star, t_star+w)` minus the same for possible (`w` = `spike_window`). Timesteps without a full context/future window are `null`.

For **pixel baseline**, `surprise_metric` is `pixel_smooth_l1` (frame reconstruction error). For **JEPA**, `latent_smooth_l1`.

## Model comparison (`runs/comparison.json`)

Side-by-side JEPA vs pixel baseline from `scripts/eval_compare.py`.

```json
{
  "schema_version": 1,
  "jepa": { "probes": {}, "voe_by_type": {}, "voe_surprise_metric": "latent_smooth_l1" },
  "pixel": { "probes": {}, "voe_by_type": {}, "voe_surprise_metric": "pixel_smooth_l1" },
  "delta_voe_spike_jepa_minus_pixel": { "teleport_occlusion": 0.01 }
}
```

Positive `delta_voe_spike` means JEPA shows a larger VoE spike than the pixel baseline for that violation type.

## React dashboard (`viz/dashboard/`)

Vite + React + Recharts app that consumes the fixtures above.

```bash
python scripts/sync_viz_fixtures.py --jepa-run paper_mid --pixel-run paper_mid
cd viz/dashboard && npm install && npm run dev
```

`sync_viz_fixtures.py` copies `runs/comparison.json`, JEPA VoE eval, ablations, and `runs/{jepa,pixel}/<run_id>/summary.json` into `viz/fixtures/` and `viz/dashboard/public/fixtures/`. Defaults: `--jepa-run paper_mid --pixel-run paper_mid`.

Default loads `/fixtures/sample_comparison.json`, `sample_eval.json`, `jepa_summary.json`, `pixel_summary.json`, `sample_ablations.json`, and `sample_episode/`. Override with query params: `?comparison=...&voe=...&jepa_summary=...&pixel_summary=...&ablations=...`. See [`dashboard/README.md`](dashboard/README.md) and [`docs/DEMO.md`](../docs/DEMO.md).

## Ablations (`runs/ablations.json`)

From `scripts/run_ablations.py`.

```json
{
  "schema_version": 1,
  "occlusion_duration": {
    "results": [
      { "label": "short", "mean_occlusion_frames": 3.0, "spike_score": 0.01 }
    ]
  },
  "architecture": {
    "results": [
      {
        "predictor_type": "mlp",
        "mean_voe_spike": 0.0,
        "probes": { "xy": { "val_r2": 0.3 }, "visible": { "val_acc": 0.9 } }
      }
    ]
  }
}
```
