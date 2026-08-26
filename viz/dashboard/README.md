# physjepa results dashboard

Vite + React + Recharts UI for `paper_mid` metrics.

## Run locally

```bash
# From repo root — refresh fixtures from latest runs
python scripts/sync_viz_fixtures.py --jepa-run paper_mid --pixel-run paper_mid

cd viz/dashboard
npm install
npm run dev    # http://localhost:5173
```

## Tabs

| Tab | Data source |
|-----|-------------|
| Comparison | `public/fixtures/sample_comparison.json` |
| VoE Surprise | `public/fixtures/sample_eval.json` |
| Linear Probes | comparison JSON (probe columns) |
| Ablations | `public/fixtures/sample_ablations.json` |
| Training | `jepa_summary.json`, `pixel_summary.json` |
| Episode Replay | `sample_episode/` |

## Override fixtures (query params)

```
?comparison=/fixtures/sample_comparison.json
&voe=/fixtures/sample_eval.json
&jepa_summary=/fixtures/jepa_summary.json
&pixel_summary=/fixtures/pixel_summary.json
&ablations=/fixtures/sample_ablations.json
```

## Build static site

```bash
npm run build
npm run preview
```

Schema details: [`../SCHEMA.md`](../SCHEMA.md). Demo script: [`../../docs/DEMO.md`](../../docs/DEMO.md).
