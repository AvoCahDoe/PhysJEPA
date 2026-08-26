# Documentation

| Doc | Purpose |
|-----|---------|
| [`report.md`](report.md) | Paper-style write-up with mid-scale (`paper_mid`) results |
| [`DEMO.md`](DEMO.md) | Presentation guide — talking points and walkthrough checklist |
| [`DEPLOY.md`](DEPLOY.md) | Vercel deploy for the interactive demo |
| [`figures/`](figures/) | Static PNGs for slides (regenerate with `python scripts/export_report_figures.py`) |

**Quick links:** [Live demo](https://physjepa.vercel.app) · root [`README.md`](../README.md) · JSON schema [`viz/SCHEMA.md`](../viz/SCHEMA.md)

## Regenerate demo assets

```bash
python scripts/export_voe_demo_pairs.py
python scripts/sync_viz_fixtures.py --jepa-run paper_mid --pixel-run paper_mid
python scripts/export_report_figures.py
cd viz/dashboard && npm run dev
```
