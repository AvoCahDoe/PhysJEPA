# Documentation

| Doc | Purpose |
|-----|---------|
| [`report.md`](report.md) | Paper-style write-up with mid-scale (`paper_mid`) results |
| [`DEMO.md`](DEMO.md) | Live demo / SOP talking points + graph checklist |
| [`figures/`](figures/) | Static PNGs for slides (regenerate with `python scripts/export_report_figures.py`) |

**Quick links:** root [`README.md`](../README.md) · JSON schema [`viz/SCHEMA.md`](../viz/SCHEMA.md) · research brief [`plan.md`](../plan.md)

## Regenerate everything for a demo

```bash
python scripts/sync_viz_fixtures.py --jepa-run paper_mid --pixel-run paper_mid
python scripts/export_report_figures.py
cd viz/dashboard && npm run dev
```
