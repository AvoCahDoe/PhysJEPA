# Demo guide (`paper_mid`)

Use this for a live walkthrough, SOP interview, or slide deck. All numbers are from `runs/comparison.json` and `runs/ablations.json` (1000 train episodes, 80 epochs, RTX 4070).

## One-liner (SOP)

> I designed a controlled diagnostic to test whether latent-space prediction induces physical abstractions that pixel-space prediction does not, using violation-of-expectation methodology from developmental psychology. At mid scale, JEPA shows a selective surprise spike on impossible bounces but not on teleport-under-occlusion — a nuanced partial result, not a clean win for either side.

## Headline numbers to cite

| Metric | JEPA | Pixel | Takeaway |
|--------|------|-------|----------|
| xy probe val R² | **0.421** | 0.415 | Both encode position; JEPA slightly ahead |
| visible probe val acc | **0.944** | 0.940 | High for both (appearance + occlusion cues) |
| VoE impossible_bounce Δ | **+0.028** | — | JEPA spikes more at impossible bounce |
| VoE teleport Δ | −0.018 | — | JEPA *less* surprised (permanence not learned) |

## What to show (pick 2–3)

### Live dashboard

```bash
python scripts/sync_viz_fixtures.py --jepa-run paper_mid --pixel-run paper_mid
cd viz/dashboard && npm run dev
```

Open http://localhost:5173 and walk through tabs in this order:

1. **Training** — 80-epoch JEPA vs pixel loss curves (not smoke).
2. **Comparison** — bar chart of Δ VoE spike; point at **impossible bounce** (+0.028).
3. **VoE Surprise** — possible vs impossible curves; note divergence at `t*` for bounce (switch mentally — dashboard defaults to teleport; cite static figure below for bounce).
4. **Linear Probes** — xy / visible table.
5. **Ablations** — transformer highest mean VoE spike; medium occlusion worst teleport spike.

### Static slides (`docs/figures/`)

| File | Say this |
|------|----------|
| `fig_train_loss.png` | Both models converge over 80 epochs on 1k episodes |
| `fig_voe_bounce.png` | **Best JEPA story** — impossible branch surprises more at collision violation |
| `fig_voe_teleport.png` | Negative spike — model does not flag teleport under occlusion |
| `fig_comparison_delta.png` | Side-by-side headline: bounce vs teleport |
| `fig_ablation_occlusion.png` | Occlusion length does not give clean permanence curve |
| `fig_ablation_arch.png` | Transformer predictor yields largest mean VoE spike |

Regenerate: `python scripts/export_report_figures.py`

### Written report

Full methods + discussion: [`report.md`](report.md)

## Honest framing (avoid overclaiming)

- VoE set is small (5 pairs per violation type).
- Only **impossible bounce** shows a clear JEPA > pixel VoE advantage.
- Teleport / stop-without-collision are weak or inverted — object permanence is not demonstrated.
- 2D 64×64 sandbox — not natural video or 3D physics.

## If asked “what’s next?”

- Scale data and VoE probes; train longer or on harder violation schedules.
- Per-pair VoE replay in the dashboard (possible vs impossible side-by-side).
- Compare to V-JEPA / world-model baselines on the same VoE JSON.
