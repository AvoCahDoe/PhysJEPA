# Demo guide (`paper_mid`)

Use this for portfolio walkthroughs, interviews, or slide decks. All numbers are from `runs/comparison.json` and `runs/ablations.json` (1000 train episodes, 80 epochs, RTX 4070).

## One-liner

> I designed a controlled diagnostic to test whether latent-space prediction induces physical abstractions that pixel-space prediction does not, using violation-of-expectation methodology from developmental psychology. At mid scale, JEPA shows a selective surprise spike on impossible bounces but not on teleport-under-occlusion — a nuanced partial result, not a clean win for either side.

## Headline numbers to cite

| Metric | JEPA | Pixel | Takeaway |
|--------|------|-------|----------|
| xy probe val R² | **0.421** | 0.415 | Both encode position; JEPA slightly ahead |
| visible probe val acc | **0.944** | 0.940 | High for both (appearance + occlusion cues) |
| VoE impossible_bounce Δ | **+0.028** | — | JEPA spikes more at impossible bounce |
| VoE teleport Δ | −0.018 | — | JEPA *less* surprised (permanence not learned) |

## Live demo walkthrough

**URL:** [physjepa.vercel.app](https://physjepa.vercel.app)

Suggested order:

1. **Home** — headline metrics and project overview
2. **[/try](https://physjepa.vercel.app/try)** — scrub impossible bounce pair to **t***; show JEPA surprise gap on impossible branch
3. **[/results](https://physjepa.vercel.app/results)** — interpretation cards, then Comparison + VoE curves + Probes
4. **[/docs](https://physjepa.vercel.app/docs)** — JEPA objective and VoE spike formula (if audience is technical)

Local dev:

```bash
python scripts/sync_viz_fixtures.py --jepa-run paper_mid --pixel-run paper_mid
cd viz/dashboard && npm run dev
```

## Static slides (`docs/figures/`)

| File | Say this |
|------|----------|
| `fig_train_loss.png` | Both models converge over 80 epochs on 1k episodes |
| `fig_voe_bounce.png` | **Best JEPA story** — impossible branch surprises more at collision violation |
| `fig_voe_teleport.png` | Negative spike — model does not flag teleport under occlusion |
| `fig_comparison_delta.png` | Side-by-side headline: bounce vs teleport |
| `fig_ablation_occlusion.png` | Occlusion length does not give clean permanence curve |
| `fig_ablation_arch.png` | Transformer predictor yields largest mean VoE spike |

Regenerate: `python scripts/export_report_figures.py`

## Written report

Full methods + discussion: [`report.md`](report.md)

## Honest framing (avoid overclaiming)

- VoE set is small (5 pairs per violation type).
- Only **impossible bounce** shows a clear JEPA > pixel VoE advantage.
- Teleport / stop-without-collision are weak or inverted — object permanence is not demonstrated.
- 2D 64×64 sandbox — not natural video or 3D physics.

## If asked “what’s next?”

- Scale data and VoE probes; train longer or on harder violation schedules.
- ONNX/WASM inference in the browser for true live model runs.
- Compare to V-JEPA / world-model baselines on the same VoE JSON.
