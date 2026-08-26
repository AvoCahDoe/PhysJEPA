# Deploying the live demo (Vercel)

The interactive dashboard is a **static Vite SPA** — no Python/PyTorch on the server. All metrics and VoE replay frames ship as JSON/PNG under `viz/dashboard/public/fixtures/`.

## One-time setup

1. Push the repo to GitHub: [AvoCahDoe/PhysJEPA](https://github.com/AvoCahDoe/PhysJEPA)
2. Open [vercel.com/new](https://vercel.com/new) and import **PhysJEPA**
3. Vercel reads [`vercel.json`](../vercel.json) at repo root:
   - **Build:** `cd viz/dashboard && npm ci && npm run build`
   - **Output:** `viz/dashboard/dist`
4. Deploy — **no environment variables** required
5. Production URL: **https://physjepa.vercel.app**

## Local preview (production build)

```bash
cd viz/dashboard
npm install
npm run build
npm run preview
```

## Refresh fixtures after a new train

```bash
python scripts/export_voe_demo_pairs.py
python scripts/sync_viz_fixtures.py --jepa-run paper_mid --pixel-run paper_mid
git add viz/fixtures viz/dashboard/public/fixtures
git commit -m "Refresh demo fixtures"
git push
```

Vercel redeploys automatically on push to `main`.

## What's on the live site

| Tab | Content |
|-----|---------|
| Comparison | JEPA vs pixel VoE spike deltas |
| VoE Surprise | Aggregate surprise curves by violation type |
| **VoE Replay** | Side-by-side possible vs impossible rollouts |
| Linear Probes | xy / velocity / mass / visibility |
| Ablations | Occlusion duration + predictor architecture |
| Training | 80-epoch loss curves |
| Episode Replay | Single train rollout scrubber |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| 404 on `/fixtures/...` | Ensure `public/fixtures/` is committed; run `sync_viz_fixtures.py` |
| Blank charts | Check browser console; verify JSON paths in `loadRun.ts` use leading `/` |
| VoE Replay empty | Run `export_voe_demo_pairs.py` (needs local `data/voe/` from `generate_voe.py`) |
