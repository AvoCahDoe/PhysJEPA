import { useState } from "react";
import { AblationsPanel } from "../components/AblationsPanel";
import { ComparisonOverview } from "../components/ComparisonOverview";
import { EpisodePlayer } from "../components/EpisodePlayer";
import { ProbeComparison } from "../components/ProbeComparison";
import { TrainingCurves } from "../components/TrainingCurves";
import { VoEViolationShowcase } from "../components/VoEViolationShowcase";
import { VoESurprisePanel } from "../components/VoESurpriseChart";
import { DEFAULT_VOE_TYPE, useFixtures } from "../hooks/useFixtures";
import { violationTypes } from "../lib/loadRun";
import type { TabId } from "../types/schema";

const TABS: { id: TabId; label: string }[] = [
  { id: "compare", label: "Overview" },
  { id: "voe", label: "VoE curves" },
  { id: "voe_replay", label: "VoE replay" },
  { id: "probes", label: "Probes" },
  { id: "ablations", label: "Ablations" },
  { id: "train", label: "Training" },
  { id: "episode", label: "Episode" },
];

export function ResultsPage() {
  const { data, loading, error } = useFixtures();
  const [tab, setTab] = useState<TabId>("compare");
  const [voeType, setVoeType] = useState(DEFAULT_VOE_TYPE);

  if (loading) {
    return <p className="status page-status">Loading results…</p>;
  }
  if (error || !data) {
    return <p className="error page-status">{error ?? "Failed to load fixtures"}</p>;
  }

  const types = violationTypes(data.voe);
  const activeVoeType = types.includes(voeType) ? voeType : types[0] ?? DEFAULT_VOE_TYPE;

  let content = null;
  switch (tab) {
    case "compare":
      content = <ComparisonOverview data={data.comparison} />;
      break;
    case "voe":
      content = (
        <VoESurprisePanel
          byType={data.voe.by_type}
          selected={activeVoeType}
          onSelect={setVoeType}
        />
      );
      break;
    case "voe_replay":
      content = data.voeDemo?.pairs?.length ? (
        <VoEViolationShowcase
          pairs={data.voeDemo.pairs}
          voe={data.voe}
          comparison={data.comparison}
        />
      ) : (
        <p className="hint">No VoE demo pairs loaded.</p>
      );
      break;
    case "probes":
      content = <ProbeComparison data={data.comparison} />;
      break;
    case "ablations":
      content = data.ablations ? (
        <AblationsPanel data={data.ablations} />
      ) : (
        <p className="hint">No ablations fixture yet.</p>
      );
      break;
    case "train":
      content = (
        <TrainingCurves jepa={data.jepaSummary} pixel={data.pixelSummary} />
      );
      break;
    case "episode":
      content = <EpisodePlayer meta={data.episode} />;
      break;
  }

  return (
    <div className="page results-page">
      <header className="page-header">
        <h1>Results &amp; interpretation</h1>
        <p className="page-lead">
          Mid-scale <code>paper_mid</code> run — 1000 train episodes, 80 epochs, CUDA. Fixtures
          synced from <code>runs/comparison.json</code> and VoE eval exports.
        </p>
      </header>

      <section className="interpret panel">
        <h2>What we found</h2>
        <div className="interpret-grid">
          <article className="interpret-card positive">
            <h3>Position encoding</h3>
            <p>
              JEPA xy val R² ≈ <strong>0.421</strong> vs pixel 0.415 — position is partially linear
              in both; JEPA slightly ahead.
            </p>
          </article>
          <article className="interpret-card positive">
            <h3>Impossible bounce VoE</h3>
            <p>
              Selective spike: Δ (JEPA − pixel) ≈ <strong>+0.028</strong>. The latent predictor
              reacts more sharply to wrong restitution than pixel reconstruction.
            </p>
          </article>
          <article className="interpret-card muted">
            <h3>Weak elsewhere</h3>
            <p>
              Teleport and stop-without-collision show weak or negative VoE deltas. Pass-through-wall
              spikes are near zero — a nuanced, not universal, JEPA win.
            </p>
          </article>
          <article className="interpret-card muted">
            <h3>Velocity &amp; mass</h3>
            <p>
              Velocity R² near chance; mass essentially unreadable. Visibility ~94% acc for both
              models — occlusion is encoded but dynamics remain hard.
            </p>
          </article>
        </div>
      </section>

      <div className="results-tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={tab === t.id ? "tab active" : "tab"}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="results-content stack">{content}</div>
    </div>
  );
}
