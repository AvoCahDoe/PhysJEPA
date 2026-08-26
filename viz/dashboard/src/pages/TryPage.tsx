import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { LiveInferencePanel } from "../components/LiveInferencePanel";
import { VoEPairReplay } from "../components/VoEPairReplay";
import { DEFAULT_VOE_TYPE, useFixtures } from "../hooks/useFixtures";
import { violationTypes } from "../lib/loadRun";

export function TryPage() {
  const { data, loading, error } = useFixtures();
  const [voeType, setVoeType] = useState(DEFAULT_VOE_TYPE);
  const [currentT, setCurrentT] = useState(0);

  const voeSummary = data?.voe.by_type[voeType];
  const demoEntry = data?.voeDemo?.pairs.find((p) => p.violation_type === voeType);
  const tStar = demoEntry?.t_star ?? voeSummary?.t_star ?? 0;
  const deltaVsPixel = data?.comparison.delta_voe_spike_jepa_minus_pixel[voeType];

  const types = useMemo(
    () => (data ? violationTypes(data.voe) : []),
    [data]
  );

  if (loading) {
    return <p className="status page-status">Loading demo fixtures…</p>;
  }
  if (error || !data) {
    return <p className="error page-status">{error ?? "Failed to load fixtures"}</p>;
  }

  return (
    <div className="page try-page">
      <header className="page-header">
        <h1>Interactive playground</h1>
        <p className="page-lead">
          Scrub matched possible vs impossible rollouts. JEPA surprise updates from exported{" "}
          <code>paper_mid</code> eval curves — same metrics as offline VoE, replayed in the browser.
          Also available at <Link to="/play">/play</Link>.
        </p>
      </header>

      <div className="try-layout">
        <div className="try-replay">
          {data.voeDemo?.pairs?.length ? (
            <VoEPairReplay
              index={data.voeDemo.pairs}
              selectedType={types.includes(voeType) ? voeType : types[0]}
              onSelectType={setVoeType}
              onTimeChange={setCurrentT}
              compact
            />
          ) : (
            <section className="panel">
              <p className="hint">
                No VoE demo pairs. Run export + sync scripts locally.
              </p>
            </section>
          )}
        </div>

        {voeSummary && (
          <div className="try-inference">
            <LiveInferencePanel
              summary={voeSummary}
              violationType={voeType}
              currentT={currentT}
              tStar={tStar}
              deltaVsPixel={deltaVsPixel}
            />
          </div>
        )}
      </div>

      <section className="panel try-tips">
        <h2>How to read this</h2>
        <ul className="doc-list">
          <li>
            <strong>t*</strong> marks when the impossible branch diverges (e.g. wrong bounce
            restitution).
          </li>
          <li>
            A physics-like model should show higher surprise on the impossible branch near{" "}
            <strong>t*</strong> — strongest for impossible bounce in our mid-scale run.
          </li>
          <li>
            Use ← → arrow keys or the slider. Compare aggregate metrics on{" "}
            <Link to="/results">Results</Link>.
          </li>
        </ul>
      </section>
    </div>
  );
}

/** Alias route — same component */
export const PlayPage = TryPage;
