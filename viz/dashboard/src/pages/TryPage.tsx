import { Link } from "react-router-dom";
import { VoEViolationShowcase } from "../components/VoEViolationShowcase";
import { useFixtures } from "../hooks/useFixtures";

interface TryPageProps {
  autoPlayFirst?: boolean;
}

export function TryPage({ autoPlayFirst = false }: TryPageProps) {
  const { data, loading, error } = useFixtures();

  if (loading) {
    return <p className="status page-status">Loading demo fixtures…</p>;
  }
  if (error || !data) {
    return <p className="error page-status">{error ?? "Failed to load fixtures"}</p>;
  }

  return (
    <div className="page try-page">
      <header className="page-header">
        <h1>All violation scenarios</h1>
        <p className="page-lead">
          Every VoE probe type side by side — play each rollout, scrub timelines, and watch JEPA
          surprise respond. No picking from a menu: scroll through all four. Also at{" "}
          <Link to="/play">/play</Link> (first scenario auto-plays).
        </p>
      </header>

      {data.voeDemo?.pairs?.length ? (
        <VoEViolationShowcase
          pairs={data.voeDemo.pairs}
          voe={data.voe}
          comparison={data.comparison}
          autoPlayFirst={autoPlayFirst}
        />
      ) : (
        <section className="panel">
          <p className="hint">No VoE demo pairs. Run export + sync scripts locally.</p>
        </section>
      )}

      <section className="panel try-tips">
        <h2>How to read this</h2>
        <ul className="doc-list">
          <li>
            <strong>t*</strong> marks when the impossible branch diverges (highlighted border at
            that frame).
          </li>
          <li>
            <strong>Impossible bounce</strong> shows the clearest JEPA surprise gap (Δ ≈ +0.028 vs
            pixel). Other violations are weaker or inverted.
          </li>
          <li>
            Use the jump links at the top, or scroll. Each section has its own Play control; set
            global speed to <strong>1 · 2 · 4 fps</strong>.
          </li>
          <li>
            Aggregate charts and probes on <Link to="/results">Results</Link>.
          </li>
        </ul>
      </section>
    </div>
  );
}

/** /play — auto-starts the first violation on load */
export function PlayPage() {
  return <TryPage autoPlayFirst />;
}
