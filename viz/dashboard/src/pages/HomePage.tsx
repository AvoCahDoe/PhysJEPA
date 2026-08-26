import { Link } from "react-router-dom";

const GITHUB = "https://github.com/AvoCahDoe/PhysJEPA";

export function HomePage() {
  return (
    <div className="page home-page">
      <section className="hero">
        <div className="hero-glow" aria-hidden />
        <p className="eyebrow">Controlled 2D physics · JEPA vs pixel</p>
        <h1>Does latent prediction encode naive physics?</h1>
        <p className="hero-lead">
          PhysJEPA is a falsifiable diagnostic: train a JEPA world model and a matched pixel
          baseline on procedural pymunk rollouts, then probe whether latent space linearly encodes
          physical variables and spikes on violation-of-expectation events.
        </p>
        <div className="hero-actions">
          <Link to="/try" className="btn-primary">
            Try the demo
          </Link>
          <Link to="/results" className="btn-secondary">
            View results
          </Link>
          <Link to="/docs" className="btn-secondary">
            Read concepts
          </Link>
        </div>
      </section>

      <section className="feature-grid">
        <article className="feature-card">
          <span className="feature-icon">📐</span>
          <h2>Docs</h2>
          <p>
            Environment design, JEPA architecture, linear probes, VoE surprise metrics — with
            LaTeX formulas and links to the full report.
          </p>
          <Link to="/docs" className="feature-link">
            Explore docs →
          </Link>
        </article>
        <article className="feature-card accent">
          <span className="feature-icon">▶</span>
          <h2>Try</h2>
          <p>
            Scrub matched possible vs impossible episodes. Watch JEPA surprise update live from
            exported inference curves at each timestep.
          </p>
          <Link to="/try" className="feature-link">
            Open playground →
          </Link>
        </article>
        <article className="feature-card">
          <span className="feature-icon">📊</span>
          <h2>Results</h2>
          <p>
            Mid-scale <code>paper_mid</code> run: probe tables, VoE charts, ablations, training
            curves — with plain-language interpretation.
          </p>
          <Link to="/results" className="feature-link">
            See metrics →
          </Link>
        </article>
      </section>

      <section className="headline panel">
        <h2>Headline finding</h2>
        <div className="headline-metrics">
          <div>
            <span className="metric-label">JEPA xy val R²</span>
            <span className="metric-value">0.421</span>
          </div>
          <div>
            <span className="metric-label">VoE Δ impossible bounce</span>
            <span className="metric-value accent-green">+0.028</span>
            <span className="metric-note">JEPA &gt; pixel</span>
          </div>
          <div>
            <span className="metric-label">Train budget</span>
            <span className="metric-value">1k ep · 80 ep</span>
          </div>
        </div>
        <p className="hint">
          Nuanced result: selective VoE spike on impossible bounce, weak signal elsewhere.{" "}
          <a href={GITHUB} target="_blank" rel="noopener noreferrer">
            Full code & report on GitHub
          </a>
        </p>
      </section>
    </div>
  );
}
