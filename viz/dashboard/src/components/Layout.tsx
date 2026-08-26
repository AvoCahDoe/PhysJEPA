import type { ReactNode } from "react";
import type { TabId } from "../types/schema";

const TABS: { id: TabId; label: string }[] = [
  { id: "compare", label: "Comparison" },
  { id: "voe", label: "VoE Surprise" },
  { id: "voe_replay", label: "VoE Replay" },
  { id: "probes", label: "Linear Probes" },
  { id: "ablations", label: "Ablations" },
  { id: "train", label: "Training" },
  { id: "episode", label: "Episode Replay" },
];

const GITHUB = "https://github.com/AvoCahDoe/PhysJEPA";
const REPORT = "https://github.com/AvoCahDoe/PhysJEPA/blob/main/docs/report.md";

interface LayoutProps {
  active: TabId;
  onTab: (id: TabId) => void;
  children: ReactNode;
  error?: string | null;
  loading?: boolean;
}

export function Layout({ active, onTab, children, error, loading }: LayoutProps) {
  return (
    <div className="app">
      <header className="header">
        <div className="header-top">
          <div>
            <h1>PhysJEPA</h1>
            <p className="subtitle">
              JEPA vs pixel baseline — probes &amp; VoE diagnostics
              <span className="badge">paper_mid · 1k eps · 80 epochs</span>
            </p>
          </div>
          <nav className="header-links">
            <a href={GITHUB} target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
            <a href={REPORT} target="_blank" rel="noopener noreferrer">
              Report
            </a>
          </nav>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={active === t.id ? "tab active" : "tab"}
              onClick={() => onTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="main">
        {loading && <p className="status">Loading fixtures…</p>}
        {error && <p className="error">{error}</p>}
        {!loading && !error && children}
      </main>
    </div>
  );
}
