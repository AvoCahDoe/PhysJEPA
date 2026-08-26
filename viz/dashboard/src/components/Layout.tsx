import type { ReactNode } from "react";
import type { TabId } from "../types/schema";

const TABS: { id: TabId; label: string }[] = [
  { id: "compare", label: "Comparison" },
  { id: "voe", label: "VoE Surprise" },
  { id: "probes", label: "Linear Probes" },
  { id: "ablations", label: "Ablations" },
  { id: "train", label: "Training" },
  { id: "episode", label: "Episode Replay" },
];

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
        <h1>physjepa results</h1>
        <p className="subtitle">JEPA vs pixel baseline — probes &amp; VoE diagnostics</p>
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
