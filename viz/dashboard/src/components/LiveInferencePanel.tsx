import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { VoETypeSummary } from "../types/schema";
import { formatViolationLabel } from "../lib/loadRun";

interface LiveInferencePanelProps {
  summary: VoETypeSummary;
  violationType: string;
  currentT: number;
  tStar: number;
  deltaVsPixel?: number;
}

export function LiveInferencePanel({
  summary,
  violationType,
  currentT,
  tStar,
  deltaVsPixel,
}: LiveInferencePanelProps) {
  const possibleSurprise = summary.possible_mean[currentT] ?? 0;
  const impossibleSurprise = summary.impossible_mean[currentT] ?? 0;
  const gap = impossibleSurprise - possibleSurprise;
  const atViolation = currentT === tStar;

  const chartData = useMemo(
    () =>
      summary.t.map((t, i) => ({
        t,
        possible: summary.possible_mean[i],
        impossible: summary.impossible_mean[i],
        isCurrent: t === currentT,
      })),
    [summary, currentT]
  );

  const barData = [
    { name: "Possible", value: possibleSurprise, fill: "#5b9bd5" },
    { name: "Impossible", value: impossibleSurprise, fill: "#c75c5c" },
  ];

  const verdict =
    gap > 0.01
      ? "Elevated surprise on impossible branch"
      : gap < -0.01
        ? "Lower surprise on impossible (unexpected)"
        : "Similar surprise on both branches";

  return (
    <section className="panel inference-panel">
      <div className="inference-header">
        <h2>Live JEPA inference</h2>
        <span className="chip chip-live">precomputed · paper_mid</span>
      </div>
      <p className="hint">
        Surprise = smooth L1 between predicted and target latents at each timestep.
        Scrub the replay — values update from the exported eval curve (same as offline VoE).
      </p>

      <div className="inference-stats">
        <div className={`stat-card ${atViolation ? "highlight" : ""}`}>
          <span className="stat-label">timestep t</span>
          <span className="stat-value">{currentT}</span>
          {atViolation && <span className="stat-badge">t* violation</span>}
        </div>
        <div className="stat-card">
          <span className="stat-label">possible surprise</span>
          <span className="stat-value accent-blue">{possibleSurprise.toFixed(5)}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">impossible surprise</span>
          <span className="stat-value accent-red">{impossibleSurprise.toFixed(5)}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Δ (imp − poss)</span>
          <span className={`stat-value ${gap >= 0 ? "accent-green" : "accent-red"}`}>
            {gap >= 0 ? "+" : ""}
            {gap.toFixed(5)}
          </span>
        </div>
      </div>

      <p className="inference-verdict">{verdict}</p>

      <div className="chart-wrap chart-compact">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={barData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="name" tick={{ fill: "#aaa" }} />
            <YAxis tick={{ fill: "#aaa" }} />
            <Tooltip
              contentStyle={{ background: "#1a1f28", border: "1px solid #444" }}
              formatter={(v: number) => v.toFixed(5)}
            />
            <Bar dataKey="value" name="surprise" radius={[4, 4, 0, 0]}>
              {barData.map((entry) => (
                <Cell key={entry.name} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-wrap">
        <h3 className="chart-subtitle">{formatViolationLabel(violationType)} — full curve</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="t" tick={{ fill: "#888", fontSize: 10 }} />
            <YAxis tick={{ fill: "#888", fontSize: 10 }} />
            <Tooltip
              contentStyle={{ background: "#1a1f28", border: "1px solid #444" }}
              formatter={(v: number) => v.toFixed(5)}
            />
            <ReferenceLine x={tStar} stroke="#e8a838" strokeDasharray="4 4" />
            <ReferenceLine x={currentT} stroke="#7eb8ff" strokeWidth={2} />
            <Bar dataKey="possible" fill="#5b9bd588" name="possible" />
            <Bar dataKey="impossible" fill="#c75c5c88" name="impossible" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {deltaVsPixel != null && (
        <p className="hint meta-line">
          VoE spike Δ vs pixel baseline (JEPA − pixel):{" "}
          <strong className={deltaVsPixel >= 0 ? "accent-green" : "accent-red"}>
            {deltaVsPixel >= 0 ? "+" : ""}
            {deltaVsPixel.toFixed(4)}
          </strong>
          {" · "}aggregate spike score: {summary.spike_score.toFixed(5)}
        </p>
      )}
    </section>
  );
}
