import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { VoETypeSummary } from "../types/schema";
import { formatViolationLabel } from "../lib/loadRun";

interface VoESurpriseChartProps {
  summary: VoETypeSummary;
  violationType: string;
}

export function VoESurpriseChart({ summary, violationType }: VoESurpriseChartProps) {
  const chartData = useMemo(() => {
    return summary.t.map((t, i) => ({
      t,
      possible: summary.possible_mean[i],
      impossible: summary.impossible_mean[i],
    }));
  }, [summary]);

  return (
    <section className="panel">
      <h2>{formatViolationLabel(violationType)}</h2>
      <p className="hint">
        t* = {summary.t_star} · spike score = {summary.spike_score.toFixed(5)} · pre-* gap ={" "}
        {(summary.pre_tstar_abs_gap ?? 0).toFixed(6)}
      </p>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="t"
              label={{ value: "timestep t", position: "insideBottom", offset: -4, fill: "#888" }}
              tick={{ fill: "#aaa" }}
            />
            <YAxis
              label={{ value: "surprise", angle: -90, position: "insideLeft", fill: "#888" }}
              tick={{ fill: "#aaa" }}
            />
            <Tooltip
              contentStyle={{ background: "#1a1f28", border: "1px solid #444" }}
              formatter={(v: number) => (v != null ? v.toFixed(5) : "—")}
            />
            <Legend />
            <ReferenceLine
              x={summary.t_star}
              stroke="#e8a838"
              strokeDasharray="4 4"
              label={{ value: "t*", fill: "#e8a838", position: "top" }}
            />
            <Line
              type="monotone"
              dataKey="possible"
              name="possible"
              stroke="#5b9bd5"
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="impossible"
              name="impossible"
              stroke="#c75c5c"
              dot={false}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

interface VoESurprisePanelProps {
  byType: Record<string, VoETypeSummary>;
  selected: string;
  onSelect: (key: string) => void;
}

export function VoESurprisePanel({ byType, selected, onSelect }: VoESurprisePanelProps) {
  const types = Object.keys(byType);
  const summary = byType[selected];

  return (
    <div>
      <div className="selector-row">
        <label htmlFor="voe-type">Violation type</label>
        <select
          id="voe-type"
          value={selected}
          onChange={(e) => onSelect(e.target.value)}
        >
          {types.map((k) => (
            <option key={k} value={k}>
              {formatViolationLabel(k)}
            </option>
          ))}
        </select>
      </div>
      {summary && <VoESurpriseChart summary={summary} violationType={selected} />}
    </div>
  );
}
