import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ComparisonDoc } from "../types/schema";
import { formatViolationLabel } from "../lib/loadRun";

interface ComparisonOverviewProps {
  data: ComparisonDoc;
}

export function ComparisonOverview({ data }: ComparisonOverviewProps) {
  const bars = Object.entries(data.delta_voe_spike_jepa_minus_pixel).map(
    ([type, delta]) => ({
      type,
      label: formatViolationLabel(type),
      delta,
    })
  );

  return (
    <section className="panel">
      <h2>JEPA vs pixel — VoE spike delta</h2>
      <p className="hint">
        Positive bar: JEPA shows a larger surprise spike at violation than pixel baseline.
      </p>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={bars} margin={{ top: 8, right: 16, left: 8, bottom: 64 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="label"
              angle={-25}
              textAnchor="end"
              interval={0}
              height={70}
              tick={{ fill: "#aaa", fontSize: 11 }}
            />
            <YAxis tick={{ fill: "#aaa" }} />
            <Tooltip
              contentStyle={{ background: "#1a1f28", border: "1px solid #444" }}
            />
            <Bar dataKey="delta" name="Δ spike (JEPA − pixel)">
              {bars.map((entry) => (
                <Cell
                  key={entry.type}
                  fill={entry.delta >= 0 ? "#5b9bd5" : "#c75c5c"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="cards">
        <div className="card">
          <h3>JEPA</h3>
          <p>run: {data.jepa.run_id}</p>
          <p>xy val R²: {data.jepa.probes.xy?.val_r2?.toFixed(3) ?? "—"}</p>
          <p>visible val acc: {data.jepa.probes.visible?.val_acc?.toFixed(3) ?? "—"}</p>
        </div>
        <div className="card">
          <h3>Pixel</h3>
          <p>run: {data.pixel.run_id}</p>
          <p>xy val R²: {data.pixel.probes.xy?.val_r2?.toFixed(3) ?? "—"}</p>
          <p>visible val acc: {data.pixel.probes.visible?.val_acc?.toFixed(3) ?? "—"}</p>
        </div>
      </div>
    </section>
  );
}
