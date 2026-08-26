import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AblationsDoc } from "../types/schema";

interface AblationsPanelProps {
  data: AblationsDoc;
}

export function AblationsPanel({ data }: AblationsPanelProps) {
  const occ = data.occlusion_duration.results.map((r) => ({
    label: r.label,
    occlusion_frames: r.mean_occlusion_frames ?? 0,
    spike: r.spike_score ?? 0,
  }));
  const arch = data.architecture.results.map((r) => ({
    predictor: r.predictor_type,
    mean_spike: r.mean_voe_spike,
    xy_r2: r.probes?.xy?.val_r2 ?? null,
    visible_acc: r.probes?.visible?.val_acc ?? null,
  }));

  return (
    <div className="stack">
      <section className="panel">
        <h2>Occlusion duration</h2>
        <p className="hint">
          {data.occlusion_duration.description ??
            "VoE teleport spike vs how long the object stays occluded."}
        </p>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={occ}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="label" tick={{ fill: "#aaa" }} />
              <YAxis yAxisId="left" tick={{ fill: "#aaa" }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fill: "#aaa" }} />
              <Tooltip contentStyle={{ background: "#1a1f28", border: "1px solid #444" }} />
              <Legend />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="occlusion_frames"
                name="mean occluded frames"
                stroke="#e8a838"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="spike"
                name="VoE spike"
                stroke="#5b9bd5"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <h2>Predictor architecture</h2>
        <p className="hint">
          {data.architecture.description ??
            "Mean VoE spike across violation types by predictor type."}
        </p>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={arch}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="predictor" tick={{ fill: "#aaa" }} />
              <YAxis tick={{ fill: "#aaa" }} />
              <Tooltip contentStyle={{ background: "#1a1f28", border: "1px solid #444" }} />
              <Bar dataKey="mean_spike" name="mean VoE spike" fill="#5b9bd5" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <table className="probe-table">
          <thead>
            <tr>
              <th>Predictor</th>
              <th>Mean VoE spike</th>
              <th>xy val R²</th>
              <th>visible val acc</th>
            </tr>
          </thead>
          <tbody>
            {arch.map((r) => (
              <tr key={r.predictor}>
                <td>{r.predictor}</td>
                <td>{r.mean_spike.toFixed(5)}</td>
                <td>{r.xy_r2 != null ? r.xy_r2.toFixed(3) : "—"}</td>
                <td>{r.visible_acc != null ? r.visible_acc.toFixed(3) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
