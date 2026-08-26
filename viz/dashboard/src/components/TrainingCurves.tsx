import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TrainingSummary } from "../types/schema";

interface TrainingCurvesProps {
  jepa: TrainingSummary;
  pixel: TrainingSummary;
}

function mergeCurves(jepa: TrainingSummary, pixel: TrainingSummary) {
  const maxLen = Math.max(jepa.curves.step.length, pixel.curves.step.length);
  const rows = [];
  for (let i = 0; i < maxLen; i++) {
    rows.push({
      step: jepa.curves.step[i] ?? pixel.curves.step[i] ?? i,
      jepa_loss: jepa.curves.loss[i],
      pixel_loss: pixel.curves.loss[i],
      jepa_std: jepa.curves.latent_std[i],
      pixel_std: pixel.curves.latent_std[i],
    });
  }
  return rows;
}

export function TrainingCurves({ jepa, pixel }: TrainingCurvesProps) {
  const data = mergeCurves(jepa, pixel);

  return (
    <section className="panel">
      <h2>Training curves</h2>
      <p className="hint">
        JEPA run: {jepa.run_id} (best val {jepa.best_val_loss?.toFixed(4) ?? "—"}) · Pixel run:{" "}
        {pixel.run_id} (best val {pixel.best_val_loss?.toFixed(4) ?? "—"})
      </p>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="step" tick={{ fill: "#aaa" }} />
            <YAxis tick={{ fill: "#aaa" }} />
            <Tooltip contentStyle={{ background: "#1a1f28", border: "1px solid #444" }} />
            <Legend />
            <Line type="monotone" dataKey="jepa_loss" name="JEPA loss" stroke="#5b9bd5" dot={false} />
            <Line type="monotone" dataKey="pixel_loss" name="Pixel loss" stroke="#c75c5c" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="step" tick={{ fill: "#aaa" }} />
            <YAxis tick={{ fill: "#aaa" }} />
            <Tooltip contentStyle={{ background: "#1a1f28", border: "1px solid #444" }} />
            <Legend />
            <Line type="monotone" dataKey="jepa_std" name="JEPA latent std" stroke="#7cb87c" dot={false} />
            <Line type="monotone" dataKey="pixel_std" name="Pixel latent std" stroke="#b89cb8" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
