import type { ComparisonDoc } from "../types/schema";
import { formatViolationLabel } from "../lib/loadRun";

interface ProbeComparisonProps {
  data: ComparisonDoc;
}

const TARGETS = ["xy", "vxvy", "mass", "visible"] as const;

function fmtProbe(target: string, side: ComparisonDoc["jepa"]["probes"][string]) {
  if (target === "visible") {
    return side.val_acc != null ? `acc ${side.val_acc.toFixed(3)}` : "—";
  }
  if (side.val_r2 != null) {
    return `R² ${side.val_r2.toFixed(3)} (mse ${side.val_mse?.toFixed(4) ?? "?"})`;
  }
  return "—";
}

export function ProbeComparison({ data }: ProbeComparisonProps) {
  return (
    <section className="panel">
      <h2>Linear probes (validation)</h2>
      <p className="hint">Frozen encoder; body id 0. Higher R² / acc = more physics in latents.</p>
      <table className="probe-table">
        <thead>
          <tr>
            <th>Target</th>
            <th>JEPA</th>
            <th>Pixel</th>
          </tr>
        </thead>
        <tbody>
          {TARGETS.map((t) => (
            <tr key={t}>
              <td>{formatViolationLabel(t)}</td>
              <td>{fmtProbe(t, data.jepa.probes[t] ?? {})}</td>
              <td>{fmtProbe(t, data.pixel.probes[t] ?? {})}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
