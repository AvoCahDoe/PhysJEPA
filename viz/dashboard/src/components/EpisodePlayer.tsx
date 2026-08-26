import { useCallback, useEffect, useState } from "react";
import type { EpisodeMeta } from "../types/schema";
import { paths } from "../lib/loadRun";

interface EpisodePlayerProps {
  meta: EpisodeMeta;
}

export function EpisodePlayer({ meta }: EpisodePlayerProps) {
  const [t, setT] = useState(0);
  const T = meta.T;

  const clamp = useCallback((v: number) => Math.max(0, Math.min(T - 1, v)), [T]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") setT((x) => clamp(x - 1));
      if (e.key === "ArrowRight") setT((x) => clamp(x + 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clamp]);

  const frame = meta.trajectory[t];
  const body = frame?.bodies?.[0];

  return (
    <section className="panel episode-panel">
      <h2>Episode replay</h2>
      <p className="hint">
        seed {meta.seed} · {meta.split} · {T} frames · use ← → keys
      </p>
      <div className="episode-layout">
        <div className="frame-box">
          <img
            src={paths.episodeFrame(t)}
            alt={`frame ${t}`}
            width={meta.resolution[0] * 4}
            height={meta.resolution[1] * 4}
            className="frame-img"
          />
        </div>
        <div className="episode-meta">
          <label htmlFor="frame-slider">t = {t}</label>
          <input
            id="frame-slider"
            type="range"
            min={0}
            max={T - 1}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
          />
          {body && (
            <dl className="body-stats">
              <dt>position</dt>
              <dd>
                ({body.x.toFixed(3)}, {body.y.toFixed(3)})
              </dd>
              <dt>velocity</dt>
              <dd>
                ({body.vx.toFixed(3)}, {body.vy.toFixed(3)})
              </dd>
              <dt>visible</dt>
              <dd>{body.visible ? "yes" : "no (occluded)"}</dd>
              <dt>mass</dt>
              <dd>{body.mass.toFixed(3)}</dd>
            </dl>
          )}
          {meta.violation && (
            <p className="violation-note">
              VoE violation: {meta.violation.type} at t*={meta.violation.t_star}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
