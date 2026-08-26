import { useCallback, useEffect, useMemo, useState } from "react";
import type { EpisodeMeta, VoEDemoEntry } from "../types/schema";
import { fetchJson, formatViolationLabel, paths } from "../lib/loadRun";

interface VoEPairReplayProps {
  index: VoEDemoEntry[];
  selectedType: string;
  onSelectType: (t: string) => void;
  onTimeChange?: (t: number) => void;
  compact?: boolean;
  /** Start from t=0 and auto-play when the pair loads (e.g. /play route). */
  autoPlay?: boolean;
}

function PairPanel({
  label,
  meta,
  t,
  tStar,
  frameBase,
  branch,
}: {
  label: string;
  meta: EpisodeMeta;
  t: number;
  tStar: number;
  frameBase: string;
  branch: "possible" | "impossible";
}) {
  const body = meta.trajectory[t]?.bodies?.[0];
  const atStar = t === tStar;

  return (
    <div className={`pair-panel ${atStar ? "at-star" : ""}`}>
      <h3>{label}</h3>
      <div className="frame-box">
        <img
          src={paths.voeDemoFrame(frameBase, branch, t)}
          alt={`${label} frame ${t}`}
          width={meta.resolution[0] * 4}
          height={meta.resolution[1] * 4}
          className="frame-img"
        />
      </div>
      {body && (
        <dl className="body-stats compact">
          <dt>pos</dt>
          <dd>
            ({body.x.toFixed(2)}, {body.y.toFixed(2)})
          </dd>
          <dt>vis</dt>
          <dd>{body.visible ? "yes" : "occluded"}</dd>
        </dl>
      )}
    </div>
  );
}

export function VoEPairReplay({
  index,
  selectedType,
  onSelectType,
  onTimeChange,
  compact = false,
  autoPlay = false,
}: VoEPairReplayProps) {
  const entry = useMemo(
    () => index.find((p) => p.violation_type === selectedType) ?? index[0],
    [index, selectedType]
  );

  const [possible, setPossible] = useState<EpisodeMeta | null>(null);
  const [impossible, setImpossible] = useState<EpisodeMeta | null>(null);
  const [t, setT] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const T = possible?.T ?? impossible?.T ?? 1;
  const tStar = entry?.t_star ?? 0;
  const fps = possible?.fps ?? 12;

  const setTAndNotify = useCallback(
    (next: number) => {
      setT(next);
      onTimeChange?.(next);
    },
    [onTimeChange]
  );

  const clamp = useCallback((v: number) => Math.max(0, Math.min(T - 1, v)), [T]);

  useEffect(() => {
    if (!entry) return;
    let cancelled = false;
    setLoadErr(null);
    setIsPlaying(false);
    (async () => {
      try {
        const [poss, imposs] = await Promise.all([
          fetchJson<EpisodeMeta>(paths.voeDemoMeta(entry.base, "possible")),
          fetchJson<EpisodeMeta>(paths.voeDemoMeta(entry.base, "impossible")),
        ]);
        if (cancelled) return;
        setPossible(poss);
        setImpossible(imposs);

        const initT = autoPlay ? 0 : Math.min(entry.t_star, poss.T - 1);
        setTAndNotify(initT);
        if (autoPlay) setIsPlaying(true);

        for (let i = 0; i < poss.T; i++) {
          for (const branch of ["possible", "impossible"] as const) {
            const img = new Image();
            img.src = paths.voeDemoFrame(entry.base, branch, i);
          }
        }
      } catch (e) {
        if (!cancelled) {
          setLoadErr(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [entry, autoPlay, setTAndNotify]);

  useEffect(() => {
    if (!isPlaying || T <= 1) return;
    const intervalMs = 1000 / fps;
    const id = window.setInterval(() => {
      setT((prev) => {
        if (prev >= T - 1) {
          setIsPlaying(false);
          return prev;
        }
        const next = prev + 1;
        onTimeChange?.(next);
        return next;
      });
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [isPlaying, T, fps, onTimeChange]);

  const handlePlayPause = useCallback(() => {
    setIsPlaying((playing) => {
      if (playing) return false;
      setT((prev) => {
        if (prev >= T - 1) {
          onTimeChange?.(0);
          return 0;
        }
        return prev;
      });
      return true;
    });
  }, [T, onTimeChange]);

  const handleRestart = useCallback(() => {
    setIsPlaying(true);
    setTAndNotify(0);
  }, [setTAndNotify]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === " ") {
        e.preventDefault();
        handlePlayPause();
        return;
      }
      if (isPlaying) return;
      if (e.key === "ArrowLeft") {
        setTAndNotify(clamp(t - 1));
      }
      if (e.key === "ArrowRight") {
        setTAndNotify(clamp(t + 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clamp, isPlaying, t, setTAndNotify, handlePlayPause]);

  if (!entry) {
    return <p className="hint">No VoE demo pairs loaded.</p>;
  }

  if (loadErr) {
    return <p className="error">{loadErr}</p>;
  }

  if (!possible || !impossible) {
    return <p className="status">Loading pair…</p>;
  }

  return (
    <section className={`panel voe-replay-panel ${compact ? "compact" : ""}`}>
      {!compact && (
        <>
          <h2>VoE pair replay</h2>
          <p className="hint">
            Matched possible vs impossible rollouts · shared timeline ·{" "}
            <strong>t*={tStar}</strong> is the violation · ← → keys
          </p>
        </>
      )}

      <div className="selector-row">
        <label htmlFor="voe-replay-type">Violation</label>
        <select
          id="voe-replay-type"
          value={selectedType}
          onChange={(e) => onSelectType(e.target.value)}
        >
          {index.map((p) => (
            <option key={p.pair_id} value={p.violation_type}>
              {formatViolationLabel(p.violation_type)}
            </option>
          ))}
        </select>
        {entry.description && (
          <span className="pair-desc">{entry.description}</span>
        )}
      </div>

      <div className="pair-grid">
        <PairPanel
          label="Possible"
          meta={possible}
          t={t}
          tStar={tStar}
          frameBase={entry.base}
          branch="possible"
        />
        <PairPanel
          label="Impossible"
          meta={impossible}
          t={t}
          tStar={tStar}
          frameBase={entry.base}
          branch="impossible"
        />
      </div>

      <div className="replay-controls">
        <button
          type="button"
          className="btn-play"
          onClick={handlePlayPause}
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button type="button" className="btn-replay" onClick={handleRestart}>
          Restart
        </button>
        <span className="replay-progress">
          Frame {t + 1} / {T}
          {isPlaying && <span className="chip chip-live">playing</span>}
        </span>
      </div>

      <div className="timeline-wrap">
        <label htmlFor="pair-slider">
          t = {t}
          {t === tStar && <span className="star-badge"> t* violation</span>}
        </label>
        <input
          id="pair-slider"
          type="range"
          min={0}
          max={T - 1}
          value={t}
          disabled={isPlaying}
          onChange={(e) => {
            setIsPlaying(false);
            setTAndNotify(Number(e.target.value));
          }}
        />
        <div className="timeline-markers">
          <span>0</span>
          <span
            className="t-star-marker"
            style={{ left: `${(tStar / Math.max(1, T - 1)) * 100}%` }}
          >
            t*
          </span>
          <span>{T - 1}</span>
        </div>
      </div>
    </section>
  );
}
