import { useCallback, useEffect, useMemo, useState } from "react";
import type { EpisodeMeta, VoEDemoEntry } from "../types/schema";
import { fetchJson, formatViolationLabel, paths } from "../lib/loadRun";

interface VoEPairReplayProps {
  index: VoEDemoEntry[];
  selectedType: string;
  onSelectType: (t: string) => void;
  onTimeChange?: (t: number) => void;
  compact?: boolean;
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
}: VoEPairReplayProps) {
  const entry = useMemo(
    () => index.find((p) => p.violation_type === selectedType) ?? index[0],
    [index, selectedType]
  );

  const [possible, setPossible] = useState<EpisodeMeta | null>(null);
  const [impossible, setImpossible] = useState<EpisodeMeta | null>(null);
  const [t, setT] = useState(0);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  useEffect(() => {
    if (!entry) return;
    let cancelled = false;
    setLoadErr(null);
    (async () => {
      try {
        const [poss, imposs] = await Promise.all([
          fetchJson<EpisodeMeta>(paths.voeDemoMeta(entry.base, "possible")),
          fetchJson<EpisodeMeta>(paths.voeDemoMeta(entry.base, "impossible")),
        ]);
        if (cancelled) return;
        setPossible(poss);
        setImpossible(imposs);
        const initT = Math.min(entry.t_star, poss.T - 1);
        setT(initT);
        onTimeChange?.(initT);
      } catch (e) {
        if (!cancelled) {
          setLoadErr(e instanceof Error ? e.message : String(e));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [entry]);

  const T = possible?.T ?? impossible?.T ?? 1;
  const tStar = entry?.t_star ?? 0;

  const clamp = useCallback((v: number) => Math.max(0, Math.min(T - 1, v)), [T]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") {
        setT((x) => {
          const next = clamp(x - 1);
          onTimeChange?.(next);
          return next;
        });
      }
      if (e.key === "ArrowRight") {
        setT((x) => {
          const next = clamp(x + 1);
          onTimeChange?.(next);
          return next;
        });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clamp]);

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
          onChange={(e) => {
            const next = Number(e.target.value);
            setT(next);
            onTimeChange?.(next);
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
