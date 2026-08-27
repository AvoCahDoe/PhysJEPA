import { useMemo, useState } from "react";
import type { ComparisonDoc, VoEDemoEntry, VoESurpriseDoc } from "../types/schema";
import { formatViolationLabel } from "../lib/loadRun";
import { LiveInferencePanel } from "./LiveInferencePanel";
import { PLAYBACK_SPEEDS, VoEPairReplay, type PlaybackSpeed } from "./VoEPairReplay";

export const VIOLATION_ORDER = [
  "impossible_bounce",
  "teleport_occlusion",
  "pass_through_wall",
  "stop_without_collision",
] as const;

function violationSlug(type: string) {
  return type.replace(/_/g, "-");
}

function sortPairs(pairs: VoEDemoEntry[]) {
  return [...pairs].sort((a, b) => {
    const ai = VIOLATION_ORDER.indexOf(a.violation_type as (typeof VIOLATION_ORDER)[number]);
    const bi = VIOLATION_ORDER.indexOf(b.violation_type as (typeof VIOLATION_ORDER)[number]);
    const ar = ai === -1 ? 999 : ai;
    const br = bi === -1 ? 999 : bi;
    return ar - br;
  });
}

interface VoEViolationSectionProps {
  entry: VoEDemoEntry;
  voeSummary: VoESurpriseDoc["by_type"][string] | undefined;
  deltaVsPixel?: number;
  autoPlay?: boolean;
  playbackFps: PlaybackSpeed;
}

function VoEViolationSection({
  entry,
  voeSummary,
  deltaVsPixel,
  autoPlay = false,
  playbackFps,
}: VoEViolationSectionProps) {
  const [currentT, setCurrentT] = useState(0);
  const tStar = entry.t_star ?? voeSummary?.t_star ?? 0;
  const highlight = (deltaVsPixel ?? 0) > 0.01;

  return (
    <section
      id={violationSlug(entry.violation_type)}
      className={`violation-section panel ${highlight ? "violation-highlight" : ""}`}
    >
      <header className="violation-header">
        <div>
          <h2>{formatViolationLabel(entry.violation_type)}</h2>
          {entry.description && <p className="violation-desc">{entry.description}</p>}
        </div>
        {deltaVsPixel != null && (
          <span
            className={`delta-badge ${deltaVsPixel >= 0 ? "positive" : "negative"}`}
            title="VoE spike Δ (JEPA − pixel)"
          >
            Δ {deltaVsPixel >= 0 ? "+" : ""}
            {deltaVsPixel.toFixed(4)}
          </span>
        )}
      </header>

      <div className="violation-layout">
        <VoEPairReplay
          entry={entry}
          onTimeChange={setCurrentT}
          autoPlay={autoPlay}
          playbackFps={playbackFps}
          showSpeedControl={false}
          enableKeyboard={false}
          compact
        />
        {voeSummary && (
          <LiveInferencePanel
            summary={voeSummary}
            violationType={entry.violation_type}
            currentT={currentT}
            tStar={tStar}
            deltaVsPixel={deltaVsPixel}
          />
        )}
      </div>
    </section>
  );
}

interface VoEViolationShowcaseProps {
  pairs: VoEDemoEntry[];
  voe: VoESurpriseDoc;
  comparison: ComparisonDoc;
  autoPlayFirst?: boolean;
}

export function VoEViolationShowcase({
  pairs,
  voe,
  comparison,
  autoPlayFirst = false,
}: VoEViolationShowcaseProps) {
  const sorted = useMemo(() => sortPairs(pairs), [pairs]);
  const [playbackFps, setPlaybackFps] = useState<PlaybackSpeed>(2);

  if (!sorted.length) {
    return <p className="hint">No VoE demo pairs loaded.</p>;
  }

  return (
    <div className="violation-showcase">
      <div className="showcase-toolbar panel">
        <p className="showcase-toolbar-lead">
          Four held-out violation types — scroll to compare possible vs impossible rollouts and
          JEPA surprise at each timestep.
        </p>
        <nav className="violation-jump-nav" aria-label="Jump to violation">
          {sorted.map((p) => (
            <a key={p.pair_id} href={`#${violationSlug(p.violation_type)}`}>
              {formatViolationLabel(p.violation_type)}
            </a>
          ))}
        </nav>
        <div className="speed-control showcase-speed" role="group" aria-label="Playback speed">
          <span className="speed-label">Speed (all players)</span>
          {PLAYBACK_SPEEDS.map((speed) => (
            <button
              key={speed}
              type="button"
              className={`speed-btn ${playbackFps === speed ? "active" : ""}`}
              onClick={() => setPlaybackFps(speed)}
              aria-pressed={playbackFps === speed}
            >
              {speed} fps
            </button>
          ))}
        </div>
      </div>

      <div className="violation-stack">
        {sorted.map((entry, i) => (
          <VoEViolationSection
            key={entry.pair_id}
            entry={entry}
            voeSummary={voe.by_type[entry.violation_type]}
            deltaVsPixel={comparison.delta_voe_spike_jepa_minus_pixel[entry.violation_type]}
            autoPlay={autoPlayFirst && i === 0}
            playbackFps={playbackFps}
          />
        ))}
      </div>
    </div>
  );
}

export { violationSlug };
