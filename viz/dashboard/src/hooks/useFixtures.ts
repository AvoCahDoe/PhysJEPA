import { useEffect, useState } from "react";
import { fetchJson, paths } from "../lib/loadRun";
import type {
  AblationsDoc,
  ComparisonDoc,
  EpisodeMeta,
  TrainingSummary,
  VoEDemoIndex,
  VoESurpriseDoc,
} from "../types/schema";

export interface FixtureBundle {
  comparison: ComparisonDoc;
  voe: VoESurpriseDoc;
  jepaSummary: TrainingSummary;
  pixelSummary: TrainingSummary;
  episode: EpisodeMeta;
  ablations: AblationsDoc | null;
  voeDemo: VoEDemoIndex | null;
}

export function useFixtures() {
  const [data, setData] = useState<FixtureBundle | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [comparison, voe, jepaSummary, pixelSummary, episode, ablations, voeDemo] =
          await Promise.all([
            fetchJson<ComparisonDoc>(paths.comparison()),
            fetchJson<VoESurpriseDoc>(paths.voeEval()),
            fetchJson<TrainingSummary>(paths.jepaSummary()),
            fetchJson<TrainingSummary>(paths.pixelSummary()),
            fetchJson<EpisodeMeta>(paths.episodeMeta()),
            fetchJson<AblationsDoc>(paths.ablations()).catch(() => null),
            fetchJson<VoEDemoIndex>(paths.voeDemoIndex()).catch(() => null),
          ]);
        if (!cancelled) {
          setData({ comparison, voe, jepaSummary, pixelSummary, episode, ablations, voeDemo });
          setLoading(false);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return { data, loading, error };
}

export const DEFAULT_VOE_TYPE = "impossible_bounce";
