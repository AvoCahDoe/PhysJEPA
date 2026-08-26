import { useEffect, useState } from "react";
import { AblationsPanel } from "./components/AblationsPanel";
import { ComparisonOverview } from "./components/ComparisonOverview";
import { EpisodePlayer } from "./components/EpisodePlayer";
import { Layout } from "./components/Layout";
import { ProbeComparison } from "./components/ProbeComparison";
import { TrainingCurves } from "./components/TrainingCurves";
import { VoESurprisePanel } from "./components/VoESurpriseChart";
import { fetchJson, paths, violationTypes } from "./lib/loadRun";
import type {
  AblationsDoc,
  ComparisonDoc,
  EpisodeMeta,
  TabId,
  TrainingSummary,
  VoESurpriseDoc,
} from "./types/schema";
import "./index.css";

export default function App() {
  const [tab, setTab] = useState<TabId>("compare");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [comparison, setComparison] = useState<ComparisonDoc | null>(null);
  const [voe, setVoe] = useState<VoESurpriseDoc | null>(null);
  const [jepaSummary, setJepaSummary] = useState<TrainingSummary | null>(null);
  const [pixelSummary, setPixelSummary] = useState<TrainingSummary | null>(null);
  const [episode, setEpisode] = useState<EpisodeMeta | null>(null);
  const [ablations, setAblations] = useState<AblationsDoc | null>(null);
  const [voeType, setVoeType] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [cmp, voeDoc, jepa, pixel, ep, abl] = await Promise.all([
          fetchJson<ComparisonDoc>(paths.comparison()),
          fetchJson<VoESurpriseDoc>(paths.voeEval()),
          fetchJson<TrainingSummary>(paths.jepaSummary()),
          fetchJson<TrainingSummary>(paths.pixelSummary()),
          fetchJson<EpisodeMeta>(paths.episodeMeta()),
          fetchJson<AblationsDoc>(paths.ablations()).catch(() => null),
        ]);
        if (cancelled) return;
        setComparison(cmp);
        setVoe(voeDoc);
        setJepaSummary(jepa);
        setPixelSummary(pixel);
        setEpisode(ep);
        setAblations(abl);
        const types = violationTypes(voeDoc);
        if (types.length > 0) setVoeType(types[0]);
        setLoading(false);
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

  let content = null;
  if (comparison && voe && jepaSummary && pixelSummary && episode) {
    switch (tab) {
      case "compare":
        content = <ComparisonOverview data={comparison} />;
        break;
      case "voe":
        content = (
          <VoESurprisePanel
            byType={voe.by_type}
            selected={voeType}
            onSelect={setVoeType}
          />
        );
        break;
      case "probes":
        content = <ProbeComparison data={comparison} />;
        break;
      case "ablations":
        content = ablations ? (
          <AblationsPanel data={ablations} />
        ) : (
          <p className="hint">
            No ablations fixture yet. Run{" "}
            <code>python scripts/run_ablations.py</code> then{" "}
            <code>python scripts/sync_viz_fixtures.py</code>.
          </p>
        );
        break;
      case "train":
        content = <TrainingCurves jepa={jepaSummary} pixel={pixelSummary} />;
        break;
      case "episode":
        content = <EpisodePlayer meta={episode} />;
        break;
    }
  }

  return (
    <Layout active={tab} onTab={setTab} loading={loading} error={error}>
      {content}
    </Layout>
  );
}
