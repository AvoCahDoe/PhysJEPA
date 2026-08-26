function param(name: string, fallback: string): string {
  const url = new URL(window.location.href);
  return url.searchParams.get(name) ?? fallback;
}

export async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to load ${path}: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const paths = {
  comparison: () => param("comparison", "/fixtures/sample_comparison.json"),
  voeEval: () => param("voe", "/fixtures/sample_eval.json"),
  jepaSummary: () => param("jepa_summary", "/fixtures/jepa_summary.json"),
  pixelSummary: () => param("pixel_summary", "/fixtures/pixel_summary.json"),
  episodeMeta: () => param("episode_meta", "/fixtures/sample_episode/meta.json"),
  episodeFrame: (t: number) =>
    `${param("episode_base", "/fixtures/sample_episode")}/frames/${String(t).padStart(6, "0")}.png`,
  ablations: () => param("ablations", "/fixtures/sample_ablations.json"),
  voeDemoIndex: () => param("voe_demo", "/fixtures/voe_demo/index.json"),
  voeDemoMeta: (base: string, branch: "possible" | "impossible") =>
    `${param("voe_demo_base", "/fixtures/voe_demo")}/${base}/${branch}/meta.json`,
  voeDemoFrame: (base: string, branch: "possible" | "impossible", t: number) =>
    `${param("voe_demo_base", "/fixtures/voe_demo")}/${base}/${branch}/frames/${String(t).padStart(6, "0")}.png`,
};

export function violationTypes(doc: { by_type: Record<string, unknown> }): string[] {
  return Object.keys(doc.by_type);
}

export function formatViolationLabel(key: string): string {
  return key.replace(/_/g, " ");
}
