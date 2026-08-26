export type NullableSeries = (number | null)[];

export interface ProbeMetrics {
  train_mse?: number;
  val_mse?: number;
  train_r2?: number;
  val_r2?: number;
  train_acc?: number;
  val_acc?: number;
  train_bce?: number;
  val_bce?: number;
  n_train?: number;
  n_val?: number;
}

export interface VoETypeSummary {
  t_star: number;
  t: number[];
  possible_mean: NullableSeries;
  impossible_mean: NullableSeries;
  possible_std?: NullableSeries;
  impossible_std?: NullableSeries;
  spike_score: number;
  pre_tstar_abs_gap?: number;
  n_pairs?: number;
}

export interface VoESurpriseDoc {
  schema_version: number;
  run_id?: string;
  model_type?: string;
  surprise_metric?: string;
  context_len?: number;
  by_type: Record<string, VoETypeSummary>;
}

export interface ModelEvalSummary {
  model_type: string;
  ckpt: string;
  run_id: string;
  probes: Record<string, ProbeMetrics>;
  voe_by_type: Record<
    string,
    { spike_score: number; pre_tstar_abs_gap: number; t_star: number }
  >;
  voe_surprise_metric?: string;
}

export interface ComparisonDoc {
  schema_version: number;
  jepa: ModelEvalSummary;
  pixel: ModelEvalSummary;
  delta_voe_spike_jepa_minus_pixel: Record<string, number>;
}

export interface TrainingSummary {
  schema_version: number;
  run_id: string;
  curves: {
    step: number[];
    loss: number[];
    latent_std: number[];
    latent_norm?: number[];
    val_loss?: number[];
  };
  best_val_loss: number | null;
  config?: Record<string, unknown>;
  model_type?: string;
}

export interface EpisodeBody {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  angle: number;
  visible: boolean;
  mass: number;
}

export interface EpisodeMeta {
  schema_version: number;
  split: string;
  seed: number | null;
  T: number;
  resolution: [number, number];
  fps: number;
  violation: { type: string; t_star: number; pair_id?: string } | null;
  trajectory: { t: number; bodies: EpisodeBody[] }[];
}

export type TabId =
  | "compare"
  | "voe"
  | "voe_replay"
  | "probes"
  | "train"
  | "episode"
  | "ablations";

export interface VoEDemoEntry {
  pair_id: string;
  violation_type: string;
  t_star: number;
  base: string;
  description?: string | null;
}

export interface VoEDemoIndex {
  schema_version: number;
  pairs: VoEDemoEntry[];
}

export interface OcclusionAblationRow {
  label: string;
  voe_dir?: string;
  mean_occlusion_frames: number | null;
  spike_score: number | null;
  pre_tstar_abs_gap?: number | null;
  t_star?: number | null;
  ckpt?: string;
}

export interface ArchitectureAblationRow {
  predictor_type: string;
  ckpt?: string;
  mean_voe_spike: number;
  voe_by_type?: Record<string, { spike_score: number; t_star?: number }>;
  probes?: Record<string, { val_r2?: number; val_acc?: number; val_mse?: number }>;
}

export interface AblationsDoc {
  schema_version: number;
  occlusion_duration: {
    description?: string;
    results: OcclusionAblationRow[];
  };
  architecture: {
    description?: string;
    results: ArchitectureAblationRow[];
  };
}