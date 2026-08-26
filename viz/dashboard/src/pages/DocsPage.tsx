import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";

const SECTIONS = [
  {
    id: "question",
    title: "Research question",
    body: (
      <>
        <p>
          After self-supervised future-latent prediction only — no physics labels — does a JEPA
          encoder show (i) linearly recoverable physical variables and (ii) elevated prediction
          error specifically at held-out physical violations, more so than a pixel baseline?
        </p>
        <p>
          A positive answer supports the JEPA hypothesis in miniature. A null or nuanced answer
          characterizing <em>where</em> the claim fails is equally valuable.
        </p>
      </>
    ),
  },
  {
    id: "jepa",
    title: "JEPA objective",
    body: (
      <>
        <p>
          Given context frames, an online encoder maps observations to latents. A predictor
          forecasts future latents; an EMA target encoder provides stable targets (anti-collapse).
        </p>
        <BlockMath math="\mathcal{L}_{\text{JEPA}} = \mathrm{SmoothL1}\big(\hat{z}_{t+k}, \mathrm{sg}[\bar{z}_{t+k}]\big)" />
        <p>
          where <InlineMath math="\hat{z}_{t+k}" /> is the predicted latent,
          <InlineMath math="\bar{z}_{t+k}" /> is the EMA-target latent, and
          <InlineMath math="\mathrm{sg}[\cdot]" /> stops gradients through the target branch.
        </p>
      </>
    ),
  },
  {
    id: "pixel",
    title: "Pixel baseline",
    body: (
      <>
        <p>
          Matched-capacity encoder + decoder reconstructs the next frame in pixel space with the
          same training budget.
        </p>
        <BlockMath math="\mathcal{L}_{\text{pixel}} = \mathrm{SmoothL1}(\hat{x}_{t+1}, x_{t+1})" />
        <p>
          Pixel prediction can absorb unpredictable sensory detail; JEPA must discard it to predict
          in latent space — the core contrast under test.
        </p>
      </>
    ),
  },
  {
    id: "probes",
    title: "Linear probes",
    body: (
      <>
        <p>
          Freeze the encoder; train linear heads on ground-truth variables never used during
          self-supervised training:
        </p>
        <ul className="doc-list">
          <li>
            <strong>xy</strong> — position → R²
          </li>
          <li>
            <strong>vxvy</strong> — velocity → R²
          </li>
          <li>
            <strong>mass</strong> — mass → R²
          </li>
          <li>
            <strong>visible</strong> — occlusion flag → accuracy
          </li>
        </ul>
        <p>
          For regression probes, R² on validation:
        </p>
        <BlockMath math="R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}" />
      </>
    ),
  },
  {
    id: "voe",
    title: "Violation of expectation (VoE)",
    body: (
      <>
        <p>
          Inspired by developmental psychology: infants look longer at physically impossible
          events. We construct matched <em>possible</em> vs <em>impossible</em> rollouts with
          violations at timestep <InlineMath math="t^*" />:
        </p>
        <ul className="doc-list">
          <li>Teleport under occlusion</li>
          <li>Pass through wall</li>
          <li>Stop without collision</li>
          <li>Impossible bounce (wrong restitution)</li>
        </ul>
        <p>Surprise at timestep t:</p>
        <BlockMath math="S(t) = \mathrm{SmoothL1}\big(\hat{z}_t, \bar{z}_t\big) \quad \text{(JEPA)}" />
        <p>
          Aggregate spike score compares branches around the violation (higher on impossible =
          physics-like):
        </p>
        <BlockMath math="\Delta_{\text{spike}} = \mathbb{E}_{t \in \mathcal{W}}[S_{\text{imp}}(t)] - \mathbb{E}_{t \in \mathcal{W}}[S_{\text{poss}}(t)]" />
        <p>
          where <InlineMath math="\mathcal{W}" /> is a window around{" "}
          <InlineMath math="t^*" />.
        </p>
      </>
    ),
  },
  {
    id: "env",
    title: "Environment",
    body: (
      <>
        <p>
          Pymunk 2D world: balls/blocks, gravity, collisions, occluders. Frames are 64×64 RGB.
          Train seeds &lt; 9000; VoE evaluation seeds ≥ 9000 (held out).
        </p>
        <p>
          Each episode exports PNG frames plus <code>meta.json</code> with per-body trajectory
          (position, velocity, visibility flags).
        </p>
      </>
    ),
  },
  {
    id: "setup",
    title: "paper_mid setup",
    body: (
      <>
        <table className="doc-table">
          <tbody>
            <tr>
              <th>Train data</th>
              <td>1000 procedural episodes</td>
            </tr>
            <tr>
              <th>Budget</th>
              <td>80 epochs, batch 16, latent dim 256</td>
            </tr>
            <tr>
              <th>Encoder</th>
              <td>Small CNN → 256-d latent</td>
            </tr>
            <tr>
              <th>Predictor</th>
              <td>MLP (ablations: GRU, Transformer)</td>
            </tr>
            <tr>
              <th>Hardware</th>
              <td>CUDA training (RTX 4070 class)</td>
            </tr>
          </tbody>
        </table>
      </>
    ),
  },
];

export function DocsPage() {
  return (
    <div className="page docs-page">
      <header className="page-header">
        <h1>Concepts &amp; math</h1>
        <p className="page-lead">
          A concise reference for the PhysJEPA diagnostic. See also the{" "}
          <a
            href="https://github.com/AvoCahDoe/PhysJEPA/blob/main/docs/report.md"
            target="_blank"
            rel="noopener noreferrer"
          >
            full technical report
          </a>
          .
        </p>
      </header>

      <nav className="docs-toc">
        {SECTIONS.map((s) => (
          <a key={s.id} href={`#${s.id}`}>
            {s.title}
          </a>
        ))}
      </nav>

      <div className="docs-sections">
        {SECTIONS.map((s) => (
          <article key={s.id} id={s.id} className="panel doc-section">
            <h2>{s.title}</h2>
            <div className="doc-body">{s.body}</div>
          </article>
        ))}
      </div>
    </div>
  );
}
