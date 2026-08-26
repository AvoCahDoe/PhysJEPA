Good choice — this one has a clean story: self-supervised prediction, no labels, and you're asking a question people actually care about (does JEPA learn physics or just texture statistics). Here's the full plan.

1. Core research question

Does a JEPA-style predictor, trained only to predict masked/future latents (no reconstruction, no physics supervision), spontaneously encode:

Object permanence (occluded objects still "exist" in latent space)
Naive physics priors (gravity, inertia, non-penetration of solids)
Causal collision dynamics (momentum transfer, bouncing)

The paper you're implicitly rebutting/extending: work on video prediction models tested against violation-of-expectation (VoE) paradigms — the finding that most models are surprised by nothing, because pixel-space prediction lets them cheat (blur through impossible events rather than flag them). JEPA's whole pitch (LeCun 2022, "A Path Towards Autonomous Machine Intelligence") is that predicting in latent space instead of pixel space should force the model to learn abstractions. This project empirically tests that claim in a controlled, interpretable setting.

2. Environment: build your own simple physics simulator

Don't use a heavy existing benchmark (IntPhys, Physion, CATER) as your training environment — too slow to iterate on and you won't fully control confounds. Instead:

Build a 2D simulator (pymunk or a custom Box2D wrapper) with balls/blocks, gravity, collisions, occluders.
Procedurally generate scenes: object count, mass, initial velocity, occluder position, randomized.
Render as simple sprites (64×64 or 84×84 grayscale/RGB) — keep it cheap.
Separately, hand-construct a small VoE probe set: matched pairs of "possible" vs "impossible" rollouts (object teleports during occlusion, object passes through wall, object stops without collision, object changes mass/speed impossibly). This probe set is never used in training — it's held out purely for evaluation. This is the part reviewers/committees will find compelling because it's a clean, controlled diagnostic.
3. Model architecture
Encoder: small CNN (ResNet-lite) mapping frame → latent.
Predictor: transformer or MLP predicting latent at t+k from latents at t, t-1, ... (context frames), following V-JEPA's masked/future-latent prediction objective.
Target encoder: EMA copy of the encoder (standard JEPA anti-collapse trick — no negative pairs needed).
Train purely self-supervised on your procedurally generated rollouts. No labels, no physics equations given.

Keep it genuinely small: this should run on a single GPU (or Colab-class compute) in days, not weeks. That's important for a PhD-application project — you want it finished, not aspirational.

4. Probing methodology (the core contribution)

Two complementary evaluations:

(a) Linear probes — freeze the encoder, train a linear classifier on top of latents to predict ground-truth physical variables (position, velocity, mass, whether occluded object still exists) that the model was never trained on. If a linear probe recovers these well, that's evidence the representation organizes physics-relevant structure, not just JEPA memorizing task-specific shortcuts.

(b) Surprise/violation-of-expectation test — the more interesting one. Feed the model a rollout, then compute the predictor's prediction error (or predicted-vs-actual latent distance) at the moment of a physical violation vs. a matched possible event. If the model is doing something physics-like, prediction error should spike specifically at the impossible event, mirroring the VoE literature in infant cognition (Baillargeon, Spelke) that this whole line of ML work self-consciously borrows from. This is your headline plot: "surprise" over time, possible vs impossible event, showing a spike or no spike.

(c) Ablations (this is what makes it a project rather than a demo):

JEPA (latent prediction) vs. a pixel-reconstruction baseline (e.g. plain autoencoder / video prediction model) — does latent-space prediction actually help vs. pixel-space, as LeCun's argument predicts?
Vary occlusion duration — does object permanence degrade with longer occlusion?
Vary context length / architecture (transformer vs. RNN predictor) — does it matter for how long physics is "remembered"?
5. What a strong result looks like (and what a null result looks like — plan for both)
Positive story: JEPA shows VoE-spike behavior and good linear probe accuracy; pixel-reconstruction baseline doesn't (or does much less). This directly supports the JEPA hypothesis and is a nice clean finished paper/report.
Negative/nuanced story (equally publishable, arguably more interesting for a PhD app since it shows independent thinking): JEPA encodes some physics (position, velocity) linearly but fails at genuine object permanence under long occlusion, revealing a specific limitation. This is actually a better SOP story because "I found a real limitation and characterized it carefully" reads as more mature research taste than "I confirmed the hypothesis."

Either outcome is a fine result. Don't optimize for the model working perfectly — optimize for a clean, well-controlled experiment.

6. Timeline (assuming ~2–3 months part-time)
Phase	Time	Output
Lit review + simulator build	2 weeks	Working procedural physics generator + VoE probe set
JEPA implementation + training pipeline	2–3 weeks	Training loop, EMA target encoder, sanity-check on toy data
Baseline (pixel-prediction model)	1 week	Comparable-capacity baseline
Probing + VoE experiments	2–3 weeks	Linear probe results, surprise curves
Ablations	1–2 weeks	Occlusion duration, architecture variants
Write-up (report/paper-style, + repo + README)	2 weeks	GitHub repo, arXiv-style writeup, plots
7. Related work you should read/cite (I can pull current papers if you want)
LeCun's JEPA position paper; I-JEPA and V-JEPA papers (Meta AI)
Physion / IntPhys / CATER benchmarks (physical reasoning benchmarks in ML)
Piloto et al., "Intuitive physics learning in a deep-learning model inspired by developmental psychology" (a very close prior work — you should explicitly position against this)
Baillargeon's VoE paradigm in developmental psychology (the methodological ancestor)
Any 2025–2026 follow-ups on V-JEPA-2 or world models for physical reasoning (worth a fresh search before you start, since this moves fast)
8. For your SOP

Frame it as: "I designed a controlled diagnostic to test a specific, falsifiable claim from the JEPA research agenda — that latent-space prediction induces physical abstractions that pixel-space prediction does not — using methodology borrowed from developmental psychology. I found [X]." That sentence alone signals you can (a) identify a testable claim in a trendy area instead of just building on top of it, (b) design controlled experiments, (c) engage across ML and cognitive science, which is exactly the "unusual combination" admissions committees like.