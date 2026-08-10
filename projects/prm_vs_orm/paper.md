# Process vs. Outcome Reward Models for RL on a 370M Math Reasoner

**A controlled comparison on a shared Olmo3-370M backbone**

> Status: methods + intermediate results complete. The two final GRPO eval numbers
> (ORM-RL, PRM-RL) are pending — both arms are queued on 8×A100 at time of writing —
> and are marked **[pending]** below.

---

## Abstract

We ask a single question: given a fixed small policy, does a **process reward model
(PRM)** — which scores intermediate reasoning steps — produce a better math reasoner
under reinforcement learning than an **outcome reward model (ORM)** that scores only the
final answer? To isolate the reward signal as the only moving part, we train both reward
models from the *same* SFT-initialized backbone, differing only in head width and where
supervision is placed, and use each as the reward in an otherwise identical GRPO run on
the *same* SFT policy. The policy is a 370M-parameter Olmo3. We report the SFT baseline
and a best-of-$N$ reranking gate; the two RL arms are in progress.

---

## 1. Setup and notation

A policy $\pi_\theta$ (a 370M decoder) maps a problem $x$ to a solution
$y=(y_1,\dots,y_T)$, a token sequence optionally segmented into reasoning steps
$s_1,\dots,s_K$ with last-token indices $\tau_1<\dots<\tau_K$. Each problem carries a
gold answer $a^\star(x)$. A fixed extractor $\mathrm{ext}(y)$ reads the final
`# Answer\n\n<...>` field, and correctness is
$c(x,y)=\mathbb{1}[\,\mathrm{ext}(y)=a^\star(x)\,]\in\{0,1\}$, graded by open-instruct's
`MathVerifier`/`GSM8KVerifier` (reused, not re-implemented). One dolma2 HF tokenizer and
one prompt template are used at **every** stage — a correctness requirement, since scores
must transfer across SFT, RM, and RL.

**Pipeline.** base $\to$ SFT $\to$ {ORM, PRM} $\to$ best-of-$N$ gate $\to$ GRPO$\times 2$
$\to$ eval. Primary metric GSM8K, secondary MATH.

---

## 2. Data

Supervision comes from **PRM800K** (Lightman et al., 2023): human step-level ratings of
model solutions to competition math. Each record gives a problem, a gold answer, and a
solution decomposed into steps; each step carries an integer rating
$\ell\in\{-1,0,+1\}$ (negative / neutral / positive) and, for phase-1 records, several
rated alternative completions per step. We derive three corpora under one fixed template:

- **SFT corpus** — $(x, y)$ with $y$ the chosen trajectory plus the final answer line.
- **ORM corpus** — full solutions with a single outcome label $c(x,y)$.
- **PRM corpus** — per-step targets $\ell_k$ with their boundary indices $\tau_k$.

GRPO prompts are GSM8K-train (primary) and MATH-train, each carrying $a^\star$.

---

## 3. Base model and SFT

The base is a plain-attention Olmo3-370M pretrained on 10B tokens, converted once from
its OLMo-core DCP checkpoint to Hugging Face (the converter's logit-level validation is
kept on). SFT minimizes the completion-masked negative log-likelihood

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
= -\,\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{SFT}}}
\left[\frac{1}{|y|}\sum_{t=1}^{|y|}\log \pi_\theta\!\left(y_t \mid x, y_{<t}\right)\right],
$$

with the loss masked to completion tokens only. This one SFT model is the **shared init
for both reward models** and the **shared policy init for both GRPO arms** — the pivot
that makes the comparison controlled.

---

## 4. The two reward models (the core contrast)

Both reward models share a backbone $f_\phi$ warm-started from the SFT model, producing a
hidden state $h_t\in\mathbb{R}^d$ at each position, and a single linear head
$W\in\mathbb{R}^{d\times C}$ giving logits $z_t = W^\top h_t\in\mathbb{R}^{C}$. The arms
differ *only* in $C$ and in which positions carry a label — the cleanest
process-vs-outcome contrast.

### 4.1 Outcome RM (ORM)

$C=1$. The whole-solution correctness label $c(x,y)$ is **broadcast to every response
token** and trained with binary cross-entropy,

$$
\mathcal{L}_{\mathrm{ORM}}(\phi)
= -\,\mathbb{E}\!\left[\frac{1}{|y|}\sum_{t}\Big(c\,\log\sigma(z_t)+(1-c)\,\log(1-\sigma(z_t))\Big)\right],
\qquad \sigma(u)=\tfrac{1}{1+e^{-u}} .
$$

At inference we read the **last response token** (a Cobbe-style verifier):

$$
r_{\mathrm{ORM}}(x,y)=\sigma\!\left(z_{t_{\mathrm{end}}}\right)\in(0,1).
$$

### 4.2 Process RM (PRM)

$C=3$ over $\{\text{neg},\text{neu},\text{pos}\}$, with
$p_\phi(\cdot\mid x,y_{\le \tau_k})=\mathrm{softmax}(z_{\tau_k})$. Supervision is placed at
each **step-boundary token** $\tau_k$, and — following the paper — only up to the first
negative step $k^\star=\min\{k:\ell_k=-1\}$ (nothing after the first mistake is
labeled):

$$
\mathcal{L}_{\mathrm{PRM}}(\phi)
= -\,\mathbb{E}\!\left[\sum_{k\le k^\star}\log p_\phi\!\left(\ell_k \mid x, y_{\le \tau_k}\right)\right].
$$

Post-tokenization alignment of $\tau_k$ to HF token indices is the principal correctness
check for this arm. At scoring time **neutral counts as positive**, so per-step
correctness is

$$
q_k = p_\phi(\text{pos}\mid\cdot) + p_\phi(\text{neu}\mid\cdot),
$$

and the solution score is the product aggregation (Lightman et al.'s best; min and mean
are also logged):

$$
r_{\mathrm{PRM}}(x,y)=\prod_{k=1}^{K} q_k,
\qquad
\log r_{\mathrm{PRM}} = \sum_{k=1}^{K}\log q_k .
$$

The log form makes explicit a mild **length bias**: since $\log q_k\le 0$, the score is
monotone non-increasing in the number of steps $K$.

---

## 5. Best-of-$N$ gate (go / no-go)

Before spending GPU-days on RL, we test whether either RM is a useful ranker at all. For
each test problem we draw $N$ solutions $\{y^{(i)}\}_{i=1}^N\sim\pi_{\mathrm{SFT}}(\cdot\mid x)$
and compare three selectors:

$$
\hat y_{\mathrm{RM}} = \arg\max_i\, r(x,y^{(i)}),\quad r\in\{r_{\mathrm{ORM}},r_{\mathrm{PRM}}\};
\qquad
\hat y_{\mathrm{maj}} = \text{modal } \mathrm{ext}(y^{(i)}),
$$

reporting selection accuracy $\mathbb{E}_x[c(x,\hat y)]$ and the ceiling
$\mathrm{pass@}N=\mathbb{E}_x\big[\mathbb{1}[\exists i:\,c(x,y^{(i)})=1]\big]$. **Decision
rule:** proceed to RL only if some RM's best-of-$N$ beats majority vote.

**Result (measured).** Across both RM training regimes, RM reranking did **not** beat
majority vote on the GSM8K gate slice:

| RM training regime | best RM best-of-$N$ acc. | majority-vote acc. | verdict |
|---|---|---|---|
| off-policy (PRM800K) | 2.5% | 3.0% | no-go |
| on-policy (SFT samples) | 1.5% | 2.5% | no-go |

The gap is small and the absolute numbers are near the floor — as expected for a 370M
model on GSM8K. The reading is that at this scale the **policy**, not the reward model, is
the binding constraint: neither RM can rank solutions the base policy rarely produces
correctly (low pass@$N$). RL was nonetheless run past this gate as a deliberate decision,
to measure the PRM-vs-ORM *difference under RL* even if the expected effect is small.

---

## 6. Reward bridge: a learned RM as the GRPO reward

open-instruct's GRPO reward normally comes from a ground-truth checker. We replace it with
a `VerifierFunction` $R$ that loads a trained RM and returns its score for each rollout,

$$
R(x,y) = r_{\mathrm{type}}(x,y),\qquad \mathrm{type}\in\{\mathrm{ORM},\mathrm{PRM}\},
$$

registered through the sanctioned `--reward_plugins` hook and routed to by a per-prompt
dataset key (no edit to core open-instruct). The RM runs as a batched scorer over
completions; VRAM sharing with the vLLM rollout engines on the same node is the main new
integration risk.

---

## 7. Reinforcement learning: GRPO

We use group-relative policy optimization (GRPO): value-network-free, with the group mean
as the baseline. For a prompt $x$ we sample a group of $G$ completions
$\{y^{(i)}\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid x)$, score them
$R_i=R(x,y^{(i)})$, and form group-normalized advantages

$$
A_i = \frac{R_i-\mu}{\sigma+\varepsilon},\qquad
\mu=\frac1G\sum_{j}R_j,\quad
\sigma=\sqrt{\frac1G\sum_j (R_j-\mu)^2},
$$

broadcast to every token of $y^{(i)}$. The policy maximizes a clipped surrogate with a KL
penalty to the frozen SFT reference $\pi_{\mathrm{ref}}=\pi_{\mathrm{SFT}}$:

$$
\mathcal{J}(\theta)=
\mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y^{(i)}|}\sum_{t}
\min\!\Big(\rho_{i,t}\,A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\,A_i\Big)\right]
-\ \beta\,\mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\mathrm{ref}}\right],
$$

$$
\rho_{i,t}=\frac{\pi_\theta\!\left(y^{(i)}_t\mid x,y^{(i)}_{<t}\right)}
{\pi_{\theta_{\mathrm{old}}}\!\left(y^{(i)}_t\mid x,y^{(i)}_{<t}\right)} .
$$

Because $A_i$ centers each group at zero mean, only *relative* reward within a group
matters — which is exactly why a noisy learned RM can still drive learning, and equally
why it can be reward-hacked; the KL term $\beta$ and light format/length checks blunt
that. Both arms are identical except for which RM defines $R$.

**Configuration.** $G=8$ samples/prompt, $32$ unique prompts/rollout, KL weight
$\beta=0.05$, learning rate $3\times10^{-7}$, temperature $0.8$, response length $512$,
bf16 on 8×A100 (4 learners + 4 vLLM engines via Ray). With $50000$ episodes and
$32\times8=256$ episodes/step this is $\lfloor 50000/256\rfloor = 195$ optimization steps.

---

## 8. Evaluation protocol

For $M\in\{\text{SFT},\text{ORM-RL},\text{PRM-RL}\}$ we report GSM8K (primary) and MATH
(secondary), each with greedy accuracy and majority-vote over $k=8$ samples:

$$
\mathrm{acc}_{\mathrm{greedy}}(M)=\mathbb{E}_x\big[c(x,\hat y^{\,M}_{\mathrm{greedy}})\big],
\qquad
\mathrm{maj@}8(M)=\mathbb{E}_x\big[c\big(x,\ \text{modal }\mathrm{ext} \text{ of } 8\ \text{samples}\big)\big],
$$

graded by the same verifier used for labels and the gate. The headline comparison is
$\mathrm{PRM\text{-}RL}$ vs. $\mathrm{ORM\text{-}RL}$, with SFT as the floor.

---

## 9. Results

**Baselines and gate (measured).** SFT establishes the floor; the best-of-$N$ gate
(§5) found RM reranking below majority vote in both training regimes — a small, near-floor
signal consistent with the policy being the bottleneck at 370M.

**RL arms (pending).** Both GRPO arms share every hyperparameter and the SFT init; only
$R$ differs.

| Model | GSM8K greedy | GSM8K maj@8 | MATH greedy | MATH maj@8 |
|---|---|---|---|---|
| SFT (baseline) | — | — | — | — |
| ORM-RL | **[pending]** | **[pending]** | **[pending]** | **[pending]** |
| PRM-RL | **[pending]** | **[pending]** | **[pending]** | **[pending]** |

The primary read is the signed difference $\mathrm{acc}(\text{PRM-RL})-\mathrm{acc}(\text{ORM-RL})$
on GSM8K greedy, with maj@8 and MATH as corroboration.

---

## 10. Discussion and limitations

- **Scale caps the effect.** At 370M, GSM8K accuracy is single-digit; the gate shows
  pass@$N$ is low, so a reranker/RL reward has little correct-vs-incorrect structure to
  exploit. The PRM-vs-ORM difference may be small or within noise — an honest a-priori
  expectation, not a post-hoc excuse.
- **Product aggregation is length-biased.** $r_{\mathrm{PRM}}=\prod_k q_k$ penalizes
  longer solutions; min/mean aggregations are logged as robustness checks.
- **Reward hacking.** A learned RM as reward invites degenerate completions; the KL
  anchor to SFT and format/length checks are the guardrails, monitored via reward-mean
  and completion-length curves.
- **One label source.** Both RMs derive from PRM800K; a different step-labeling scheme
  could shift the balance.

---

## 11. Reproducibility

Every stage runs on the eduLLM platform from a pinned commit: base$\to$HF conversion via
OLMo-core; SFT, GRPO, and (forked for scored labels) reward-model training via
open-instruct; vLLM for rollouts. Checkpoints and derived data live in S3; training is
tracked in Weights & Biases. The ORM/PRM trainer and the RM$\to$reward bridge are the only
custom, experiment-defining code; everything else is off-the-shelf open-instruct.

---

### References

- K. Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021 (ORM / last-token verifier).
- H. Lightman et al., *Let's Verify Step by Step*, 2023 (PRM800K; per-step reward, product aggregation).
- Z. Shao et al., *DeepSeekMath* / GRPO, 2024 (group-relative advantage).
- AllenAI, *open-instruct* (SFT, GRPO with Ray+vLLM, verifier interface).
