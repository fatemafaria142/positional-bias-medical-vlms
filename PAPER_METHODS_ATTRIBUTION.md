# Word-Level Attribution of Text Prompts in VoxTell (paper draft)

Draft text for the Methods section. Placeholders in `[brackets]` need the final values from the runs.

## 1. Motivation

VoxTell segments a 3D volume from a free-text prompt such as *"Segment the left cerebral white matter in this image."*
For a bilateral structure, two words carry the meaning: the **side** ("left"/"right") and the **organ name**.
A model that is grounded in the image should use both. A model with a positional bias can ignore the side word and
segment the organ it finds most salient. To test this, we ask a causal question about the prompt:
*how much does each word contribute to segmentation quality?* We answer it with three attribution methods that
differ in cost and rigour: exact Shapley values as the reference, KernelSHAP as a library cross-check, and LIME as
the commonly used baseline.

## 2. Setup: prompt as a cooperative game

**Players.** Ablating raw tokens breaks the grammar and produces out-of-distribution prompts. We therefore define
role-level players, $N=\{\text{position},\ \text{organ}\}$ ($n=2$). Function words (*the, in, this, image*) always stay
in the prompt. A role that is "absent" is replaced by a neutral filler, so the prompt stays well formed.

**Payoff.** For a coalition $S\subseteq N$ of present roles, let $\hat M(S)$ be the VoxTell mask (sigmoid $>0.5$)
for the prompt built from $S$. The payoff $v(S)$ is a segmentation score of $\hat M(S)$. We report four scores, chief
among them $v_{\text{GT}}(S)=\mathrm{Dice}(\hat M(S), M_{\text{GT}})$ (accuracy) and
$v_{\text{self}}(S)=\mathrm{Dice}(\hat M(S), \hat M(N))$ (change relative to the model's own full-prompt output).
With $n=2$ there are only $2^n=4$ prompts per organ, so $v$ is evaluated exhaustively once and cached. All three
methods below read from this same table (except LIME, see 5), which makes them directly comparable.

## 3. Exact Shapley values

**What it is.** The Shapley value is the unique attribution that satisfies efficiency, symmetry, the dummy
property and additivity. It is the standard notion of a "fair" credit split, and here it says how much of the
segmentation score each role is responsible for.

$$
\phi_i(v)=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(n-|S|-1)!}{n!}\Big(v(S\cup\{i\})-v(S)\Big)
$$

**How it is computed.** We evaluate the formula over the cached table, with no sampling and no randomness. For two
roles it reduces to the mean of a role's marginal contribution when added first and when added last:

$$
\phi_{\text{position}}=\tfrac12\big[v(\{p\})-v(\varnothing)\big]+\tfrac12\big[v(\{p,o\})-v(\{o\})\big].
$$

Efficiency holds by construction: $\phi_{\text{position}}+\phi_{\text{organ}}=v(N)-v(\varnothing)$.

**Why it matters for this paper.** $\phi_{\text{position}}$ is a direct measure of side grounding. A value near
zero means that removing "left"/"right" does not change the mask, so the model is not using the side word. It is
the reference against which the other two methods are judged. The cost is $2^n$ model calls, which is trivial for
$n=2$ but prohibitive for full-sentence token attribution ($n\approx 10$ or more per prompt, per subject, per organ).
That is the reason approximate methods matter beyond this study.

## 4. SHAP (KernelSHAP)

**What it is.** SHAP frames attribution as an additive feature-attribution model
$g(z')=\phi_0+\sum_i\phi_i z'_i$ and shows that Shapley values are the only solution that is locally accurate,
consistent and missing-feature-respecting. KernelSHAP estimates them model-agnostically by weighted regression.

$$
\phi=\arg\min_{\phi}\sum_{z'}k(z')\Big(v(h(z'))-\phi_0-\sum_i\phi_i z'_i\Big)^2,\qquad
k(z')=\frac{n-1}{\binom{n}{|z'|}\,|z'|\,(n-|z'|)}
$$

**How it is computed.** We use `shap.KernelExplainer` on the same payoff table, with the all-absent prompt as
background, the full prompt as the explained instance, and `nsamples` $=2^n$. When every coalition is enumerated,
KernelSHAP recovers exact Shapley up to numerical precision. We store `matches_exact_shapley` per row as a
verification of both implementations.

**Why it matters.** This is the check that our hand-written Shapley code is correct. It also shows the
practitioner's route: with more roles or tokens, $2^n$ enumeration is infeasible and one lowers `nsamples`,
trading exactness for cost.

## 5. LIME

**What it is.** LIME explains one prediction by fitting an interpretable surrogate $g$ (a sparse linear model) in the
neighbourhood of the input, weighted by proximity $\pi_x$:

$$
\xi(x)=\arg\min_{g\in G}\ \mathcal L(f,g,\pi_x)+\Omega(g),\qquad
\mathcal L=\sum_{z}\pi_x(z)\big(f(z)-g(z')\big)^2 .
$$

**How it is computed.** We use `lime.lime_text.LimeTextExplainer` with `num_features=1` and `[num_samples]` random
perturbations. LIME removes raw words at random, so it does not use the role structure above. Because the explainer
expects class probabilities, the classifier function returns the pseudo-probability pair $[1-\mathrm{Dice},\ \mathrm{Dice}]$.
The reported `lime_weight` is the surrogate's coefficient for the word.

**Why it matters.** LIME is the most widely used text explainer, so readers will expect it. It differs from the
Shapley methods in two ways that we treat as findings: (i) it is stochastic, so weights vary with the seed and sample
count, and (ii) its perturbations (random word deletion) can produce prompts that VoxTell never sees in practice,
so its weights describe a different neighbourhood than the role ablation does. Agreement between LIME's sign and
$\phi$ supports the conclusion. Disagreement is reported rather than tuned away.

## 6. Comparison

| | Exact Shapley | KernelSHAP | LIME |
|---|---|---|---|
| Role in this study | Reference | Implementation check | Common baseline |
| Perturbation | All $2^n$ role coalitions | Same table, weighted regression | Random raw-word deletion |
| Deterministic | Yes | Yes at `nsamples`$=2^n$ | No |
| Additivity guarantee | Efficiency holds exactly | Holds when fully enumerated | None |
| Model calls | $2^n$ | up to $2^n$ | `[num_samples]` |
| Scales to long prompts | No | Partly | Yes |

## 7. Reading the results

- A large $\phi_{\text{position}}$ under $v_{\text{GT}}$ means the side word is needed for a correct mask, so the model is grounded on side.
- $\phi_{\text{position}}\approx 0$ while $\phi_{\text{organ}}$ carries the whole score means the model ignores side, which is the positional-bias signature.
- Report all three per organ and subject: exact Shapley and SHAP should coincide (`matches_exact_shapley = True`),
  while the LIME deviation quantifies how much a random-deletion explainer can mislead on this task.

## 8. Reproducibility

Code: `scripts/build_groundedness_investigation_notebooks.py` (rows 5-7). Outputs: `word_attribution_*.csv`
(exact Shapley), `shap_attribution_*.csv` (SHAP), `lime_attribution_*.csv` (LIME). Software: `shap` [version],
`lime` [version]; LIME seed [seed], `num_samples` [value].

## References

- Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games II*.
- Ribeiro, M. T., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*.
- Lundberg, S. M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
