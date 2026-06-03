# Variance Tests, ANOVA & Non-Parametric

When you have three or more groups, pairwise t-tests explode the false-positive rate; ANOVA replaces them with
one global test built on a ratio of variances. This page covers ANOVA, order statistics, and the rank-based
non-parametric tests that drop the normality assumption entirely.

!!! tip "Rapid Recall"
    ANOVA tests whether any of three or more group means differ, using $F=\text{between-group variance}/
    \text{within-group variance}$: near 1 means no difference, much greater than 1 means at least one differs.
    It controls the overall false-positive rate that many pairwise t-tests would inflate, and a significant
    result needs a post-hoc test to say which groups differ. Order statistics are your data sorted by rank, the
    basis of medians, percentiles, and non-parametric methods. The Wilcoxon test is the rank-based alternative
    to the t-test, robust to outliers and non-normality. The Kolmogorov-Smirnov test compares distributions by
    the largest gap between their cumulative curves, a classic data-drift detector.

## §9 Tests for Variances & ANOVA

**The problem ANOVA solves.** You have **three or more groups** and want to know if their means differ. First
instinct: run a t-test on every pair. With 3 groups that is 3 tests; it explodes from there. **ANOVA's
solution:** one single test asking "are *any* of these means different?" while holding the overall
false-positive rate at $\alpha$.

**What ANOVA tests.**

- $H_0$: all group means equal ($\mu_1 = \mu_2 = \mu_3 = \dots$)
- $H_1$: at least one differs (it does *not* say which)

$$F = \frac{\text{between-group variance}}{\text{within-group variance}} = \frac{\text{signal (do group means spread apart?)}}{\text{noise (within-group scatter)}}$$

F is the ratio of variation explained by group differences to leftover within-group noise, signal over noise.

- **$F \approx 1$:** between-group is no bigger than noise, means look the same, fail to reject.
- **$F \gg 1$:** means spread further than chance allows, reject; at least one differs.

Because it is a ratio of two variances, it follows the **F-distribution**, with $df_1 = (\text{groups} - 1)$
and $df_2 = (\text{total observations} - \text{groups})$.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 700 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Between vs within group variance">
      <line x1="30" y1="190" x2="670" y2="190" stroke="#2a3140" stroke-width="1.5"/>
      <circle cx="120" cy="150" r="4" fill="#5db0ff"/><circle cx="140" cy="160" r="4" fill="#5db0ff"/><circle cx="105" cy="165" r="4" fill="#5db0ff"/>
      <line x1="90" y1="155" x2="160" y2="155" stroke="#5db0ff" stroke-width="2"/>
      <text x="125" y="205" fill="#5db0ff" font-size="12" text-anchor="middle">Group A</text>
      <circle cx="330" cy="95" r="4" fill="#e8b04b"/><circle cx="350" cy="105" r="4" fill="#e8b04b"/><circle cx="315" cy="110" r="4" fill="#e8b04b"/>
      <line x1="300" y1="100" x2="370" y2="100" stroke="#e8b04b" stroke-width="2"/>
      <text x="335" y="205" fill="#e8b04b" font-size="12" text-anchor="middle">Group B</text>
      <circle cx="540" cy="55" r="4" fill="#5ad19a"/><circle cx="560" cy="65" r="4" fill="#5ad19a"/><circle cx="525" cy="68" r="4" fill="#5ad19a"/>
      <line x1="510" y1="60" x2="580" y2="60" stroke="#5ad19a" stroke-width="2"/>
      <text x="545" y="205" fill="#5ad19a" font-size="12" text-anchor="middle">Group C</text>
      <line x1="30" y1="110" x2="670" y2="110" stroke="#fff" stroke-width="1.2" stroke-dasharray="5 5"/>
      <text x="640" y="104" fill="#e6e9ef" font-size="11" text-anchor="end">grand mean</text>
      <text x="40" y="40" fill="#9aa4b6" font-size="12">Between = how far group means sit from grand mean (signal)</text>
      <text x="40" y="58" fill="#9aa4b6" font-size="12">Within = scatter of dots around their own group line (noise)</text>
    </svg>
<figcaption>ANOVA compares between-group spread (how far group means sit from the grand mean) against within-group scatter (noise around each group's own mean).</figcaption>
</figure>

**Needed, when to use, follow-up.** **Needed** because many pairwise t-tests inflate false positives; ANOVA
gives one global test at controlled $\alpha$. **Use** when one categorical factor has **3 or more levels** and
a continuous outcome. (With exactly 2 groups, ANOVA reduces to the t-test: $F = t^2$.) **Follow-up:** ANOVA says
only *that* some difference exists, not *which*; if you reject, run **post-hoc** pairwise tests corrected for
multiple comparisons (e.g. Tukey's HSD). **Assumptions:** roughly equal variances and normal residuals;
otherwise use Welch's ANOVA or Kruskal-Wallis.

A worked between-group sum of squares (SSB) with three groups of size 3 and a grand mean of 82.22:

$$SSB = 3[(82.33-82.22)^2 + (72.33-82.22)^2 + (92.00-82.22)^2] = 3[0.012 + 97.81 + 95.65] = 3 \times 193.47 = 580.4$$
$$F = \frac{290.2}{5.56} = 52.2$$

## §10 Order Statistics

**The concept:** sort your data smallest to largest; each value in its sorted position is an order statistic.
The smallest is the 1st, the largest the nth, the middle one the median. They are your data values ranked by
position rather than by collection order.

**Where we use it:** medians and percentiles (the basis of box plots), the foundation of nearly all
non-parametric tests (which work on ranks), extreme-value analysis (floods, insurance tails, p99 latency,
literally an order statistic), and robust statistics that resist outliers. The connecting idea: **replace raw
values with their ranks and you no longer depend on normality**, the door into non-parametric testing.

## §11 Wilcoxon & Kolmogorov-Smirnov Tests

**Wilcoxon test.** A non-parametric alternative to the t-test. Two versions: the **signed-rank test**
(counterpart to the paired t-test) and the **rank-sum test** or Mann-Whitney U (counterpart to the two-sample
t-test). Use it when the t-test's assumptions break, small samples, non-normal data, ordinal data (1 to 5
ratings), outliers that would wreck a mean. Because it uses ranks, a wild outlier just becomes "the highest
rank" rather than dragging the result. **Trade-off:** slightly less powerful than the t-test *when* the data
really is normal, you give up a little sensitivity for not needing normality.

**Kolmogorov-Smirnov (KS) goodness-of-fit.** A test for whether a sample came from a particular distribution
(one-sample), or whether two samples share a distribution (two-sample). Use it for checking whether data is
plausibly normal (or any distribution) before a test that assumes it; comparing two samples for the same
distribution. In ML this is a classic way to **detect data drift** (feature distribution in training versus
production). It is fully general: no assumption about the shape, just the biggest gap between cumulative curves.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="KS test: two cumulative curves and the maximum gap">
      <line x1="50" y1="210" x2="660" y2="210" stroke="#2a3140" stroke-width="1.5"/>
      <line x1="50" y1="210" x2="50" y2="30" stroke="#2a3140" stroke-width="1.5"/>
      <text x="40" y="38" fill="#9aa4b6" font-size="11" text-anchor="end">1.0</text>
      <text x="40" y="210" fill="#9aa4b6" font-size="11" text-anchor="end">0</text>
      <path d="M50,210 C200,205 280,120 360,90 C460,55 560,40 660,35" fill="none" stroke="#5db0ff" stroke-width="2.5"/>
      <path d="M50,210 C180,200 250,170 360,140 C470,108 560,55 660,35" fill="none" stroke="#e8b04b" stroke-width="2.5"/>
      <line x1="360" y1="90" x2="360" y2="140" stroke="#ef6f6f" stroke-width="3"/>
      <text x="372" y="118" fill="#ef6f6f" font-size="13">max gap (D)</text>
      <text x="500" y="55" fill="#5db0ff" font-size="12">reference / sample 1</text>
      <text x="120" y="195" fill="#e8b04b" font-size="12">sample 2</text>
      <text x="350" y="235" fill="#9aa4b6" font-size="12" text-anchor="middle">value &rarr;</text>
    </svg>
<figcaption>The Kolmogorov-Smirnov statistic D is the largest vertical gap between two cumulative distribution curves.</figcaption>
</figure>

## Interview Questions

**Q1: Why use ANOVA instead of many pairwise t-tests?**
Because running a t-test on every pair of groups inflates the overall false-positive rate as the number of
comparisons grows. ANOVA gives a single global test of whether any group mean differs, holding the family-wise
error at $\alpha$. It works by comparing between-group variance to within-group variance through an F-statistic.

**Q2: What does an F-statistic near 1 versus much greater than 1 tell you in ANOVA?**
Near 1 means the spread between group means is no larger than the within-group noise, so the means look the
same and you fail to reject. Much greater than 1 means the group means spread apart more than chance allows, so
you reject and conclude at least one group differs, though ANOVA does not say which, requiring a post-hoc test.

**Q3: Why are rank-based non-parametric tests robust to outliers?**
Because they replace raw values with their ranks, so an extreme value simply becomes the highest or lowest rank
rather than dragging a mean. This removes the dependence on normality and on the magnitude of outliers, at the
cost of slightly less power than the t-test when the data really is normal.

**Q4: What does the Kolmogorov-Smirnov test measure, and how is it used in ML?**
It measures the largest vertical gap between two cumulative distribution functions, either a sample against a
reference distribution or two samples against each other, with no assumption about shape. In machine learning it
is a standard way to detect data drift by comparing a feature's distribution in training versus production.
