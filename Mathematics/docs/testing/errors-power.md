# Errors, FDR & Power

Every test can be wrong two ways: crying wolf when nothing is there, or missing a real effect. This page maps
the two error types, connects them to the false discovery rate and to precision and recall, and develops
statistical power and the four levers of power analysis.

!!! tip "Rapid Recall"
    Type I error (false positive) is rejecting a true null, with probability exactly $\alpha$ that you set.
    Type II error (false negative) is missing a real effect, with probability $\beta$ that depends on effect
    size, sample size, noise, and $\alpha$. Shrinking one grows the other unless you pull the curves apart with
    more data. The false discovery rate is the fraction of your discoveries that are false, which balloons
    under many comparisons, controlled by Benjamini-Hochberg rather than per-test $\alpha$. Precision is
    $TP/(TP+FP)$ and equals $1-\text{FDR}$; recall is $TP/(TP+FN)$ and equals power $1-\beta$. Power analysis
    fixes three of effect size, sample size, $\alpha$, and variability to solve for the fourth.

## §3 Type I & II Errors, FDR, Precision/Recall

|  | $H_0$ actually TRUE (no effect) | $H_0$ actually FALSE (real effect) |
|---|---|---|
| You reject $H_0$ | Type I error (false positive) | Correct (true positive) |
| You fail to reject $H_0$ | Correct (true negative) | Type II error (false negative) |

**Type I, false positive (cried wolf).** Rejecting a true null, declared an effect that does not exist. **Its
probability is exactly $\alpha$**, by definition. Set $\alpha = 0.05$ and you explicitly accept a 5% chance of
crying wolf when the null is true. You set that rate yourself.

**Type II, false negative (missed the wolf).** Failing to reject a false null, a real effect existed and you
missed it. **Its probability is $\beta$.** You do not set $\beta$ directly; it depends on the true effect size,
sample size, noise, and $\alpha$. Detection is hard when the wolf is small (tiny effect), in fog (high
variance), or glimpsed briefly (small sample).

**The trade-off.** Picture two overlapping curves, the test statistic under $H_0$ and under $H_1$, with a
vertical decision line between them. Slide it right to shrink $\alpha$ (less false-positive tail) and you
automatically grow $\beta$. The **only** way to shrink both is to pull the curves apart: get more data.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 700 270" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Two overlapping distributions showing alpha, beta and power">
      <line x1="30" y1="220" x2="670" y2="220" stroke="#2a3140" stroke-width="1.5"/>
      <path d="M30,220 C150,220 180,70 290,70 C400,70 430,220 540,220 Z" fill="#5db0ff" fill-opacity="0.16" stroke="#5db0ff" stroke-width="2"/>
      <path d="M160,220 C280,220 310,70 420,70 C530,70 560,220 680,220 Z" fill="#e8b04b" fill-opacity="0.16" stroke="#e8b04b" stroke-width="2"/>
      <line x1="370" y1="40" x2="370" y2="228" stroke="#fff" stroke-width="1.5" stroke-dasharray="5 4"/>
      <text x="370" y="20" fill="#e6e9ef" font-size="12" text-anchor="middle">decision threshold</text>
      <path d="M370,220 C400,220 410,150 420,120 C440,180 450,220 480,220 Z" fill="#ef6f6f" fill-opacity="0.45"/>
      <path d="M300,220 C330,220 345,150 370,120 L370,220 Z" fill="#b08bff" fill-opacity="0.4"/>
      <text x="250" y="240" fill="#5db0ff" font-size="12" text-anchor="middle">H&#8320; (no effect)</text>
      <text x="470" y="240" fill="#e8b04b" font-size="12" text-anchor="middle">H&#8321; (real effect)</text>
      <text x="410" y="170" fill="#ef6f6f" font-size="11">&alpha;</text>
      <text x="345" y="170" fill="#b08bff" font-size="11">&beta;</text>
      <text x="500" y="110" fill="#5ad19a" font-size="12">power = area of H&#8321; past the line</text>
    </svg>
<figcaption>The decision line splits two overlapping curves: alpha is the false-positive tail of the null, beta the missed region of the alternative, and power the area of the alternative past the line.</figcaption>
</figure>

**Which error is worse? It depends on cost.**

- **Deadly-disease screening:** a false negative can be fatal; a false positive is a scary follow-up. Tolerate more Type I to crush Type II, so loose threshold.
- **Criminal trials:** "better ten guilty go free than one innocent convicted." The false positive is deemed worse, so a brutally high bar, more false negatives accepted.
- **Spam filter:** a real email lost to spam (false positive) is worse than one spam reaching the inbox (false negative), so tuned conservatively.

### False discovery rate (FDR)

**Why it matters, multiple comparisons.** Run 1000 tests where almost nothing is real. At $\alpha = 0.05$ you
get about 50 false positives from noise. If only about 10 real effects exist and you catch them all, you report
about 60 "discoveries," 50 of which are garbage, so **FDR $\approx 50/60 \approx 83\%$**, even though $\alpha$
was a respectable 0.05. When making thousands of guesses, you control the FDR (e.g. the **Benjamini-Hochberg**
procedure), not per-test $\alpha$.

### Connection to precision and recall

Same 2x2 grid, renamed. In ML, "positive" means predicted the interesting class. False positive equals Type I
error; false negative equals Type II error.

$$\text{Precision} = \frac{TP}{TP + FP} \qquad\text{and}\qquad \text{FDR} = 1 - \text{Precision}$$

Of everything I flagged positive, what fraction was actually positive? Destroyed by false positives.

$$\text{Recall} = \frac{TP}{TP + FN} = \text{Power} = 1 - \beta$$

Of all the actual positives out there, what fraction did I catch? Destroyed by false negatives. This equals
statistical power.

## §4 Statistical Power & Power Analysis

$$\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_0 \text{ is false})$$

Power is your ability to detect a real effect when one exists.

**Where and why we use it.**

- **Before the study (the big one): sample-size planning.** "How many samples so that, if a meaningful effect exists, I have a good chance (usually 80%) of detecting it?" Underpowered studies waste money and, in medicine, are ethically dubious, doomed to find nothing from the start.
- **After the study (diagnostic):** a non-significant result from a low-power study tells you almost nothing, you cannot tell "no wolf" from "eyes closed." This is why "failed to reject $H_0$" is such a weak conclusion.

**The 80% convention** ($\beta \le 0.20$) treats a false negative as about $4\times$ less costly than a false
positive ($\alpha = 0.05$). High-stakes studies aim for 0.90 to 0.95.

### The four levers

Fix any three and the fourth is determined, this is power analysis.

| Lever | Effect on power | Intuition |
|---|---|---|
| Effect size | bigger effect gives more power | A sumo wrestler is easier to spot than a child. Usually nature's, not yours. |
| Sample size n | more data gives more power | Curves get skinnier and pull apart. **Your main knob.** |
| $\alpha$ | looser $\alpha$ gives more power | Easier to reject, but more false positives. Buying power with $\alpha$. |
| Variability $\sigma$ | more noise gives less power | Fatter, more overlapping curves. Reduce via better measurement / paired designs. |

**Power analysis, solving for sample size.**

$$n \approx \frac{2\left(z_{1-\alpha/2} + z_{1-\beta}\right)^{2}}{d^{2}}$$

Required n grows with the certainty you demand (z-values for $\alpha$ and power) and shrinks fast as the effect
$d$ grows. A worked example with $\alpha=0.05$ (so $z=1.96$), 80% power (so $z=0.84$), and effect size
$d=0.3$:

$$n \approx \frac{2(1.96 + 0.84)^{2}}{(0.3)^{2}} = \frac{2(2.80)^{2}}{0.09} = \frac{15.68}{0.09} \approx 174$$

## Interview Questions

**Q1: Define Type I and Type II errors and say which one you control directly.**
A Type I error is a false positive, rejecting a true null, and its probability is exactly the significance
level $\alpha$ that you choose. A Type II error is a false negative, failing to detect a real effect, with
probability $\beta$ that you do not set directly because it depends on the effect size, sample size, noise, and
$\alpha$. You control $\alpha$ directly and influence $\beta$ mainly through sample size.

**Q2: Why can a respectable $\alpha$ of 0.05 still produce mostly false discoveries?**
Because of multiple comparisons. Running many tests where few effects are real produces roughly $\alpha$ times
the number of tests as false positives, which can swamp the handful of true positives. With 1000 tests and 10
real effects you might report 60 discoveries, 50 of them false, an 83% false discovery rate, which is why you
control the FDR with procedures like Benjamini-Hochberg rather than the per-test $\alpha$.

**Q3: How do precision and recall map onto hypothesis-testing concepts?**
Precision, $TP/(TP+FP)$, is one minus the false discovery rate and is destroyed by false positives, the Type I
errors. Recall, $TP/(TP+FN)$, equals statistical power $1-\beta$ and is destroyed by false negatives, the Type
II errors. They are the same confusion matrix viewed from the machine-learning side.

**Q4: What are the four levers of power, and which is the one you usually control?**
Effect size, sample size, the significance level $\alpha$, and the variability $\sigma$. Bigger effects, larger
samples, looser $\alpha$, and lower noise all increase power. Sample size is your main knob, since effect size
is usually fixed by nature, raising $\alpha$ buys power at the cost of more false positives, and reducing noise
requires better measurement or paired designs.
