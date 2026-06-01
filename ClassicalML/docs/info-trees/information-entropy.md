# Information & Entropy

Before any loss function or split criterion, there is one idea: information is the number of optimal yes/no questions needed to pin down an outcome. This page builds it rigorously, surprise as one quantity seen before and after the event, the Cauchy functional-equation argument that forces the logarithm, and the definition, shape, and units of entropy.

!!! tip "Rapid Recall"
    The information in an outcome of probability p is \(-\log_2 p\) bits, the number of halvings needed to single it out, and surprise and information are the same number viewed before versus after the event. Additivity of independent surprises plus the weakest regularity forces \(I(p)=-C\log p\) by Cauchy's functional equation, with the constant C just a unit choice that cancels from every argmax and gradient. Entropy is expected surprise, \(H=-\sum_i p_i\log_2 p_i\), a symmetric dome for the binary case peaking at 1 bit when \(p=0.5\) and zero at purity.

## §1 What information actually is

Shannon "information" has nothing to do with meaning, facts, or content. He deliberately threw meaning away. It is a count of something physical.

!!! note "Core definition"
    The *information content* of an outcome is the number of optimal yes/no questions needed to identify it. A *bit* is one yes/no question's worth, a count of questions, not a substance.

### The concrete game

Pick one of **8 equally likely** items. Identify it with yes/no questions by *halving the possibilities each time*: 8 → 4 → 2 → 1. Always exactly **3 questions**, and \({\log}_2 8=3\). The log is literally counting halvings: to go from \(N\) candidates to 1 you need \({\log}_2 N\) halvings.

So an outcome of probability \(p=1/8\) carries \(-{\log}_2(1/8)=3\) bits. Rarer outcome → smaller \(p\) → more halvings → more questions → more information. A near-certain event (\(p\approx 1\)) needs about 0 questions, you already know.

### Why "bits": storage and transmission

Encode each yes/no answer as one binary digit (yes = 1, no = 0). Three answers form a 3-digit code like `101`; \(2^3=8\) codes cover 8 items exactly. So three phrasings are the **same number**:

- 3 yes/no questions to identify it,
- 3 binary digits to store / transmit it,
- \(-{\log}_2(1/8)=3\) bits of information.

"This cost me 3 bits" = "an optimal questioner needs 3 questions / I must physically write 3 binary digits to pin it down." This is why we care: it is the answer to *how much memory* and *how much bandwidth* a message consumes.

!!! note "What is information?"
    Reduction of uncertainty, measured by how many binary choices it takes to remove it. A message is informative precisely when it rules out many possibilities. Not meaning, uncertainty removed.

## §2 Surprise and information: one quantity, two viewpoints

The block that confused us: surprise and information aren't two things to "fit together," they're the same number viewed before vs after the event.

Take one outcome with probability \(p\). The quantity \(-{\log}_2 p\) has two readings:

| Viewpoint | Name | Question |
| --- | --- | --- |
| Before it happens (forward) | **Surprise** | "How shocked will I be if this occurs?" |
| After it happens (backward) | **Information** | "How much did learning this resolve?" |

They are *forced* to be equal because both equal "how many possibilities did this outcome eliminate." A low-probability outcome was hiding among many alternatives, so it is both surprising (you didn't expect this one out of many) and informative (learning it killed all the others). High surprise is equivalent to many possibilities ruled out is equivalent to high information. They cannot come apart.

!!! note "Mental model to lock in"
    $$\text{surprise of an outcome}\,=\,\text{information gained by observing it}\,=\,-{\log}_2 p$$
    Not "related," *identical*, named for whether you stand before the event (surprise) or after it (information).

## §3 Why the entropy formula is what it is

The honest derivation, not "a function that does these three things is \(-\log p\)," but a theorem that forces it uniquely.

### The one real axiom: independent surprises add

For independent events with probabilities \(p\) and \(q\), surprise should satisfy \(I(pq)=I(p)+I(q)\). Why additivity?

- **Information accumulates linearly.** Ten independent coin flips carry ten times the information of one, no overlap, because independent. Linear scaling with count means it adds, not multiplies. If surprise multiplied, two flips would be the *square* of one flip's surprise, nonsense as "amount learned."
- **It bridges two worlds.** Probabilities of independent events *multiply* (\(pq\)); we want information to *add*. We need a map from the multiplicative world of probabilities to the additive world of information. That requirement is the axiom.

### The theorem: additivity plus continuity forces a logarithm

Want \(I(p)\) on \((0,1]\) with (1) \(I(pq)=I(p)+I(q)\) and (2) \(I\) continuous (or merely monotonic / locally bounded, very weak).

!!! note "Claim and proof sketch"
    **Claim:** the only solutions are \(I(p)=-C\log p\), \(C>0\).

    **Proof sketch (Cauchy's functional equation).** Let \(g(x)=I(e^{-x})\) for \(x\ge 0\). With \(p=e^{-x},\,q=e^{-y}\), additivity becomes
    $$g(x+y)=g(x)+g(y).$$
    The only continuous (or monotonic / locally bounded) solutions are linear: \(g(x)=Cx\). Translate back: \(I(p)=-C\log p\).

So you genuinely *cannot* have any other function. Monotonicity ("rarer = more surprising") just pins \(C>0\). And \(I(1)=0\) is a *consequence*, not a separate axiom: \(g(0)=g(0+0)=2g(0)\Rightarrow g(0)=0\Rightarrow I(1)=0\).

### Why the constant in front doesn't matter

There is **nothing** forcing \(C=1\). The constant is real and free, it is a choice of *units*, exactly like meters vs feet:

| Constant / base | Unit |
| --- | --- |
| \(C=1/\ln 2\) (i.e. \({\log}_2\)) | bits |
| \(C=1\) (natural log) | nats |
| \(C=1/\ln 10\) | bans / dits (historical) |

!!! note "Why nobody justifies the base"
    Every decision and gradient is invariant to \(C\). In a tree, multiplying all entropies by \(C\) multiplies Information Gain by \(C\), the argmax is unchanged. In training, loss times a constant has gradient times the constant, absorbed by the learning rate. The base washes out of everything that matters. Base 2 for the clean "bits = questions" story; natural log for clean calculus.

!!! note "The honest logical structure"
    1. **Axiom:** information from independent events adds.
    2. **Plus** weakest regularity (continuity / monotonicity).
    3. **Theorem (Cauchy):** forces \(I(p)=-C\log p\), nothing else.
    4. **Convention:** pick \(C\) = pick your unit. Irrelevant downstream.
    5. Entropy = expected surprise = \(\sum_i p_i\,I(p_i)=-C\sum_i p_i\log p_i\).

## §4 Entropy: definition, shape, units

!!! note "Definition"
    $$H(S)=\mathrm{E}[\text{surprise}]=\sum_i p_i{\log}_2\frac{1}{p_i}=-\sum_i p_i{\log}_2 p_i$$
    Entropy is the *average surprise* = the *average information per outcome* = the average number of binary digits to transmit the source under optimal coding. All the same expectation of the same per-outcome number \(-{\log}_2 p_i\).

"Expected amount of surprise" is the *definition*, not an analogy. A low-entropy source (e.g. 90% one outcome) is cheap to transmit; a high-entropy uniform source costs the full \({\log}_2 C\) bits every time.

### The non-uniform insight

When outcomes are unequal, give common things short codes and rare things long codes (Morse: 'E' = one dot, 'Q' = four symbols). The *average* number of questions per outcome under optimal coding is exactly the entropy. Information of one outcome = \(-{\log}_2 p_i\); entropy = average over the distribution.

### Shape for the binary case (memorize)

For class proportion \(p\): \(\,H(p)=-p{\log}_2 p-(1-p){\log}_2(1-p)\). A symmetric dome: 0 at \(p=0\), peaks at **1.0** when \(p=0.5\), back to 0 at \(p=1\). Pure = no surprise = 0; 50/50 = max surprise = 1 bit.

<figure class="diagram diagram-dark" markdown="0">
<svg aria-label="Entropy and Gini curves for binary case" role="img" viewBox="0 0 560 300" xmlns="http://www.w3.org/2000/svg">
<rect fill="none" height="300" width="560" x="0" y="0"></rect>
<!-- axes -->
<line stroke="#2a3240" stroke-width="1.5" x1="60" x2="520" y1="250" y2="250"></line>
<line stroke="#2a3240" stroke-width="1.5" x1="60" x2="60" y1="250" y2="30"></line>
<!-- gridlines -->
<line stroke="#1c222e" stroke-dasharray="3 4" stroke-width="1" x1="60" x2="520" y1="60" y2="60"></line>
<line stroke="#1c222e" stroke-dasharray="3 4" stroke-width="1" x1="60" x2="520" y1="155" y2="155"></line>
<line stroke="#1c222e" stroke-dasharray="3 4" stroke-width="1" x1="290" x2="290" y1="250" y2="30"></line>
<!-- Entropy curve: peak 1.0 at p=0.5 -> y=60 ; p axis 60..520 -->
<path d="M60,250 Q175,60 290,60 Q405,60 520,250" fill="none" stroke="#e8b059" stroke-width="2.5"></path>
<!-- Gini curve: peak 0.5 at p=0.5 -> y=155 -->
<path d="M60,250 Q175,155 290,155 Q405,155 520,250" fill="none" stroke="#5fb3a3" stroke-width="2.5"></path>
<!-- labels -->
<text fill="#e8b059" font-family="monospace" font-size="13" x="430" y="48">Entropy (peak 1.0)</text>
<text fill="#5fb3a3" font-family="monospace" font-size="13" x="430" y="140">Gini (peak 0.5)</text>
<text fill="#9aa6b8" font-family="monospace" font-size="11" text-anchor="end" x="48" y="64">1.0</text>
<text fill="#9aa6b8" font-family="monospace" font-size="11" text-anchor="end" x="48" y="159">0.5</text>
<text fill="#9aa6b8" font-family="monospace" font-size="11" text-anchor="end" x="48" y="254">0</text>
<text fill="#9aa6b8" font-family="monospace" font-size="11" text-anchor="middle" x="60" y="272">0</text>
<text fill="#9aa6b8" font-family="monospace" font-size="11" text-anchor="middle" x="290" y="272">0.5</text>
<text fill="#9aa6b8" font-family="monospace" font-size="11" text-anchor="middle" x="520" y="272">1.0</text>
<text fill="#9aa6b8" font-family="Helvetica" font-size="12" text-anchor="middle" x="290" y="292">class proportion p</text>
</svg>
<figcaption>Both impurity curves bottom out at purity (p=0,1) and peak at the uniform split (p=0.5). Gini is a scaled near-twin of entropy, the reason they pick the same split about 99% of the time.</figcaption>
</figure>

### Units recap

**Nats** = same entropy via \(\ln\) (base \(e\)); 1 nat is about 1.44 bits. Used because \(\frac{d}{dx}\ln x=1/x\) has no stray \(\ln 2\), cleaner calculus, so ML losses default to natural log. **Bits** when the story is storage / transmission. Pure unit choice; rescales everything uniformly, changes no argmax or optimum.

## Interview questions

**Q1: What is the information content of an outcome, and why is it \(-\log p\)?**
It is the number of optimal yes/no questions needed to identify the outcome, where each question ideally halves the remaining possibilities, so going from N candidates to one takes \(\log_2 N\) halvings and an outcome of probability p carries \(-\log_2 p\) bits. This is forced, not chosen: requiring that independent surprises add, \(I(pq)=I(p)+I(q)\), plus the mildest regularity gives Cauchy's functional equation, whose only solutions are \(-C\log p\). The base just sets the unit, bits or nats.

**Q2: Why does the choice of logarithm base not matter?**
The base only changes the multiplicative constant C, which is a unit choice like meters versus feet. In a tree, scaling all entropies by C scales information gain by C, leaving the argmax unchanged; in training, scaling the loss scales the gradient by the same constant, which the learning rate absorbs. So nothing downstream, no decision and no optimum, depends on it.

**Q3: What is entropy, and what is its shape for two classes?**
Entropy is expected surprise, \(-\sum_i p_i\log_2 p_i\), equivalently the average number of bits to transmit the source under optimal coding. For the binary case it is a symmetric dome: zero at a pure node where p is 0 or 1, rising to a maximum of 1 bit at the 50/50 split. Pure means no surprise; maximally mixed means maximum surprise.

**Q4: A node has 8 positives and 2 negatives. Is its entropy closer to 0 or to 1?**
Closer to 0. Computing \(-0.8\log_2 0.8 - 0.2\log_2 0.2\) gives about 0.72 bits. The node is fairly pure since it is mostly positive, so it is less mixed than 50/50 and carries less than the 1-bit maximum surprise.
