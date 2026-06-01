# ANN Algorithms

Exact K-NN over `N` vectors is `O(N × d)` — fine at 10K, painful at 10M, impossible at 1B. **Approximate Nearest Neighbor** algorithms trade a sliver of recall for orders of magnitude in speed. Vector stores are just wrappers around these algorithms, so understanding the algorithms is how you tune the wrappers.

!!! tip "Rapid Recall"
    Three families: **graph** (HNSW), **partition** (IVF), **quantization** (PQ). HNSW is the 2026 default in-memory index — multi-layer navigable graph, greedy-walk top to bottom, beam search on layer 0. Sub-linear query, great recall, RAM-hungry; the defining limit is memory. IVF partitions vector space into `nlist` k-means cells and scans only the `nprobe` nearest cells per query; lighter on memory, friendlier to writes. IVF-PQ adds product quantization on the residual vectors — compresses each vector to a handful of bytes (e.g. 3 KB → 8 bytes), and the `Nd` term vanishes from memory. That single change is what makes billion-scale RAG fit in RAM. **Decision**: default HNSW; switch to IVF-PQ when RAM is binding or past ~500M vectors; LSH and KD-tree are obsolete for dense embeddings.

## The three families at a glance

| | HNSW | IVF | IVF-PQ |
|---|---|---|---|
| Family | Graph (navigable small world) | Partition (k-means cells) | Partition + quantization |
| Build | Slow, no training | k-means training | k-means + PQ codebook |
| Query | Fastest at high recall | `O(√N)`, tunable | Same as IVF |
| Memory | Heavy (vectors + graph links) | Moderate (vectors + centroids) | **Light** — vectors compressed away |
| Updates | Incremental OK; deletes degrade | Cheap inserts, easy deletes | Cheap, with periodic re-train |
| Knob | `ef_search` | `nprobe` | `nprobe` + code size |
| 2026 status | Dominant default | Useful when writes churn | What unlocks billion-scale RAM |

The right way to think about them: HNSW navigates a graph; IVF prunes by region; PQ compresses the vectors themselves. The frontier between them isn't quality — it's the binding constraint (latency, RAM, write churn).

## §5.1 HNSW (Hierarchical Navigable Small World)

HNSW is built from two ideas stacked on top of each other.

### Idea 1: small-world graphs, navigate by greedy hops

Like "six degrees of separation": most links are local, a few are long-range, and those long links let you cross the network fast. Each vector is a node linked to its nearby vectors plus a handful of longer links. To search, you start somewhere and **greedily walk toward the query** — move to whichever neighbor is closest to the query, repeat, stop when no neighbor is closer.

The flaw with a single flat graph: you get stuck in tiny local steps early on. You need big jumps first, small jumps near the end. That's where the second idea comes in.

### Idea 2: hierarchy (the skip-list trick)

<figure class="diagram diagram-light" markdown="0">
<svg viewbox="0 0 640 280" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace">
<text x="14" y="28" font-size="12" fill="#9a3412">Layer 2 — sparse, long jumps</text>
<circle cx="90" cy="50" r="7" fill="#fbeee6" stroke="#9a3412" stroke-width="1.5"/>
<circle cx="540" cy="50" r="7" fill="#fbeee6" stroke="#9a3412" stroke-width="1.5"/>
<line x1="97" y1="50" x2="533" y2="50" stroke="#9a3412" stroke-width="1" opacity="0.5"/>
<text x="90" y="74" font-size="10" fill="#4a443b" text-anchor="middle">entry</text>
<text x="14" y="120" font-size="12" fill="#a67c1a">Layer 1 — medium</text>
<circle cx="90" cy="142" r="6" fill="#f7efd8" stroke="#a67c1a" stroke-width="1.5"/>
<circle cx="260" cy="142" r="6" fill="#f7efd8" stroke="#a67c1a" stroke-width="1.5"/>
<circle cx="400" cy="142" r="6" fill="#f7efd8" stroke="#a67c1a" stroke-width="1.5"/>
<circle cx="540" cy="142" r="6" fill="#f7efd8" stroke="#a67c1a" stroke-width="1.5"/>
<line x1="96" y1="142" x2="254" y2="142" stroke="#a67c1a" stroke-width="1" opacity="0.5"/>
<line x1="266" y1="142" x2="394" y2="142" stroke="#a67c1a" stroke-width="1" opacity="0.5"/>
<line x1="406" y1="142" x2="534" y2="142" stroke="#a67c1a" stroke-width="1" opacity="0.5"/>
<text x="14" y="210" font-size="12" fill="#1d4e57">Layer 0 — all nodes, dense</text>
<circle cx="90" cy="232" r="5" fill="#eaf0f1" stroke="#1d4e57" stroke-width="1.5"/>
<circle cx="175" cy="232" r="5" fill="#eaf0f1" stroke="#1d4e57" stroke-width="1.5"/>
<circle cx="260" cy="232" r="5" fill="#eaf0f1" stroke="#1d4e57" stroke-width="1.5"/>
<circle cx="330" cy="232" r="5" fill="#eaf0f1" stroke="#1d4e57" stroke-width="1.5"/>
<circle cx="400" cy="232" r="5" fill="#eaf0f1" stroke="#1d4e57" stroke-width="1.5"/>
<circle cx="470" cy="232" r="6.5" fill="#5b6e2f" stroke="#3a4a1c" stroke-width="1.5"/>
<circle cx="540" cy="232" r="5" fill="#eaf0f1" stroke="#1d4e57" stroke-width="1.5"/>
<line x1="95" y1="232" x2="535" y2="232" stroke="#1d4e57" stroke-width="0.8" opacity="0.35"/>
<line x1="540" y1="57" x2="540" y2="135" stroke="#888" stroke-width="0.8" stroke-dasharray="3 3"/>
<line x1="400" y1="148" x2="400" y2="226" stroke="#888" stroke-width="0.8" stroke-dasharray="3 3"/>
<text x="470" y="258" font-size="10" fill="#5b6e2f" text-anchor="middle">query target</text>
</svg>
<figcaption>Enter at the top, hop far across the sparse layer, drop down, refine. Big jumps first (few nodes), small jumps last (dense). The green target sits on layer 0; the descent gets there in O(log N) hops.</figcaption>
</figure>

Layer 0 has **every** node; each layer up keeps ~1/M of the layer below (exponentially fewer). Top = sparse with long links; bottom = dense. **Search descends top-down:** greedily walk each layer toward the query, drop down a layer, repeat; do the thorough search on layer 0.

Mental model: top layers are an express highway to the right city; layer 0 is local streets to the exact address.

### The three hyperparameters that actually matter

```python
import hnswlib

index = hnswlib.Index(space='cosine', dim=768)
index.init_index(max_elements=N, M=16, ef_construction=200)
index.add_items(data, ids)
index.set_ef(50)        # ef_search — tune live, no rebuild
labels, dist = index.knn_query(query, k=10)
```

| Param | What it does | Trade |
|---|---|---|
| `M` | Links per node (graph degree). 12–16 typical, 32–64 for high-dim or high-recall workloads. | ↑ recall, ↑ RAM, ↑ build time. Memory ~linear in `M`. |
| `ef_construction` | Candidates considered while inserting. 100–200 typical. | ↑ graph quality, ↑ build time only (one-time cost). |
| `ef` (search) | Candidate list size per query. Must be ≥ k. | ↑ recall, ↑ latency. **Tuned live — this is the production dial.** |

Mental split: **`M` and `ef_construction` are build-time** (fix graph quality, pay once); **`ef` is query-time** (trade latency for recall on every request).

### Strengths, weaknesses, the ceiling

- **Strengths:** SOTA recall/latency in RAM; ~`O(log N)` search; query-time tunable; no training step (unlike IVF).
- **Weaknesses:** *memory hungry* — graph links cost RAM **on top of** the vectors (the defining limit); slow and expensive builds and inserts (each insert runs a search); deletes are awkward (tombstone + periodic rebuild); all in-memory (use DiskANN for on-disk).

!!! warning "Interview trap"
    "How does HNSW scale to a billion vectors?" — the honest answer is: **it doesn't, gracefully.** That's where IVF-PQ, DiskANN, or sharding come in. Knowing the RAM ceiling is the senior signal.

See [HNSW From Scratch](hnsw-from-scratch.md) for a full numpy implementation that walks every line of the algorithm.

## §5.2 IVF (Inverted File index)

The name is borrowed from text search: an inverted file maps `word → docs containing it`. IVF does the spatial version — partition the vector space into **regions**, map each `region → vectors in it`. At query time, find which region(s) the query lands in and scan *only those*.

### Yes — it's just k-means

You partition by running **k-means** with `nlist` clusters. Each centroid defines a Voronoi cell. The centroids *are* the inverted file's keys.

<figure class="diagram diagram-light" markdown="0">
<svg viewbox="0 0 560 260" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono, monospace">
<rect x="20" y="20" width="520" height="220" rx="10" fill="none" stroke="#d8d2c4"/>
<g fill="#eaf0f1" stroke="#1d4e57"><circle cx="110" cy="80" r="5"/><circle cx="140" cy="62" r="5"/><circle cx="95" cy="105" r="5"/></g>
<circle cx="135" cy="100" r="8" fill="#1d4e57"/><text x="135" y="128" font-size="9" fill="#4a443b" text-anchor="middle">A</text>
<g fill="#eef4ee" stroke="#5b6e2f"><circle cx="300" cy="72" r="5"/><circle cx="330" cy="90" r="5"/><circle cx="285" cy="100" r="5"/></g>
<circle cx="310" cy="92" r="8" fill="#5b6e2f"/><text x="310" y="120" font-size="9" fill="#4a443b" text-anchor="middle">B</text>
<g fill="#fbeee6" stroke="#9a3412"><circle cx="450" cy="180" r="5"/><circle cx="480" cy="196" r="5"/><circle cx="438" cy="205" r="5"/></g>
<circle cx="462" cy="195" r="8" fill="#9a3412"/><text x="462" y="223" font-size="9" fill="#4a443b" text-anchor="middle">C</text>
<g fill="#f7efd8" stroke="#a67c1a"><circle cx="130" cy="195" r="5"/><circle cx="160" cy="208" r="5"/></g>
<circle cx="148" cy="202" r="8" fill="#a67c1a"/><text x="148" y="230" font-size="9" fill="#4a443b" text-anchor="middle">D</text>
<circle cx="320" cy="160" r="7" fill="#1f1c17"/><text x="320" y="148" font-size="10" fill="#1f1c17" text-anchor="middle">query</text>
<line x1="320" y1="160" x2="312" y2="100" stroke="#5b6e2f" stroke-width="1.2" stroke-dasharray="4 3"/>
<line x1="320" y1="160" x2="455" y2="188" stroke="#9a3412" stroke-width="1.2" stroke-dasharray="4 3" opacity="0.6"/>
</svg>
<figcaption>nprobe=2: scan cells B and C (nearest centroids), skip A and D. The risk lives at the borders — a true neighbor just inside A's edge gets missed entirely. That's the edge problem.</figcaption>
</figure>

How k-means actually picks and moves centroids — worth knowing because interviewers ask:

- **Init (k-means++):** first centroid random, then each subsequent one chosen with probability proportional to *squared distance from the nearest existing centroid* — so points far from existing centroids are likelier to be picked. This *spreads them out* (the "diverge" property), avoiding clumps.
- **Lloyd's loop:** (1) assign every point to nearest centroid; (2) move each centroid to the mean of its points; repeat until stable.

Centroids aren't engineered to be equidistant — they settle into the data's density, and k-means++ just gives a good spread-out start.

### Parameters

- `nlist` — number of clusters. Rule of thumb ≈ `√N`. More cells = less to scan per query, but more centroids to compare against upfront.
- `nprobe` — cells scanned per query. The **recall vs speed dial**, IVF's analogue of HNSW's `ef`. Tunable live. `nprobe=1` is fastest but misses border neighbors (the edge problem above).

### IVF vs HNSW at a glance

| | HNSW | IVF (flat) |
|---|---|---|
| Structure | Layered proximity graph | k-means cells + inverted lists |
| Build | Slow, no training | Needs k-means training |
| Query | Fastest at high recall | Fast, slightly behind |
| Memory | **Heavy** (graph links) | **Light** (centroids + assignments) |
| Inserts / deletes | Expensive / awkward | Cheap / easy |
| Dial | `ef` | `nprobe` |

**HNSW** wins on recall-per-latency *when it fits in RAM* (most production RAG, up to tens of millions of vectors). **IVF (+ PQ)** wins when RAM is the binding constraint and N is enormous (hundreds of millions to billions), or when write/delete churn is high. They're different points on the memory-vs-quality frontier — which is why FAISS lets you combine them.

## §5.3 Product Quantization (PQ)

**The problem PQ solves:** vectors are big. A 768-dim float32 vector is 3,072 bytes; a billion of them is 3 TB and won't fit in RAM. PQ compresses each vector into a tiny code (e.g. 8 bytes).

### How PQ encodes — the "product" trick

1. **Split** each vector into `m` sub-vectors (768-dim with m=8 → 8 chunks of 96 dims each).
2. **Quantize each chunk** with its own k-means (256 centroids per subspace fits in 1 byte).
3. **Vector → m bytes** (the centroid ID per subspace). 3,072 bytes → 8 bytes, ~380x compression.

"Product" because the full approximation is the *Cartesian product* of per-subspace codebooks: `m=8 subspaces × 256 options = 256^8 ≈ 18 quintillion` representable vectors, from only `8 × 256` stored centroids. Combinatorial power, tiny storage.

### Search without decompressing — ADC

**Asymmetric Distance Computation:** the query stays full-precision; database vectors are quantized (hence "asymmetric"). At query time:

1. Split the query into the same `m` sub-vectors.
2. Build a lookup table: for each subspace, distance from the query's sub-vector to all 256 centroids → an `8 × 256` table. Cost: 2,048 small distances, **computed once per query**.
3. Score each database vector by its code: `table[0][42] + table[1][7] + ...` — `m` lookups plus `m` adds. No per-vector distance math at all.

Squared L2 is separable across dimension chunks, so the total distance = sum of per-subspace distances = sum of `m` table lookups. One scoring: from 768 multiply-adds down to 8 lookups + 8 adds (~50–100x faster), *plus* ~380x smaller storage.

### Why PQ pairs with IVF (not HNSW)

This is the canonical interview question on IVF-PQ.

- **IVF and PQ are complementary.** IVF answers "*which* vectors to look at" (prune by cell); PQ answers "how to store and compare each cheaply" (compress). Alone, IVF still stores fat vectors; alone, PQ must scan all `N` codes. Together (**IVF-PQ**) you prune to a few cells *and* every vector is 8 bytes → billion-scale in RAM.
- **IVF-PQ stores the PQ code of the residual** (vector minus its cell centroid). Residuals are smaller and more similar within a cell, so PQ quantizes them more accurately. IVF hands PQ a better-conditioned input.
- **Why not PQ on HNSW?** HNSW's greedy graph navigation needs *accurate* distances to pick the next hop; PQ's approximate distances make worse hops and recall degrades more than for IVF (where PQ only ranks an already-pruned shortlist). Also, HNSW's real memory hog is the *links*, which PQ doesn't shrink. (An `HNSWPQ` index exists but the value is much weaker.)
- **Why not PQ alone?** It works (called "flat PQ") but it scans all `N` codes — fine up to a few million, but past that IVF's pruning earns its keep.

The cost of PQ is recall — it's lossy. Standard fix: **rerank**. IVF-PQ gets a coarse top-100 fast, then re-scores those against the original full-precision vectors for an accurate top-10. Same coarse-then-exact philosophy as parent-child chunking.

### Quick numbers

100M vectors × 1024 dims × float32 = 400 GB. IVF-PQ with `m=16` 8-bit codes = 16 bytes/vec = **1.6 GB**. That's 250x reduction for roughly 5% recall loss.

### IVF-PQ in FAISS

```python
import faiss
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index.train(train_vec)   # learns k-means centroids AND PQ codebooks
index.add(db_vec)        # assigns to cells, stores PQ codes of residuals
index.nprobe = 16
D, I = index.search(query, k=10)
```

## §5.4 LSH and KD-tree — the casualties of high dimensions

### LSH (Locality-Sensitive Hashing)

Normal hashing scatters similar inputs apart; LSH does the opposite — hash functions where **similar vectors collide on purpose**. Hash the query, look only in its bucket.

**Random hyperplane hashing** (for cosine): a random hyperplane gives each vector 1 bit (which side it's on). `b` hyperplanes give a `b`-bit signature; matching signatures = same bucket. Two vectors agree on a random plane with probability proportional to their angular closeness. Use `L` independent tables; a vector is a candidate if it collides in *any* table.

- Dials: `b` (bits per table — stricter buckets) and `L` (tables — more recall, more memory).
- Strengths: `O(1)` hashing, simple, distributable, trivial streaming inserts, strong theoretical guarantees.
- Weakness: needs many tables for HNSW-level recall → Pareto-dominated for dense embeddings.
- **Still alive for:** extreme-scale distributed/streaming systems; and **MinHash for Jaccard similarity on sets** (dedup, near-duplicate detection — used to dedupe LLM training corpora).

### KD-tree

Binary search generalized to space: recursively median-split, cycling axes (x, then y, then z…). Search descends to the query's leaf, then backtracks, pruning branches whose box is too far to contain anything closer.

- **Strength:** *exact* nearest neighbor, and genuinely fast in **low dimensions (≤ ~10–20)** — geo, point clouds, small tabular k-NN.
- **Fatal weakness — curse of dimensionality:** as dimensions grow, almost everything is roughly equidistant, the splitting planes stop separating near from far, pruning fails, and search degrades to ~`O(N)` (worse than brute force once you count overhead). At 768/1536 dims, useless.

The unifying thread: KD-trees fail because exact spatial-partitioning pruning dies in high dimensions. LSH and the graph/cluster methods (HNSW, IVF) *survive* precisely because they're **approximate** and don't rely on clean geometric partitioning. So the answer to "why not a KD-tree for embeddings?" is *curse of dimensionality* — pruning becomes ineffective and the structure degenerates to a linear scan.

### Annoy and ScaNN — where they fit

Two more names that come up in interviews:

- **Annoy** (Spotify) is the same tree family as KD-tree, but uses **random hyperplanes** instead of axis-aligned splits, builds a *forest* of trees instead of one, and is mmap-friendly so it loads instantly and can be shared across processes. Redesigned to survive high-D embeddings, but still lower recall than HNSW. Mostly legacy in 2026.
- **ScaNN** (Google) *uses* the PQ family but optimizes for **ranking accuracy** (anisotropic quantization — preserves the components that matter for inner-product ranking, not raw reconstruction) rather than reconstruction accuracy. Then wraps it in partition + rerank. Think "smarter PQ for ranking," not plain PQ. SOTA at scale, especially for maximum-inner-product search; needs tuning.

So the broader family map:

| | Graph | Tree | Quantization |
|---|---|---|---|
| Modern survivor | HNSW | Annoy (legacy) | PQ, ScaNN |
| Recall vs latency | Best overall | Lower | SOTA at scale |
| Memory | High | Low-ish (mmap) | Low |
| Updates | Incremental OK | Immutable (rebuild) | Limited/batch |

In production at real scale, the best systems *compose*: graph navigation for coarse search, quantization for compression. Not one pure algorithm.

## §5.5 Complexity table, decoded

Notation: `N` = vectors, `d` = dims, `M` = HNSW links per node, `nlist` = IVF clusters, `nprobe` = cells scanned.

| Algorithm | Build | Query | Memory |
|---|---|---|---|
| Flat (exact) | `O(1)` | `O(Nd)` | `O(Nd)` |
| HNSW | `O(N log N · M)` | `O(log N · ef_search)` | `O(Nd + `**`NM`**`)` |
| IVF | `O(N · kmeans_iters)` | `O(nprobe · N/nlist · d)` | `O(Nd + nlist·d)` |
| IVF-PQ | + PQ train | same as IVF | `O(`**`N · code_bytes`**`)` |

What's actually going on:

- **Flat** query `O(Nd)`: compare to every vector, each costing `d`. Memory `O(Nd)`: store every vector.
- **HNSW** query `O(log N · ef_search)`: about `log N` hops, examining `ef_search` candidates each. Memory `+ NM` is the graph links (`N` nodes × `M` links per node) — HNSW's memory tax.
- **IVF** query: scan `nprobe` cells, each holding roughly `N/nlist` vectors, each costing `d`. Memory `+ nlist·d` is just the centroids (tiny vs HNSW's `NM`).
- **IVF-PQ** memory `N · code_bytes`: the `Nd` term *vanishes* — no full vectors stored, only compact codes. **This is the billion-scale-in-RAM win.**

The two cells to memorize: HNSW memory has **+ NM** (heavy graph links); IVF-PQ is **N · code_bytes** with *no Nd* (vectors compressed away). That single contrast is the entire HNSW-vs-IVFPQ story: speed and recall vs RAM.

## §5.6 FAISS recipes

```python
import faiss

# HNSW
idx = faiss.IndexHNSWFlat(1024, 32)
idx.hnsw.efConstruction = 200
idx.hnsw.efSearch = 100
idx.add(xb)
D, I = idx.search(xq, 10)

# IVF-PQ
quantizer = faiss.IndexFlatL2(1024)
idx = faiss.IndexIVFPQ(quantizer, 1024, 4096, 16, 8)  # nlist=4096, m=16, 8 bits per code
idx.train(xb[:100000])
idx.add(xb)
idx.nprobe = 32
```

## §5.7 Decision rule

HNSW by default. Escalate to IVF-PQ when RAM is the binding constraint or you're past roughly 500M vectors. Ship Annoy only for static, read-only, legacy-compatible workloads. LSH only for genuinely streaming/distributed scale or for set-based MinHash dedup. KD-tree never, for any modern embedding workload.

## Interview Questions

**Q5: Your ANN recall is 70% but eval expects 95%. Diagnose.**

(1) Pre vs post filter? Post-filter + K=10 means retrieve 10 then filter, often leaves you with <10 results. (2) `efSearch` (HNSW) or `nprobe` (IVF) too low. (3) Vectors not normalized, metric mismatch. (4) Stale index after many deletes — HNSW soft-deletes degrade, reindex.

**Q6: Explain HNSW vs IVF in one minute.**

HNSW: hierarchical navigable graph, greedy-walk from a top-level entry down to the query's neighbors. Sub-linear query, great recall, high RAM. IVF: k-means partitions into `nlist` Voronoi cells, search the top `nprobe` cells only. `O(√N)`, lower RAM, tunable per query. HNSW for quality when it fits in RAM; IVF (and IVF-PQ) for scale and write churn.

**Q7: How does HNSW scale to a billion vectors?**

It doesn't, gracefully — that's the senior signal. HNSW's defining limit is RAM: every vector plus `M` graph links per node has to live in memory. At a billion vectors with `d=1024` float32 and `M=32`, you're looking at ~4 TB for vectors and another ~256 GB for links. Past ~500M you escalate to IVF-PQ (codes compress the vectors away), DiskANN (graph on SSD), or shard across nodes.

**Q8: Why does PQ pair with IVF instead of HNSW?**

IVF prunes vectors to a shortlist (the right cells); PQ then scores the shortlist cheaply from compressed codes — approximate distances are fine for ranking an already-narrow candidate set, and IVF-PQ stores the *residual* (vector minus cell centroid), which quantizes more accurately. HNSW's greedy graph navigation, by contrast, *needs* accurate distances at every hop or it routes wrong; PQ's approximation degrades recall much more there. Also, HNSW's real RAM hog is the links, which PQ wouldn't compress anyway.
