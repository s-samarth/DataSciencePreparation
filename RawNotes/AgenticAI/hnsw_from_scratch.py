"""
HNSW from scratch — a teaching implementation.

Goal: make every step of the paper readable. This is NOT optimized.
A real library (hnswlib, FAISS) is 100x faster and handles edge cases.
But this version mirrors the actual algorithm so you can SEE it work.

Two production details deliberately simplified (flagged inline):
  1. Neighbor selection: we keep the M closest. The paper uses a smarter
     "heuristic" that also keeps diverse neighbors so the graph stays navigable.
  2. We don't prune a node's neighbor list when it gets over-connected.
"""

import numpy as np
import heapq
from collections import defaultdict


def distance(a, b):
    # Squared L2. Monotonic with L2, and cheaper (no sqrt). Fine for ranking.
    return np.sum((a - b) ** 2)


class HNSW:
    def __init__(self, M=16, ef_construction=200, ml=None, seed=42):
        self.M = M                          # links per node per layer
        self.M_max0 = 2 * M                 # layer 0 allows more links (denser bottom)
        self.ef_construction = ef_construction
        # ml controls how tall the hierarchy grows. 1/ln(M) is the paper's value.
        self.ml = ml if ml is not None else 1.0 / np.log(M)
        self.rng = np.random.default_rng(seed)

        self.vectors = {}                   # node_id -> vector
        self.graph = defaultdict(lambda: defaultdict(set))  # layer -> node -> {neighbors}
        self.entry_point = None             # node id of the top-of-hierarchy entry
        self.top_layer = -1

    def _random_level(self):
        # Geometric distribution: most nodes get level 0, few get high levels.
        # This is exactly the skip-list layer-assignment trick.
        return int(-np.log(self.rng.random()) * self.ml)

    def _search_layer(self, query, entry_points, ef, layer):
        """Greedy best-first search WITHIN one layer.
        Returns the ef closest nodes found, as a list of (dist, node_id)."""
        visited = set(entry_points)
        # candidates: min-heap by distance (closest to expand next)
        # results:    max-heap by distance (so we can pop the farthest to cap at ef)
        candidates = []
        results = []
        for ep in entry_points:
            d = distance(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))   # negate for max-heap behavior

        while candidates:
            dist_c, c = heapq.heappop(candidates)   # closest unexpanded candidate
            farthest_result = -results[0][0]        # current worst in results
            # If the closest remaining candidate is farther than our worst kept
            # result, no point continuing — greedy stop condition.
            if dist_c > farthest_result:
                break
            for neighbor in self.graph[layer][c]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = distance(query, self.vectors[neighbor])
                farthest_result = -results[0][0]
                if d < farthest_result or len(results) < ef:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)      # drop the farthest, keep ef best
        # Return as (dist, id), closest first
        return sorted([(-nd, n) for nd, n in results])

    def insert(self, node_id, vector):
        self.vectors[node_id] = vector
        level = self._random_level()

        # First node ever: it becomes the entry point and we're done.
        if self.entry_point is None:
            self.entry_point = node_id
            self.top_layer = level
            for lyr in range(level + 1):
                self.graph[lyr][node_id] = set()
            return

        ep = [self.entry_point]
        # Phase 1: from the top of the hierarchy down to the node's own top
        # layer, just GREEDILY descend (ef=1) to find a good entry point.
        for lyr in range(self.top_layer, level, -1):
            ep = [self._search_layer(vector, ep, ef=1, layer=lyr)[0][1]]

        # Phase 2: from the node's top layer down to 0, do a wide search
        # (ef_construction) and connect to the M best neighbors at each layer.
        for lyr in range(min(level, self.top_layer), -1, -1):
            found = self._search_layer(vector, ep, self.ef_construction, lyr)
            # --- SIMPLIFICATION 1: keep M closest. Paper uses select_neighbors_heuristic. ---
            M_layer = self.M_max0 if lyr == 0 else self.M
            neighbors = [n for _, n in found[:M_layer]]

            self.graph[lyr][node_id] = set(neighbors)
            for nbr in neighbors:
                self.graph[lyr][nbr].add(node_id)   # bidirectional link
                # --- SIMPLIFICATION 2: real HNSW prunes nbr's list if it now
                #     exceeds M_max. We let it grow. Fine for small demos. ---
            ep = [n for _, n in found]              # carry candidates down a layer

        # If this node is taller than the current hierarchy, it's the new entry point.
        if level > self.top_layer:
            self.top_layer = level
            self.entry_point = node_id

    def search(self, query, k=5, ef=50):
        if self.entry_point is None:
            return []
        ep = [self.entry_point]
        # Descend the express layers greedily (ef=1) to get near the target.
        for lyr in range(self.top_layer, 0, -1):
            ep = [self._search_layer(query, ep, ef=1, layer=lyr)[0][1]]
        # Thorough search on the dense bottom layer.
        found = self._search_layer(query, ep, ef=max(ef, k), layer=0)
        return found[:k]   # list of (dist, node_id), closest first


if __name__ == "__main__":
    # ---- Sanity check: does it find true nearest neighbors? ----
    rng = np.random.default_rng(0)
    N, dim = 2000, 32
    data = rng.random((N, dim)).astype(np.float32)

    index = HNSW(M=16, ef_construction=200)
    for i in range(N):
        index.insert(i, data[i])

    # Pick a random query, compare HNSW top-5 against brute-force top-5.
    q = rng.random(dim).astype(np.float32)

    approx = index.search(q, k=5, ef=50)
    approx_ids = [nid for _, nid in approx]

    brute = sorted(range(N), key=lambda i: distance(q, data[i]))[:5]

    print("HNSW  top-5:", approx_ids)
    print("Brute top-5:", brute)
    overlap = len(set(approx_ids) & set(brute))
    print(f"Recall@5: {overlap}/5 = {overlap/5:.0%}")

    # Measure recall over many queries to show it's reliably high.
    hits = 0
    Q = 200
    for _ in range(Q):
        q = rng.random(dim).astype(np.float32)
        a = {nid for _, nid in index.search(q, k=5, ef=50)}
        b = set(sorted(range(N), key=lambda i: distance(q, data[i]))[:5])
        hits += len(a & b)
    print(f"\nMean Recall@5 over {Q} queries: {hits/(Q*5):.1%}")
