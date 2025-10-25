# ============================================================
#  Catan 7-Hex QUBO Generator
#  Generates a valid QUBO file (qubo_7hex.json)
#  for use with Qiskit QAOA or D-Wave solvers.
# ============================================================

import json
from collections import defaultdict
from itertools import combinations

# -------------------- CONFIGURATION --------------------
# Example token numbers (center + 6 surrounding hexes)
# You can replace these with your real board setup
hex_tokens = {0: 8, 1: 6, 2: 5, 3: 9, 4: 10, 5: 3, 6: 4}

# Dice roll probabilities
dice_p = {
    2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36, 7: 6/36,
    8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
}

# Desired number of settlements
m = 2

# Number of candidate vertices (simplified small example)
# The actual 7-hex cluster has ~24 vertices; this keeps it manageable
N = 12

# For simplicity, create synthetic vertex values from dice rolls
v = {i: sum(dice_p[t] for t in hex_tokens.values()) / len(hex_tokens)
     for i in range(N)}

# Example adjacency pairs (no two adjacent settlements)
# For demonstration, connect vertices in a ring and a few cross-links
adj_pairs = set()
for i in range(N):
    adj_pairs.add((i, (i + 1) % N))  # ring adjacency
for i in range(0, N, 3):
    if i + 2 < N:
        adj_pairs.add((i, i + 2))    # extra link

# -------------------- BUILD QUBO --------------------
Vsum = sum(v.values())
A = 1.5 * Vsum  # adjacency penalty
B = A            # cardinality penalty

Q = defaultdict(float)

# Objective: maximize sum(v_i * x_i) → minimize -sum(v_i * x_i)
for i, vi in v.items():
    Q[(i, i)] += -vi

# Adjacency penalties
for (i, j) in adj_pairs:
    Q[(i, j)] += A

# Settlement count penalty (enforce sum x_i = m)
for i in range(N):
    Q[(i, i)] += B * (1 - 2 * m)
for i, j in combinations(range(N), 2):
    Q[(i, j)] += 2 * B

# -------------------- SAVE TO JSON --------------------
qubo_serializable = {f"{i},{j}": float(c) for (i, j), c in Q.items()}
data = {"Q": qubo_serializable, "A": A, "B": B, "m": m, "v": v}

path = "qubo_7hex.json"
with open(path, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ QUBO saved successfully to {path}")
print(f"   Variables: {N}, Adjacency pairs: {len(adj_pairs)}")
