# ============================================================
#  Catan 7-Hex QUBO + QAOA Solver (Qiskit 2.2.x, Opt 0.7.0 safe)
# ============================================================

from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import json
from itertools import combinations

# ---------- 1) Load QUBO ----------
QUBO_JSON_PATH = r"C:\Users\Serban\Desktop\project\qubo_7hex.json"

with open(QUBO_JSON_PATH, "r") as f:
    data = json.load(f)

print("🔍 Raw Q keys sample:", list(data["Q"].keys())[:10])

# ---------- 1b) Parse and sanitize QUBO ----------
Q_raw = {}
for k, v in data["Q"].items():
    cleaned = "".join(c for c in k if c.isdigit() or c == ",")
    parts = [p for p in cleaned.split(",") if p != ""]
    try:
        if len(parts) == 2:
            i, j = map(int, parts)
            Q_raw[(i, j)] = float(v)
        elif len(parts) == 1:
            i = int(parts[0])
            Q_raw[(i, i)] = float(v)
    except Exception:
        print("⚠️ Skipping malformed key:", k)

print(f"✅ Loaded {len(Q_raw)} valid QUBO entries\n")

A, B, m, vvec = data["A"], data["B"], data["m"], data["v"]

# ---------- 2) Build QuadraticProgram safely ----------
qp = QuadraticProgram("catan_7hex")
n = max(max(i, j) for (i, j) in Q_raw.keys()) + 1
for i in range(n):
    qp.binary_var(f"x{i}")

from qiskit_optimization.problems.quadratic_objective import QuadraticObjective
from qiskit_optimization.problems.linear_expression import LinearExpression
from qiskit_optimization.problems.quadratic_expression import QuadraticExpression

# ========== BUILD AND CLEAN DICTIONARIES ==========
# Helper function to validate variable names
def is_valid_var(name):
    return (isinstance(name, str) and 
            len(name) >= 2 and 
            name.startswith('x') and 
            name[1:].isdigit())

# Build raw dictionaries first
linear_dict_raw = {}
quad_dict_raw = {}

for (i, j), coef in Q_raw.items():
    # Validate indices
    if not (isinstance(i, int) and isinstance(j, int) and 0 <= i < n and 0 <= j < n):
        print(f"⚠️ Skipping invalid indices: ({i}, {j})")
        continue
    
    if i == j:
        # Diagonal term -> linear
        var_name = f"x{i}"
        linear_dict_raw[var_name] = linear_dict_raw.get(var_name, 0.0) + coef
    else:
        # Off-diagonal -> quadratic (always store with smaller index first)
        idx_min, idx_max = sorted([i, j])
        var1 = f"x{idx_min}"
        var2 = f"x{idx_max}"
        
        if var1 not in quad_dict_raw:
            quad_dict_raw[var1] = {}
        quad_dict_raw[var1][var2] = quad_dict_raw[var1].get(var2, 0.0) + coef

# Now clean the dictionaries
linear_dict = {k: v for k, v in linear_dict_raw.items() if is_valid_var(k)}

quad_dict = {}
for outer_key, inner_dict in quad_dict_raw.items():
    if not is_valid_var(outer_key):
        print(f"⚠️ Filtered invalid outer key: '{outer_key}'")
        continue
    
    clean_inner = {k: v for k, v in inner_dict.items() if is_valid_var(k)}
    if clean_inner:  # Only add if there are valid inner entries
        quad_dict[outer_key] = clean_inner

# Report what we filtered
filtered_linear = len(linear_dict_raw) - len(linear_dict)
filtered_quad = sum(len(v) for v in quad_dict_raw.values()) - sum(len(v) for v in quad_dict.values())

if filtered_linear > 0:
    print(f"⚠️ Filtered {filtered_linear} invalid linear terms")
if filtered_quad > 0:
    print(f"⚠️ Filtered {filtered_quad} invalid quadratic terms")

print(f"✅ Valid linear terms: {len(linear_dict)}")
print(f"✅ Valid quadratic terms: {sum(len(v) for v in quad_dict.values())}")

# Debug: Print the keys to see what we're passing
print("\n🔍 DEBUG - Linear dict keys:", sorted(linear_dict.keys())[:5], "...")
print("🔍 DEBUG - Quad dict outer keys:", sorted(quad_dict.keys())[:5], "...")
for key in list(quad_dict.keys())[:2]:
    print(f"🔍 DEBUG - Quad dict['{key}'] inner keys:", sorted(quad_dict[key].keys())[:5], "...")

# Create expressions with clean dictionaries
lin_expr = LinearExpression(qp, linear_dict)
print("✅ LinearExpression created successfully")

# CRITICAL FIX: Validate that all variables exist in qp before creating QuadraticExpression
qp_var_names = {var.name for var in qp.variables}
print(f"🔍 Variables registered in qp: {sorted(qp_var_names)}")

# Final validation: ensure all keys exist as actual variables in qp
quad_dict_validated = {}
for outer_key, inner_dict in quad_dict.items():
    if outer_key not in qp_var_names:
        print(f"❌ ERROR: '{outer_key}' not in qp variables!")
        continue
    
    inner_validated = {}
    for inner_key, val in inner_dict.items():
        if inner_key not in qp_var_names:
            print(f"❌ ERROR: '{inner_key}' not in qp variables!")
            continue
        inner_validated[inner_key] = val
    
    if inner_validated:
        quad_dict_validated[outer_key] = inner_validated

print(f"🔍 Validated quad_dict has {len(quad_dict_validated)} outer keys")

quad_expr = QuadraticExpression(qp, quad_dict_validated)
print("✅ QuadraticExpression created successfully")

qp._objective = QuadraticObjective(
    quadratic_program=qp,
    constant=0.0,
    linear=lin_expr,
    quadratic=quad_expr,
    sense=QuadraticObjective.Sense.MINIMIZE,
)

print(f"✅ QuadraticProgram built with {n} binary vars\n")
# ========== END OF BUILD AND CLEAN SECTION ==========

# ---------- 3) Classical baseline ----------
adj_pairs = set()
threshold = 0.5 * A
for (i, j), coef in Q_raw.items():
    if i < j and coef > threshold:
        adj_pairs.add((i, j))

def feasible(sel):
    for a, b in adj_pairs:
        if a in sel and b in sel:
            return False
    if m is not None and len(sel) != m:
        return False
    return True

best_val = -1e9
best_sets = []
indices = list(range(n))
sizes = [m] if m is not None else range(n + 1)
for k in sizes:
    for comb in combinations(indices, k):
        if not feasible(set(comb)):
            continue
        val = sum(vvec[str(i)] for i in comb)
        if val > best_val + 1e-12:
            best_val = val
            best_sets = [comb]
        elif abs(val - best_val) < 1e-12:
            best_sets.append(comb)

print("Classical exact best expected income:", best_val)
print("Best classical placements:", best_sets, "\n")

# ---------- 4) QAOA ----------
sampler = Sampler()
optimizer = COBYLA(maxiter=200)
qaoa = QAOA(sampler=sampler, reps=2, optimizer=optimizer)
meo = MinimumEigenOptimizer(qaoa)

# ---------- 5) Solve ----------
print("🚀 Running QAOA...")
result = meo.solve(qp)

print("🔹 QAOA result:")
print("Status:", result.status)
print("Objective value:", result.fval)
print("Solution vector:", result.x)

selected = [i for i, v in enumerate(result.x) if v == 1]
print("Selected vertices by QAOA:", selected)

qaoa_expected_income = sum(vvec[str(i)] for i in selected)
print("Expected income for QAOA solution:", qaoa_expected_income, "\n")

print("Classical best expected income:", best_val)
print("Classical best sets:", best_sets)
