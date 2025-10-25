from qiskit_optimization import QuadraticProgram
import json

QUBO_JSON_PATH = r"C:\Users\Serban\Desktop\project\qubo_7hex.json"

with open(QUBO_JSON_PATH) as f:
    data = json.load(f)

# Parse Q_raw
Q_raw = {tuple(map(int, k.split(","))): float(v) for k, v in data["Q"].items()}
qp = QuadraticProgram("debug")

n = max(max(i, j) for (i, j) in Q_raw) + 1
for i in range(n):
    qp.binary_var(name=f"x{i}")

linear = {f"x{i}": 0.0 for i in range(n)}
quadratic = {}

for (i, j), coef in Q_raw.items():
    if i == j:
        linear[f"x{i}"] += coef
    else:
        quadratic.setdefault(f"x{i}", {})[f"x{j}"] = coef

# 🔍  Inspect everything before minimize
print("Linear keys:", list(linear.keys()))
print("Quadratic outer keys:", list(quadratic.keys())[:10])

# Print any suspicious entries
for key in linear.keys():
    if key == "x" or not key.startswith("x"):
        print("❌ BAD linear key:", key)

for outer, sub in quadratic.items():
    if outer == "x" or not outer.startswith("x"):
        print("❌ BAD outer quadratic key:", outer)
    for inner in sub.keys():
        if inner == "x" or not inner.startswith("x"):
            print("❌ BAD inner quadratic key:", outer, "->", inner)

print("\nNow calling minimize...\n")
qp.minimize(linear=linear, quadratic=quadratic)
print("✅ minimize() succeeded.")
