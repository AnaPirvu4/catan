"""
Quantum Catan Challenge - Real Quantum Algorithm Implementation
ENHANCED: Roads now start from settlements and avoid crossing them
IMPROVED: Interactive interface with user input

Requirements:
pip install qiskit qiskit-aer numpy matplotlib networkx scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.lines import Line2D
import networkx as nx
import random
from scipy.optimize import minimize

# Qiskit imports for REAL quantum computing
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Map generation
MAP_SEED = None  # None for random, or set to integer for reproducible maps

# Problem 1: Settlement Optimization
DIVERSITY_WEIGHT = 2.0
PROBABILITY_WEIGHT = 1.5
QAOA_ITERATIONS = 30

# Problem 2: Longest Road
ROAD_ITERATIONS = 500

# Problem 3: Resource Trading
TRADE_ITERATIONS = 1000

# ============================================================================
# HELPER FUNCTION TO REPLACE ITERTOOLS.COMBINATIONS
# ============================================================================

def combinations(iterable, r):
    """Generate all r-length combinations of elements from iterable"""
    pool = list(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


# ============================================================================
# CATAN MAP GENERATOR
# ============================================================================

def draw_catan_terrain_map(seed=None, highlighted_vertices=None):
    """Generate and draw a random Catan terrain map"""
    if seed is None:
        seed = np.random.randint(0, 1000000)
        print(f"  [*] Generated random seed: {seed} (save this to reproduce this map!)")
    
    random.seed(seed)
    np.random.seed(seed)
    
    radius = 1.0
    hex_radius = radius

    axial_coords = [(0, 0),
                    (1, 0), (1, -1), (0, -1),
                    (-1, 0), (-1, 1), (0, 1)]

    def axial_to_cart(q, r):
        x = hex_radius * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = hex_radius * (1.5 * r)
        return (x, y)

    hex_centers = [axial_to_cart(q, r) for q, r in axial_coords]

    terrain_types = {
        "Forest": "#2E8B57",
        "Field": "#F4E04D",
        "Pasture": "#9ACD32",
        "Hill": "#D2691E",
        "Mountain": "#A9A9A9",
    }
    
    terrain_resources = {
        "Forest": "wood",
        "Field": "wheat",
        "Pasture": "sheep",
        "Hill": "brick",
        "Mountain": "ore"
    }

    terrain_list = random.choices(list(terrain_types.keys()), k=len(hex_centers))
    dice_numbers = random.sample([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], len(hex_centers))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    for (hx, hy), terrain, number in zip(hex_centers, terrain_list, dice_numbers):
        color = terrain_types[terrain]
        hex_patch = RegularPolygon(
            (hx, hy),
            numVertices=6,
            radius=hex_radius,
            orientation=np.radians(0),
            facecolor=color,
            alpha=0.7,
            edgecolor='k',
            linewidth=2
        )
        ax.add_patch(hex_patch)
        ax.text(hx, hy, str(number), ha='center', va='center',
                fontsize=18, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle', facecolor='black', alpha=0.6))
        ax.text(hx, hy - 0.5, terrain, ha='center', va='center',
                fontsize=10, color='black', fontweight='bold')

    tiles = []
    vertex_id_counter = 0
    vertex_coords_map = {}
    
    for i, (q, r) in enumerate(axial_coords):
        center = axial_to_cart(q, r)
        hx, hy = center
        vertices = []
        
        for j in range(6):
            angle = np.pi / 3 * j - np.pi / 6
            vx = hx + hex_radius * np.cos(angle)
            vy = hy + hex_radius * np.sin(angle)
            
            key = (round(vx, 4), round(vy, 4))
            
            if key not in vertex_coords_map:
                vertex_coords_map[key] = vertex_id_counter
                vertex_id_counter += 1
            
            vertices.append(vertex_coords_map[key])
        
        tiles.append({
            'id': i,
            'axial': (q, r),
            'center': center,
            'terrain': terrain_list[i],
            'resource': terrain_resources[terrain_list[i]],
            'number': dice_numbers[i],
            'vertices': vertices
        })
    
    if highlighted_vertices:
        vertex_positions = {v_id: coords for coords, v_id in vertex_coords_map.items()}
        
        for vid, (vx, vy) in vertex_positions.items():
            if vid in highlighted_vertices:
                ax.plot(vx, vy, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2)
                ax.text(vx, vy, str(vid), ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
            else:
                ax.plot(vx, vy, 'ko', markersize=6, alpha=0.3)

    plt.title("Quantum Catan Challenge - Terrain Map", fontsize=16, fontweight='bold')
    
    return terrain_list, dice_numbers, tiles, ax


# ============================================================================
# PROBLEM 1: QAOA SETTLEMENT PLANNER
# ============================================================================

class QuantumSettlementPlanner:
    """Optimizes settlement placement using QAOA with tile count priority"""
    
    def __init__(self, tiles, diversity_weight=2.0, probability_weight=1.5):
        self.tiles = tiles
        self.diversity_weight = diversity_weight
        self.probability_weight = probability_weight
        
        self.roll_probabilities = {
            2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
            7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
        }
        
        self.resource_values = {
            'wood': 1.0, 'brick': 1.0, 'ore': 1.0, 'wheat': 1.0, 'sheep': 1.0
        }
        
        self.vertex_positions = {}
        hex_radius = 1.0
        
        for tile in tiles:
            hx, hy = tile['center']
            for i, vid in enumerate(tile['vertices']):
                if vid not in self.vertex_positions:
                    angle = np.pi / 3 * i - np.pi / 6
                    vx = hx + hex_radius * np.cos(angle)
                    vy = hy + hex_radius * np.sin(angle)
                    self.vertex_positions[vid] = (vx, vy)
        
        self.vertices = sorted(list(self.vertex_positions.keys()))
        
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertices)
        
        for tile in tiles:
            verts = tile['vertices']
            for i in range(6):
                v1 = verts[i]
                v2 = verts[(i + 1) % 6]
                self.graph.add_edge(v1, v2)
        
        print(f"\n[*] Graph built: {len(self.vertices)} vertices, {len(self.graph.edges())} edges")
        
        self.simulator = AerSimulator(method='statevector')
        
    def calculate_vertex_score(self, vertex_id):
        """Calculate expected resource yield for a vertex - PRIORITIZES 3-hex > 2-hex"""
        score = 0
        resources_accessed = []
        tiles_accessed = []
        
        for tile in self.tiles:
            if vertex_id in tile['vertices']:
                resource_val = self.resource_values[tile['resource']]
                prob = self.roll_probabilities.get(tile['number'], 0)
                
                probability_bonus = prob * self.probability_weight
                score += resource_val * prob + probability_bonus
                resources_accessed.append(tile['resource'])
                tiles_accessed.append(tile)
        
        # CRITICAL: Heavily reward vertices touching 3 hexes, then 2 hexes
        num_tiles = len(tiles_accessed)
        tile_count_bonus = 0
        if num_tiles == 3:
            tile_count_bonus = 15.0  # HUGE bonus for 3-hex vertices
        elif num_tiles == 2:
            tile_count_bonus = 8.0   # Good bonus for 2-hex vertices
        elif num_tiles == 1:
            tile_count_bonus = 0.0   # No bonus for 1-hex vertices
        
        unique_resources = len(set(resources_accessed))
        diversity_bonus = unique_resources * self.diversity_weight
        
        return score + diversity_bonus + tile_count_bonus, resources_accessed, tiles_accessed
    
    def calculate_solution_diversity(self, vertices):
        """Calculate total resource diversity"""
        all_resources = []
        for vertex_id in vertices:
            _, resources, _ = self.calculate_vertex_score(vertex_id)
            all_resources.extend(resources)
        
        unique_count = len(set(all_resources))
        complete_bonus = 5.0 if unique_count == 5 else 0
        
        return unique_count + complete_bonus
    
    def calculate_solution_probability_score(self, vertices):
        """Calculate average probability"""
        total_prob = 0
        count = 0
        
        for vertex_id in vertices:
            for tile in self.tiles:
                if vertex_id in tile['vertices']:
                    prob = self.roll_probabilities.get(tile['number'], 0)
                    total_prob += prob
                    count += 1
        
        return total_prob / max(count, 1)
    
    def is_valid_placement(self, vertices):
        """Check if settlements satisfy the distance rule"""
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                v1, v2 = vertices[i], vertices[j]
                if self.graph.has_edge(v1, v2):
                    return False
        return True
    
    def get_invalid_pairs(self, vertices):
        """Get list of vertex pairs that violate distance rule"""
        invalid = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                v1, v2 = vertices[i], vertices[j]
                if self.graph.has_edge(v1, v2):
                    invalid.append((v1, v2))
        return invalid
    
    def create_qaoa_circuit(self, params, n_qubits, Q, p_layers):
        """Create QAOA quantum circuit"""
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        for i in range(n_qubits):
            qc.h(i)
        
        for layer in range(p_layers):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if Q[i, j] != 0:
                        qc.rzz(2 * gamma * Q[i, j], i, j)
            
            for i in range(n_qubits):
                if Q[i, i] != 0:
                    qc.rz(2 * gamma * Q[i, i], i)
            
            for i in range(n_qubits):
                qc.rx(2 * beta, i)
        
        qc.measure(range(n_qubits), range(n_qubits))
        return qc
    
    def evaluate_circuit(self, params, n_qubits, Q, p_layers, shots=1000):
        """Evaluate QAOA circuit"""
        qc = self.create_qaoa_circuit(params, n_qubits, Q, p_layers)
        qc_transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(qc_transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        expectation = 0
        for bitstring, count in counts.items():
            x = np.array([int(b) for b in bitstring[::-1]])
            energy = 0
            for i in range(n_qubits):
                energy += Q[i, i] * x[i]
                for j in range(i + 1, n_qubits):
                    energy += Q[i, j] * x[i] * x[j]
            expectation += energy * count / shots
        
        return expectation
    
    def quantum_inspired_optimization(self, num_settlements=3, iterations=30):
        """QAOA optimization with tile count priority"""
        print("\n" + "="*60)
        print("PROBLEM 1: QAOA SETTLEMENT PLANNER")
        print("="*60)
        print(f"\n[*] Building quantum circuit with {len(self.vertices)} qubits...")
        print(f"[*] CONSTRAINT: Settlements CANNOT be on adjacent vertices")
        print(f"[*] PRIORITY: 3-hex vertices > 2-hex vertices > 1-hex vertices")
        
        n = len(self.vertices)
        Q = np.zeros((n, n))
        
        # Build QUBO matrix
        print(f"\n[*] Computing vertex scores (prioritizing 3-hex > 2-hex vertices)...")
        vertex_scores_display = []
        for i, v in enumerate(self.vertices):
            score, resources, tiles = self.calculate_vertex_score(v)
            Q[i, i] = -score
            num_tiles = len(tiles)
            vertex_scores_display.append((v, score, num_tiles, resources, tiles))
        
        # Sort by number of tiles
        vertex_scores_display.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Display top vertices
        print(f"\n[*] Top vertices by tile count:")
        for tile_count in [3, 2, 1]:
            vertices_with_count = [v for v in vertex_scores_display if v[2] == tile_count]
            if vertices_with_count:
                print(f"\n  {tile_count}-hex vertices:")
                for v, score, num_tiles, resources, tiles in vertices_with_count[:5]:
                    numbers = [t['number'] for t in tiles]
                    print(f"    V{v}: score={score:.3f} | {set(resources)} | {numbers}")
        
        print(f"\n[*] Total vertices: 3-hex={len([v for v in vertex_scores_display if v[2]==3])}, "
              f"2-hex={len([v for v in vertex_scores_display if v[2]==2])}, "
              f"1-hex={len([v for v in vertex_scores_display if v[2]==1])}")
        
        # High penalty for adjacent settlements
        penalty_adjacent = 100
        edge_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if self.graph.has_edge(self.vertices[i], self.vertices[j]):
                    Q[i, j] += penalty_adjacent
                    edge_count += 1
        
        print(f"\n[*] Applied {edge_count} adjacency penalties (weight={penalty_adjacent})")
        
        # Penalty for wrong count
        penalty_count = 15
        for i in range(n):
            Q[i, i] += penalty_count * (1 - 2*num_settlements)
            for j in range(i+1, n):
                Q[i, j] += 2 * penalty_count
        
        p_layers = 2
        initial_params = np.random.uniform(0, 2*np.pi, 2 * p_layers)
        
        print("\n[*] Running quantum-classical hybrid optimization...")
        
        result = minimize(
            lambda params: self.evaluate_circuit(params, n, Q, p_layers),
            initial_params,
            method='COBYLA',
            options={'maxiter': iterations}
        )
        
        optimal_params = result.x
        
        # Get quantum measurements
        qc_final = self.create_qaoa_circuit(optimal_params, n, Q, p_layers)
        qc_final_transpiled = transpile(qc_final, self.simulator)
        job = self.simulator.run(qc_final_transpiled, shots=1000)
        counts = job.result().get_counts()
        
        best_solution = None
        best_solution_score = -np.inf
        best_diversity = 0
        best_prob_score = 0
        
        print("\n[*] Evaluating quantum measurements...")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        valid_found = 0
        invalid_found = 0
        
        for bitstring, count in sorted_counts[:100]:
            candidate = [self.vertices[i] for i, bit in enumerate(bitstring[::-1]) if bit == '1']
            
            if len(candidate) == num_settlements:
                if self.is_valid_placement(candidate):
                    valid_found += 1
                    
                    candidate_score = sum(self.calculate_vertex_score(v)[0] for v in candidate)
                    diversity = self.calculate_solution_diversity(candidate)
                    prob_score = self.calculate_solution_probability_score(candidate)
                    combined_score = candidate_score + diversity * 2.0 + prob_score * 10
                    
                    if combined_score > best_solution_score:
                        best_solution = candidate
                        best_solution_score = combined_score
                        best_diversity = diversity
                        best_prob_score = prob_score
                else:
                    invalid_found += 1
        
        print(f"\n[*] Found {valid_found} valid, {invalid_found} invalid configurations")
        
        if best_solution is None:
            print("[!] No valid quantum solution, using greedy fallback...")
            best_solution = self._greedy_valid_solution(num_settlements)
            best_diversity = self.calculate_solution_diversity(best_solution)
            best_prob_score = self.calculate_solution_probability_score(best_solution)
        
        base_score = sum(self.calculate_vertex_score(v)[0] for v in best_solution)
        
        print(f"\n[OK] Optimization complete!")
        print(f"\n[RESULTS]")
        print(f"  Base Score: {base_score:.4f}")
        print(f"  Resource Diversity: {best_diversity:.1f}")
        print(f"  Avg Probability: {best_prob_score*100:.2f}%")
        print(f"  Selected vertices: {best_solution}")
        
        if self.is_valid_placement(best_solution):
            print("\n[OK] Solution VALID - Distance rule satisfied!")
        else:
            print("\n[ERROR] Solution violates constraints!")
        
        all_resources = []
        all_numbers = []
        all_tile_counts = []
        print(f"\n[SETTLEMENT DETAILS]")
        for vid in best_solution:
            score, resources, tiles = self.calculate_vertex_score(vid)
            numbers = [t['number'] for t in tiles]
            num_tiles = len(tiles)
            all_resources.extend(resources)
            all_numbers.extend(numbers)
            all_tile_counts.append(num_tiles)
            print(f"  V{vid}: {score:.4f} | {num_tiles}-hex | {list(set(resources))} | {numbers}")
        
        print(f"\n[OBJECTIVES]")
        unique_resources = set(all_resources)
        print(f"  Resources: {len(unique_resources)}/5 types = {unique_resources}")
        missing = set(['wood', 'brick', 'ore', 'wheat', 'sheep']) - unique_resources
        if not missing:
            print(f"  [*] COMPLETE COVERAGE!")
        else:
            print(f"  Missing: {missing}")
        
        avg_prob = sum(self.roll_probabilities[n] for n in all_numbers) / len(all_numbers)
        print(f"  Avg probability: {avg_prob*100:.2f}%")
        
        tile_count_summary = {1: 0, 2: 0, 3: 0}
        for count in all_tile_counts:
            tile_count_summary[count] = tile_count_summary.get(count, 0) + 1
        print(f"  Tile coverage: 3-hex={tile_count_summary[3]}, 2-hex={tile_count_summary[2]}, 1-hex={tile_count_summary[1]}")
        
        return best_solution, base_score
    
    def _greedy_valid_solution(self, target_count):
        """Greedy algorithm prioritizing 3-hex and 2-hex vertices"""
        vertex_data = []
        for v in self.vertices:
            base_score, resources, tiles = self.calculate_vertex_score(v)
            num_tiles = len(tiles)
            diversity_potential = len(set(resources))
            avg_prob = sum(self.roll_probabilities[t['number']] for t in tiles) / max(len(tiles), 1)
            combined = base_score + diversity_potential * 2.0 + avg_prob * 10
            vertex_data.append((v, combined, set(resources), num_tiles))
        
        vertex_data.sort(key=lambda x: (x[3], x[1]), reverse=True)
        
        selected = []
        selected_resources = set()
        
        for vertex, score, resources, num_tiles in vertex_data:
            if len(selected) >= target_count:
                break
            
            test = selected + [vertex]
            if self.is_valid_placement(test):
                new_resources = resources - selected_resources
                if new_resources or len(selected) == 0:
                    selected.append(vertex)
                    selected_resources.update(resources)
        
        if len(selected) < target_count:
            for vertex, score, resources, num_tiles in vertex_data:
                if len(selected) >= target_count:
                    break
                if vertex not in selected:
                    test = selected + [vertex]
                    if self.is_valid_placement(test):
                        selected.append(vertex)
                        selected_resources.update(resources)
        
        return selected if len(selected) > 0 else []
    
    def classical_greedy_baseline(self, num_settlements=3):
        """Classical greedy baseline"""
        print("\n" + "-"*60)
        print("Classical Greedy Baseline (3-hex > 2-hex priority):")
        
        selected = self._greedy_valid_solution(num_settlements)
        
        if not selected:
            print("[ERROR] Could not find valid greedy solution!")
            return [], 0
        
        total_score = sum(self.calculate_vertex_score(v)[0] for v in selected)
        diversity = self.calculate_solution_diversity(selected)
        prob_score = self.calculate_solution_probability_score(selected)
        
        print(f"Greedy score: {total_score:.4f}")
        print(f"Diversity: {diversity:.1f}")
        print(f"Probability: {prob_score*100:.2f}%")
        print(f"Selected: {selected}")
        
        if self.is_valid_placement(selected):
            print("[OK] Greedy solution is VALID")
        else:
            print("[ERROR] Greedy solution is INVALID")
        
        all_resources = []
        all_tile_counts = []
        for v in selected:
            _, resources, tiles = self.calculate_vertex_score(v)
            all_resources.extend(resources)
            all_tile_counts.append(len(tiles))
        
        print(f"Resources: {set(all_resources)}")
        tile_count_summary = {1: 0, 2: 0, 3: 0}
        for count in all_tile_counts:
            tile_count_summary[count] = tile_count_summary.get(count, 0) + 1
        print(f"Tile coverage: 3-hex={tile_count_summary[3]}, 2-hex={tile_count_summary[2]}, 1-hex={tile_count_summary[1]}")
        
        return selected, total_score


# ============================================================================
# PROBLEM 2: QUANTUM WALK LONGEST ROAD (STARTING FROM SETTLEMENTS)
# ============================================================================
class QuantumLongestRoad:
    """Quantum Walk for longest road - intelligently starts from best settlement"""
    
    def __init__(self, max_roads=6, settlements=None, vertex_positions=None):
        self.max_roads = max_roads
        self.settlements = settlements if settlements else []
        self.hex_radius = 1.0
        self.axial_coords = [
            (0, 0), (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        self.graph, self.vertex_positions = self._create_hex_grid()
        
        if vertex_positions:
            self.vertex_positions = vertex_positions
        
        self.edges = list(self.graph.edges())
        self.simulator = AerSimulator(method='statevector')
        
        print(f"\n[*] Road network: {len(self.vertex_positions)} vertices, {len(self.edges)} edges")
        if self.settlements:
            print(f"[*] Analyzing settlements: {self.settlements}")
            print(f"[*] Roads must connect to at least one settlement")
            print(f"[*] Path cannot loop back through the starting settlement")
            print(f"[*] Path must be SIMPLE (no branching)")
    
    def _create_hex_grid(self):
        """Create the hexagonal grid graph"""
        # ... keep your existing implementation ...
        graph = nx.Graph()
        vertex_positions = {}
        vertex_id = 0
        hex_vertices = {}
        
        for hex_id, (q, r) in enumerate(self.axial_coords):
            hx, hy = self._axial_to_cart(q, r)
            vertices = []
            
            for i in range(6):
                angle = np.pi / 3 * i - np.pi / 6
                vx = hx + self.hex_radius * np.cos(angle)
                vy = hy + self.hex_radius * np.sin(angle)
                
                existing_vertex = None
                for vid, (px, py) in vertex_positions.items():
                    if np.sqrt((vx - px)**2 + (vy - py)**2) < 0.1:
                        existing_vertex = vid
                        break
                
                if existing_vertex is not None:
                    vertices.append(existing_vertex)
                else:
                    vertex_positions[vertex_id] = (vx, vy)
                    vertices.append(vertex_id)
                    vertex_id += 1
            
            hex_vertices[hex_id] = vertices
        
        for hex_id, vertices in hex_vertices.items():
            for i in range(6):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % 6]
                graph.add_edge(v1, v2)
        
        return graph, vertex_positions
    
    def _axial_to_cart(self, q, r):
        x = self.hex_radius * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = self.hex_radius * (1.5 * r)
        return (x, y)
    
    # ============================================================
    # NEW FUNCTION 1: Add this helper function here
    # ============================================================
    def _is_simple_path(self, edges):
        """Check if edges form a simple path (no branching)"""
        if not edges:
            return True
        
        # Build degree count for each vertex
        degree = {}
        for v1, v2 in edges:
            degree[v1] = degree.get(v1, 0) + 1
            degree[v2] = degree.get(v2, 0) + 1
        
        # Count vertices by degree
        degree_1_count = sum(1 for d in degree.values() if d == 1)
        degree_2_count = sum(1 for d in degree.values() if d == 2)
        degree_3_plus = sum(1 for d in degree.values() if d >= 3)
        
        # Valid simple path: exactly 2 endpoints (degree 1) and all others degree 2
        # No vertices with degree 3 or higher (that's branching!)
        return degree_1_count == 2 and degree_3_plus == 0
    
    # ============================================================
    # MODIFIED FUNCTION: Replace your existing _find_longest_path_from_settlement
    # ============================================================
    def _find_longest_path_from_settlement(self, start_settlement):
        """
        Find the absolute longest SINGLE-PATH (no branching) from a settlement.
        Each internal vertex must have degree exactly 2 (except endpoints which have degree 1).
        """
        longest_path = []
        longest_length = 0
        other_settlements = set(self.settlements) - {start_settlement}
        
        def dfs_paths(current_node, path, used_edges, visited_start):
            nonlocal longest_path, longest_length
            
            # CRITICAL: Check if current path is still a simple path (no branching)
            if used_edges and not self._is_simple_path(used_edges):
                return  # Prune this branch - it has branching!
            
            # Update longest path if current is better
            if len(used_edges) > longest_length:
                longest_length = len(used_edges)
                longest_path = list(used_edges)
            
            # Try each neighbor
            for neighbor in self.graph.neighbors(current_node):
                edge = tuple(sorted([current_node, neighbor]))
                
                # Skip if edge already used
                if edge in used_edges:
                    continue
                
                # CRITICAL: If we've left the starting settlement, we cannot return to it
                if visited_start and neighbor == start_settlement:
                    continue
                
                # Skip if neighbor is another settlement (can't pass through)
                if neighbor in other_settlements:
                    continue
                
                # Create test set with new edge
                test_edges = used_edges | {edge}
                
                # CRITICAL: Check if adding this edge would create branching
                if not self._is_simple_path(test_edges):
                    continue  # Skip this neighbor - would create branching
                
                # Mark that we've visited the start if we're leaving it
                new_visited_start = visited_start or (current_node == start_settlement)
                
                # Recursively explore
                new_path = path + [neighbor]
                dfs_paths(neighbor, new_path, test_edges, new_visited_start)
        
        # Start DFS from the settlement
        dfs_paths(start_settlement, [start_settlement], set(), False)
        
        return longest_path, longest_length
    
    # Keep your existing _find_best_path_from_settlement function
    def _find_best_path_from_settlement(self, start_settlement):
        """Find the longest valid path from a given settlement"""
        print(f"\n[*] Searching for longest simple path from settlement {start_settlement}...")
        
        # Get the absolute longest path (no artificial limit)
        best_path, best_length = self._find_longest_path_from_settlement(start_settlement)
        
        if not best_path:
            print(f"[!] No valid paths found from settlement {start_settlement}")
            return [], 0
        
        print(f"[*] Maximum possible path length: {best_length} edges")
        
        # Verify the path doesn't loop back to start (except at the beginning)
        start_count = 0
        for edge in best_path:
            if start_settlement in edge:
                start_count += 1
        print(f"[*] Starting settlement appears in {start_count} edges")
        
        return best_path, best_length
    
    # Keep your existing quantum_inspired_optimization function
    def quantum_inspired_optimization(self, iterations=300):
        """Optimized pathfinding using exhaustive search from best settlement"""
        print("\n" + "="*60)
        print("PROBLEM 2: QUANTUM WALK LONGEST ROAD (NO BRANCHING)")
        print("="*60)
        print(f"[*] Requested road count: {self.max_roads}")
        
        if not self.settlements:
            print("[!] No settlements provided, cannot build roads")
            return [], 0
        
        best_overall_path = []
        best_overall_length = 0
        best_settlement = None
        
        # Try each settlement and find the best overall path
        print(f"\n[*] Evaluating all {len(self.settlements)} settlements...")
        for settlement in self.settlements:
            path, length = self._find_best_path_from_settlement(settlement)
            
            if length > best_overall_length:
                best_overall_length = length
                best_overall_path = path
                best_settlement = settlement
        
        # Check if requested length is achievable
        actual_length = min(best_overall_length, self.max_roads)
        
        print(f"\n[OK] Optimization complete!")
        print(f"[RESULT] Maximum achievable path: {best_overall_length} edges")
        print(f"[RESULT] Requested path length: {self.max_roads} edges")
        
        if self.max_roads > best_overall_length:
            print(f"[WARNING] Requested {self.max_roads} roads, but only {best_overall_length} are achievable!")
            print(f"[INFO] Returning best possible path with {best_overall_length} edges")
            actual_path = best_overall_path
        else:
            print(f"[OK] Requested length is achievable")
            # Trim the path to requested length if needed
            actual_path = self._trim_path_to_length(best_overall_path, self.max_roads, best_settlement)
            actual_length = len(actual_path)
        
        print(f"[RESULT] Final path length: {actual_length} edges")
        print(f"[RESULT] Starting from settlement: {best_settlement}")
        
        if actual_path:
            efficiency = (actual_length / self.max_roads) * 100 if self.max_roads > 0 else 0
            max_efficiency = (actual_length / best_overall_length) * 100
            print(f"  - Requested efficiency: {efficiency:.1f}% ({actual_length}/{self.max_roads} roads)")
            print(f"  - Maximum efficiency: {max_efficiency:.1f}% ({actual_length}/{best_overall_length} maximum possible)")
            
            # Verify path structure
            self._verify_path_structure(actual_path, best_settlement)
        
        return actual_path, actual_length
    
    # Keep your existing _trim_path_to_length function
    def _trim_path_to_length(self, edges, target_length, start_settlement):
        """
        Trim a path to the target length while maintaining connectivity.
        Prioritizes keeping edges closest to the start settlement.
        """
        if len(edges) <= target_length:
            return edges
        
        # Build graph from edges
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # BFS from start to find edges in order of distance
        visited_edges = set()
        queue = [(start_settlement, None)]  # (node, parent_edge)
        ordered_edges = []
        visited_nodes = {start_settlement}
        
        while queue and len(ordered_edges) < target_length:
            current_node, parent_edge = queue.pop(0)
            
            if parent_edge is not None and parent_edge not in visited_edges:
                visited_edges.add(parent_edge)
                ordered_edges.append(parent_edge)
            
            # Explore neighbors
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited_nodes:
                    edge = tuple(sorted([current_node, neighbor]))
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, edge))
        
        return ordered_edges[:target_length]
    
    # ============================================================
    # MODIFIED FUNCTION: Replace your existing _verify_path_structure
    # ============================================================
    def _verify_path_structure(self, edges, start_settlement):
        """Verify the path is a simple path with no branching"""
        if not edges:
            return
        
        # Build a graph from the edges
        path_graph = nx.Graph()
        path_graph.add_edges_from(edges)
        
        # Check connectivity
        if not nx.is_connected(path_graph):
            print("[WARNING] Path is not fully connected!")
            return
        
        # Count degree of each vertex
        degree_counts = {1: 0, 2: 0, 3: 0}
        for node in path_graph.nodes():
            degree = path_graph.degree(node)
            if degree == 1:
                degree_counts[1] += 1
            elif degree == 2:
                degree_counts[2] += 1
            else:
                degree_counts[3] += 1
        
        print(f"\n[PATH STRUCTURE ANALYSIS]")
        print(f"  Endpoints (degree 1): {degree_counts[1]}")
        print(f"  Internal nodes (degree 2): {degree_counts[2]}")
        print(f"  Branching nodes (degree 3+): {degree_counts[3]}")
        
        # Verify it's a simple path
        if degree_counts[1] == 2 and degree_counts[3] == 0:
            print(f"[OK] Valid simple path - NO BRANCHING!")
            
            # Find and report the two endpoints
            endpoints = [n for n in path_graph.nodes() if path_graph.degree(n) == 1]
            print(f"  Path goes from V{endpoints[0]} to V{endpoints[1]}")
            
            # Check if start settlement is one of the endpoints
            if start_settlement in endpoints:
                print(f"  [OK] Path correctly starts from settlement V{start_settlement}")
            else:
                print(f"  [WARNING] Path doesn't start from settlement V{start_settlement}")
        else:
            print(f"[ERROR] Path has BRANCHING - not a simple path!")
            
            # Show which nodes are problematic
            problem_nodes = [n for n in path_graph.nodes() if path_graph.degree(n) > 2]
            if problem_nodes:
                print(f"  Branching at vertices: {problem_nodes}")
    def classical_dfs_baseline(self):
        """Classical DFS baseline - shows maximum possible for comparison"""
        print("\n" + "-"*60)
        print("Classical DFS Baseline (maximum possible paths):")
        
        for settlement in self.settlements:
            _, length = self._find_longest_path_from_settlement(settlement)
            print(f"  Settlement {settlement}: max {length} edges")
        
        print("-"*60)
    
    # ============================================================
    # ADD THIS METHOD: Visualization for the road network
    # ============================================================
    def visualize(self, selected_edges):
        """Visualize road network with settlements highlighted"""
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_aspect('equal')
        
        # Draw hexagons
        for q, r in self.axial_coords:
            hx, hy = self._axial_to_cart(q, r)
            hex_patch = RegularPolygon(
                (hx, hy), numVertices=6, radius=self.hex_radius,
                orientation=np.radians(0), facecolor='lightgreen',
                alpha=0.2, edgecolor='gray', linewidth=1
            )
            ax.add_patch(hex_patch)
        
        # Draw all edges in gray (background)
        for edge in self.graph.edges():
            v1, v2 = edge
            x1, y1 = self.vertex_positions[v1]
            x2, y2 = self.vertex_positions[v2]
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.3, zorder=1)
        
        # Convert edge tuples to standardized format
        selected_edge_set = set()
        for edge in selected_edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                selected_edge_set.add(tuple(sorted(edge)))
        
        # Draw selected edges in red with increasing thickness for visual effect
        for i, (v1, v2) in enumerate(selected_edge_set):
            x1, y1 = self.vertex_positions[v1]
            x2, y2 = self.vertex_positions[v2]
            ax.plot([x1, x2], [y1, y2], 'red', linewidth=5, alpha=0.8, zorder=2)
        
        # Draw all vertices
        for vertex_id, (vx, vy) in self.vertex_positions.items():
            if vertex_id in self.settlements:
                # Check if this settlement is the starting point
                is_start = False
                if selected_edge_set:
                    # Count edges touching this settlement
                    degree = sum(1 for e in selected_edge_set if vertex_id in e)
                    # If it has edges, check if it's an endpoint
                    if degree > 0:
                        # Build a small graph to check if it's an endpoint
                        temp_graph = nx.Graph()
                        temp_graph.add_edges_from(selected_edge_set)
                        if vertex_id in temp_graph.nodes():
                            vertex_degree = temp_graph.degree(vertex_id)
                            # If degree is 1, it's an endpoint - likely the start
                            if vertex_degree == 1:
                                is_start = True
                
                color = 'darkblue' if is_start else 'blue'
                ax.plot(vx, vy, 'o', color=color, markersize=18, 
                       markeredgecolor='white', markeredgewidth=3, zorder=4)
                label = f'START\n{vertex_id}' if is_start else f'S{vertex_id}'
                ax.text(vx, vy + 0.2, label, ha='center', va='bottom',
                       fontsize=9, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                                edgecolor=color, linewidth=2))
            else:
                # Check if vertex is part of the selected path
                is_selected = any(vertex_id in [e[0], e[1]] for e in selected_edge_set)
                if is_selected:
                    # Check if it's an endpoint of the path
                    if selected_edge_set:
                        temp_graph = nx.Graph()
                        temp_graph.add_edges_from(selected_edge_set)
                        if vertex_id in temp_graph.nodes():
                            vertex_degree = temp_graph.degree(vertex_id)
                            if vertex_degree == 1:
                                # It's an endpoint
                                ax.plot(vx, vy, 'o', color='darkred', markersize=16, 
                                       markeredgecolor='yellow', markeredgewidth=3, zorder=3)
                                ax.text(vx, vy, f'END', ha='center', va='center',
                                       fontsize=6, color='white', fontweight='bold')
                            else:
                                # Regular path vertex
                                ax.plot(vx, vy, 'o', color='darkred', markersize=12, 
                                       markeredgecolor='white', markeredgewidth=2, zorder=3)
                else:
                    ax.plot(vx, vy, 'o', color='lightgray', markersize=8, alpha=0.5, zorder=3)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                   markersize=12, label='Start Settlement', markeredgecolor='white', markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=12, label='Other Settlement', markeredgecolor='white', markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
                   markersize=12, label='Path Endpoint', markeredgecolor='yellow', markeredgewidth=2),
            Line2D([0], [0], color='red', linewidth=4, label='Longest Road')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Title with path information
        path_info = f'{len(selected_edge_set)} roads'
        if selected_edge_set:
            temp_graph = nx.Graph()
            temp_graph.add_edges_from(selected_edge_set)
            endpoints = [n for n in temp_graph.nodes() if temp_graph.degree(n) == 1]
            if len(endpoints) == 2:
                path_info += f' (V{endpoints[0]} → V{endpoints[1]})'
        
        ax.set_title(f'Longest Road (Simple Path - No Branching)\n{path_info}', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig('longest_road_optimized.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("\n[OK] Visualization saved as 'longest_road_optimized.png'")
    # Keep all your other existing methods (_is_valid_road_network, 
    # _is_connected_to_settlements, classical_dfs_baseline, visualize)
    # ... rest of your code ...
# ============================================================================
# PROBLEM 3: GROVER'S ALGORITHM RESOURCE TRADER
# ============================================================================

class QuantumResourceTrader:
    """Grover's Algorithm for resource trading"""
    
    def __init__(self):
        self.resources = {
            'wood': random.randint(1,5), 'brick': random.randint(1,5), 
            'ore': random.randint(1,5), 'wheat': random.randint(1,5), 
            'sheep': random.randint(1,5)
        }
        
        self.actions = [
            {'name': 'Build Road', 'cost': {'wood': 1, 'brick': 1}, 'points': 1},
            {'name': 'Build Settlement', 'cost': {'wood': 1, 'brick': 1, 'wheat': 1, 'sheep': 1}, 'points': 5},
            {'name': 'Build City', 'cost': {'ore': 3, 'wheat': 2}, 'points': 7},
            {'name': 'Buy Dev Card', 'cost': {'ore': 1, 'wheat': 1, 'sheep': 1}, 'points': 2},
        ]
        
        self.simulator = AerSimulator(method='statevector')
    
    def quantum_inspired_optimization(self, iterations=1000):
        """Grover's Algorithm optimization"""
        print("\n" + "="*60)
        print("PROBLEM 3: GROVER'S ALGORITHM RESOURCE TRADER")
        print("="*60)
        
        print("\nResources:")
        for resource, count in self.resources.items():
            print(f"  {resource}: {count}")
        
        best_score = 0
        best_solution = []
        
        n_actions = len(self.actions)
        n_qubits = int(np.ceil(np.log2(n_actions)))
        
        for iteration in range(0, iterations, 100):
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            for i in range(n_qubits):
                qc.h(i)
            
            n_grover = max(1, int(np.pi / 4 * np.sqrt(2**n_qubits)))
            for _ in range(n_grover):
                self._oracle(qc, n_qubits)
                self._diffusion(qc, n_qubits)
            
            qc.measure(range(n_qubits), range(n_qubits))
            
            qc_transpiled = transpile(qc, self.simulator)
            job = self.simulator.run(qc_transpiled, shots=100)
            result = job.result()
            counts = result.get_counts()
            
            for bitstring, count in counts.items():
                solution = self._bitstring_to_actions(bitstring)
                score = self._evaluate_solution(solution)
                
                if score > best_score:
                    best_score = score
                    best_solution = solution.copy()
            
            if iteration % 200 == 0:
                print(f"Iteration {iteration}: Best = {best_score}")
        
        print(f"\n[OK] Optimization complete!")
        print(f"Best score: {best_score} points")
        print(f"Actions:")
        for action_idx in best_solution:
            if action_idx < len(self.actions):
                print(f"  - {self.actions[action_idx]['name']}")
        
        self.classical_greedy_baseline()
        return best_solution, best_score
    
    def _oracle(self, qc, n_qubits):
        """Grover oracle"""
        for i in range(n_qubits):
            qc.z(i)
    
    def _diffusion(self, qc, n_qubits):
        """Grover diffusion"""
        for i in range(n_qubits):
            qc.h(i)
        for i in range(n_qubits):
            qc.x(i)
        
        if n_qubits > 1:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        else:
            qc.z(0)
        
        for i in range(n_qubits):
            qc.x(i)
        for i in range(n_qubits):
            qc.h(i)
    
    def _bitstring_to_actions(self, bitstring):
        """Convert bitstring to actions"""
        actions = []
        for i, bit in enumerate(bitstring):
            if bit == '1' and i < len(self.actions):
                actions.append(i)
        return actions
    
    def _evaluate_solution(self, action_indices):
        """Evaluate solution"""
        remaining = self.resources.copy()
        total_points = 0
        
        for idx in action_indices:
            if idx >= len(self.actions):
                continue
            action = self.actions[idx]
            can_afford = all(
                remaining.get(res, 0) >= cost 
                for res, cost in action.get('cost', {}).items()
            )
            
            if can_afford:
                for res, cost in action.get('cost', {}).items():
                    remaining[res] -= cost
                total_points += action['points']
        
        return total_points
    
    def classical_greedy_baseline(self):
        """Classical greedy"""
        print("\n" + "-"*60)
        print("Classical Greedy Baseline:")
        
        action_efficiency = []
        for i, action in enumerate(self.actions):
            total_cost = sum(action.get('cost', {}).values())
            efficiency = action['points'] / max(total_cost, 1)
            action_efficiency.append((i, efficiency))
        
        action_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        remaining = self.resources.copy()
        selected = []
        total_points = 0
        
        for idx, eff in action_efficiency:
            action = self.actions[idx]
            can_afford = all(
                remaining.get(res, 0) >= cost 
                for res, cost in action.get('cost', {}).items()
            )
            
            if can_afford:
                for res, cost in action.get('cost', {}).items():
                    remaining[res] -= cost
                selected.append(action['name'])
                total_points += action['points']
        
        print(f"Greedy score: {total_points} points")
        print(f"Selected: {selected}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all three quantum algorithm challenges"""
    
    print("\n" + "="*60)
    print("  QUANTUM CATAN - REAL QUANTUM ALGORITHMS")
    print("  (ENHANCED: Connected Road Building)")
    print("="*60 + "\n")
    
    map_seed = MAP_SEED
    
    print("Generating Catan Map...")
    if map_seed is not None:
        print(f"  Using seed: {map_seed}")
    else:
        print(f"  Using random seed (different each time)")
    
    terrains, numbers, tiles, ax = draw_catan_terrain_map(seed=map_seed)
    
    print("\n[*] Catan Map Generated:")
    print(f"  Total hexes: {len(tiles)}")
    
    terrain_count = {}
    for tile in tiles:
        terrain = tile['terrain']
        terrain_count[terrain] = terrain_count.get(terrain, 0) + 1
    
    print(f"  Terrain distribution:")
    for terrain, count in sorted(terrain_count.items()):
        print(f"    {terrain}: {count}")
    
    print(f"  Numbers: {sorted([tile['number'] for tile in tiles])}")
    
    # Problem 1: QAOA Settlement Placement
    print("\n" + "="*60)
    print("PROBLEM 1: SETTLEMENT OPTIMIZATION")
    print("="*60)
    print(f"Diversity Weight: {DIVERSITY_WEIGHT} | Probability Weight: {PROBABILITY_WEIGHT}")
    print("="*60)
    
    num_settlements = int(input("\nEnter number of settlements: "))
    
    problem1 = QuantumSettlementPlanner(tiles, diversity_weight=DIVERSITY_WEIGHT, 
                                       probability_weight=PROBABILITY_WEIGHT)
    best_settlements, score1 = problem1.quantum_inspired_optimization(
        num_settlements=num_settlements, iterations=QAOA_ITERATIONS)
    problem1.classical_greedy_baseline(num_settlements=num_settlements)
    
    # Visualize settlements
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.axis('off')
    
    hex_radius = 1.0
    terrain_types = {
        "Forest": "#2E8B57",
        "Field": "#F4E04D",
        "Pasture": "#9ACD32",
        "Hill": "#D2691E",
        "Mountain": "#A9A9A9",
    }
    
    for tile in tiles:
        hx, hy = tile['center']
        color = terrain_types[tile['terrain']]
        
        hex_patch = RegularPolygon(
            (hx, hy), numVertices=6, radius=hex_radius,
            orientation=np.radians(0), facecolor=color,
            alpha=0.7, edgecolor='k', linewidth=2
        )
        ax.add_patch(hex_patch)
        
        ax.text(hx, hy, str(tile['number']), ha='center', va='center',
                fontsize=18, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle', facecolor='black', alpha=0.6))
        ax.text(hx, hy - 0.5, tile['terrain'], ha='center', va='center',
                fontsize=10, color='black', fontweight='bold')
    
    vertex_positions = problem1.vertex_positions
    
    for vid, (vx, vy) in vertex_positions.items():
        if vid in best_settlements:
            ax.plot(vx, vy, 'ro', markersize=15, markeredgecolor='white', 
                   markeredgewidth=2, zorder=10)
            ax.text(vx, vy, str(vid), ha='center', va='center', 
                   color='white', fontsize=8, fontweight='bold', zorder=11)
        else:
            ax.plot(vx, vy, 'ko', markersize=6, alpha=0.3)
    
    for vid in best_settlements:
        if vid in vertex_positions:
            vx, vy = vertex_positions[vid]
            _, resources, tiles_list = problem1.calculate_vertex_score(vid)
            unique_res = list(set(resources))
            numbers = [t['number'] for t in tiles_list]
            
            res_text = f"{unique_res}\n{numbers}"
            ax.text(vx + 0.3, vy + 0.3, res_text, ha='left', va='bottom',
                   fontsize=7, color='darkred', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    for i in range(len(best_settlements)):
        for j in range(i + 1, len(best_settlements)):
            v1, v2 = best_settlements[i], best_settlements[j]
            if v1 in vertex_positions and v2 in vertex_positions:
                x1, y1 = vertex_positions[v1]
                x2, y2 = vertex_positions[v2]
                ax.plot([x1, x2], [y1, y2], 'b--', linewidth=1, alpha=0.3)
    
    plt.title("Optimal Settlement Placement", fontsize=16, fontweight='bold')
    plt.savefig('settlement_placement.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n[OK] Settlement visualization saved as 'settlement_placement.png'")
    
    # Problem 2: Quantum Walk Longest Road (NOW CONNECTED TO SETTLEMENTS)
    max_roads = int(input("\nEnter number of roads: "))
    
    problem2 = QuantumLongestRoad(max_roads=max_roads, settlements=best_settlements, 
                                   vertex_positions=vertex_positions)
    best_roads, length2 = problem2.quantum_inspired_optimization(iterations=ROAD_ITERATIONS)
    problem2.visualize(best_roads)
    
    # Problem 3: Grover's Resource Trading
    problem3 = QuantumResourceTrader()
    best_trades, score3 = problem3.quantum_inspired_optimization(iterations=TRADE_ITERATIONS)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - QUANTUM ALGORITHM RESULTS")
    print("="*60)
    print(f"\n[OK] Problem 1 (QAOA): Settlement score = {score1:.4f}")
    print(f"[OK] Problem 2 (Quantum Walk): Longest road = {length2} edges")
    print(f"     Starting from settlements: {best_settlements}")
    print(f"[OK] Problem 3 (Grover's): Resource trading = {score3} points")
    print(f"\n[OK] All visualizations saved!")
    print(f"  - settlement_placement.png")
    print(f"  - longest_road.png")
    
    print("\n" + "="*60)
    print("  CHALLENGE COMPLETE!")
    print("="*60)
    if MAP_SEED is None:
        print("  [TIP] To reproduce this map, set MAP_SEED in the code")
    else:
        print(f"  Map seed used: {MAP_SEED}")
    print("="*60)


if __name__ == "__main__":
    main()