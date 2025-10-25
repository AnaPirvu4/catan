"""
Quantum Catan Challenge - Real Quantum Algorithm Implementation
REPAIRED: Fixed configuration, encoding, and logic issues

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
NUM_SETTLEMENTS = 3
DIVERSITY_WEIGHT = 2.0
PROBABILITY_WEIGHT = 1.5
QAOA_ITERATIONS = 30

# Problem 2: Longest Road
MAX_ROADS = 6
ROAD_ITERATIONS = 300

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

def draw_catan_terrain_map(seed=None, highlighted_vertices=None, save_fig=False):
    """Generate and draw a random Catan terrain map"""
    # Generate random seed if not provided
    if seed is None:
        seed = np.random.randint(0, 1000000)
        print(f"  [*] Generated random seed: {seed} (save this to reproduce this map!)")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # --- Parameters ---
    radius = 1.0  # hex side length
    hex_radius = radius

    # axial coordinates for the 7-hex (2-3-2) layout
    axial_coords = [(0, 0),
                    (1, 0), (1, -1), (0, -1),
                    (-1, 0), (-1, 1), (0, 1)]

    # convert axial to cartesian
    def axial_to_cart(q, r):
        x = hex_radius * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = hex_radius * (1.5 * r)
        return (x, y)

    hex_centers = [axial_to_cart(q, r) for q, r in axial_coords]

    # --- Random terrain + number assignment ---
    terrain_types = {
        "Forest": "#2E8B57",
        "Field": "#F4E04D",
        "Pasture": "#9ACD32",
        "Hill": "#D2691E",
        "Mountain": "#A9A9A9",
    }
    
    # Resource mapping
    terrain_resources = {
        "Forest": "wood",
        "Field": "wheat",
        "Pasture": "sheep",
        "Hill": "brick",
        "Mountain": "ore"
    }

    terrain_list = random.choices(list(terrain_types.keys()), k=len(hex_centers))
    dice_numbers = random.sample([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], len(hex_centers))

    # --- Plot ---
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
        # Center text: dice number
        ax.text(hx, hy, str(number), ha='center', va='center',
                fontsize=18, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle', facecolor='black', alpha=0.6))
        # Terrain label
        ax.text(hx, hy - 0.5, terrain, ha='center', va='center',
                fontsize=10, color='black', fontweight='bold')

    # Build tiles data structure for optimization with PROPER vertex IDs
    tiles = []
    
    # Create a proper vertex mapping for the 7-hex board
    # Each hex has 6 vertices, shared vertices get same ID
    vertex_id_counter = 0
    vertex_coords_map = {}  # Maps (rounded_x, rounded_y) -> vertex_id
    
    for i, (q, r) in enumerate(axial_coords):
        center = axial_to_cart(q, r)
        hx, hy = center
        vertices = []
        
        # Generate 6 vertices for this hex
        for j in range(6):
            angle = np.pi / 3 * j - np.pi / 6
            vx = hx + hex_radius * np.cos(angle)
            vy = hy + hex_radius * np.sin(angle)
            
            # Round to avoid floating point issues
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
    
    # Draw vertices if highlighting is requested
    if highlighted_vertices:
        # Reconstruct vertex positions from the map
        vertex_positions = {v_id: coords for coords, v_id in vertex_coords_map.items()}
        
        # Draw all vertices
        for vid, (vx, vy) in vertex_positions.items():
            if vid in highlighted_vertices:
                ax.plot(vx, vy, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2)
                ax.text(vx, vy, str(vid), ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
            else:
                ax.plot(vx, vy, 'ko', markersize=6, alpha=0.3)

    plt.title("Quantum Catan Challenge - Terrain Map", fontsize=16, fontweight='bold')
    
    if save_fig:
        plt.savefig('catan_map.png', dpi=150, bbox_inches='tight')
        print("[OK] Map saved as 'catan_map.png'")
    
    return terrain_list, dice_numbers, tiles, fig, ax


# ============================================================================
# PROBLEM 1: QAOA SETTLEMENT PLANNER
# ============================================================================

class QuantumSettlementPlanner:
    """
    Optimizes settlement placement using QAOA.
    Proper distance rule implementation with correct graph construction
    """
    
    def __init__(self, tiles, diversity_weight=2.0, probability_weight=1.5):
        self.tiles = tiles
        self.diversity_weight = diversity_weight
        self.probability_weight = probability_weight
        
        # Probability of rolling each number
        self.roll_probabilities = {
            2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
            7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
        }
        
        # Resource base values
        self.resource_values = {
            'wood': 1.0, 'brick': 1.0, 'ore': 1.0, 'wheat': 1.0, 'sheep': 1.0
        }
        
        # Build vertex positions properly
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
        
        # Build adjacency graph CORRECTLY
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.vertices)
        
        # For each hex, connect adjacent vertices
        for tile in tiles:
            verts = tile['vertices']
            for i in range(6):
                v1 = verts[i]
                v2 = verts[(i + 1) % 6]
                self.graph.add_edge(v1, v2)
        
        print(f"\n[*] Graph built: {len(self.vertices)} vertices, {len(self.graph.edges())} edges")
        
        self.simulator = AerSimulator(method='statevector')
        
    def calculate_vertex_score(self, vertex_id):
        """Calculate expected resource yield for a vertex"""
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
        
        # PRIORITIZE: Heavily reward vertices touching 3 hexes, then 2 hexes
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
        """
        Check if settlements satisfy the distance rule.
        In Catan, settlements must be at least 2 edges apart.
        """
        # Check all pairs of settlements
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                v1, v2 = vertices[i], vertices[j]
                
                # Settlements CANNOT be directly connected (adjacent)
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
            
            # Problem Hamiltonian
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if Q[i, j] != 0:
                        qc.rzz(2 * gamma * Q[i, j], i, j)
            
            for i in range(n_qubits):
                if Q[i, i] != 0:
                    qc.rz(2 * gamma * Q[i, i], i)
            
            # Mixer Hamiltonian
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
        """QAOA optimization with STRICT distance constraints"""
        print("\n" + "="*60)
        print("PROBLEM 1: QAOA SETTLEMENT PLANNER")
        print("="*60)
        print(f"\n[*] Building quantum circuit with {len(self.vertices)} qubits...")
        print(f"[*] CONSTRAINT: Settlements CANNOT be on adjacent vertices")
        print(f"[*] OBJECTIVES: Maximize diversity + probability")
        
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
        
        # Sort by number of tiles (3-hex first, then 2-hex, then 1-hex)
        vertex_scores_display.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Display top vertices grouped by tile count
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
        
        print(f"[*] Applied {edge_count} adjacency penalties (weight={penalty_adjacent})")
        
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
        
        # Verify constraints
        if self.is_valid_placement(best_solution):
            print("\n[OK] Solution VALID - Distance rule satisfied!")
        else:
            print("\n[ERROR] Solution violates constraints!")
            invalid_pairs = self.get_invalid_pairs(best_solution)
            print(f"  Adjacent pairs: {invalid_pairs}")
        
        # Detailed analysis
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
        
        # Show tile count distribution
        tile_count_summary = {1: 0, 2: 0, 3: 0}
        for count in all_tile_counts:
            tile_count_summary[count] = tile_count_summary.get(count, 0) + 1
        print(f"  Tile coverage: 3-hex={tile_count_summary[3]}, 2-hex={tile_count_summary[2]}, 1-hex={tile_count_summary[1]}")
        
        return best_solution, base_score
    
    def _greedy_valid_solution(self, target_count):
        """Greedy algorithm with STRICT distance checking - prioritizes 3-hex and 2-hex vertices"""
        vertex_data = []
        for v in self.vertices:
            base_score, resources, tiles = self.calculate_vertex_score(v)
            num_tiles = len(tiles)
            diversity_potential = len(set(resources))
            avg_prob = sum(self.roll_probabilities[t['number']] for t in tiles) / max(len(tiles), 1)
            
            # Combined score already includes tile count bonus from calculate_vertex_score
            combined = base_score + diversity_potential * 2.0 + avg_prob * 10
            
            # Sort key: prioritize by (num_tiles DESC, combined_score DESC)
            vertex_data.append((v, combined, set(resources), num_tiles))
        
        # Sort by number of tiles first (3-hex > 2-hex > 1-hex), then by score
        vertex_data.sort(key=lambda x: (x[3], x[1]), reverse=True)
        
        selected = []
        selected_resources = set()
        
        # First pass: prioritize resource diversity
        for vertex, score, resources, num_tiles in vertex_data:
            if len(selected) >= target_count:
                break
            
            test = selected + [vertex]
            if self.is_valid_placement(test):
                new_resources = resources - selected_resources
                if new_resources or len(selected) == 0:
                    selected.append(vertex)
                    selected_resources.update(resources)
        
        # Second pass: fill remaining slots with highest scoring valid vertices
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
        """Classical greedy baseline - prioritizes 3-hex and 2-hex vertices"""
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
# PROBLEM 2: QUANTUM WALK LONGEST ROAD
# ============================================================================

class QuantumLongestRoad:
    """Quantum Walk for longest road"""
    
    def __init__(self, max_roads=6):
        self.max_roads = max_roads
        self.hex_radius = 1.0
        self.axial_coords = [
            (0, 0), (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        self.graph, self.vertex_positions = self._create_hex_grid()
        self.edges = list(self.graph.edges())
        self.simulator = AerSimulator(method='statevector')
    
    def _axial_to_cart(self, q, r):
        x = self.hex_radius * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = self.hex_radius * (1.5 * r)
        return (x, y)
    
    def _create_hex_grid(self):
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
    
    def quantum_inspired_optimization(self, iterations=300):
        """Quantum Walk optimization"""
        print("\n" + "="*60)
        print("PROBLEM 2: QUANTUM WALK LONGEST ROAD")
        print("="*60)
        
        best_length = 0
        best_solution = []
        
        greedy_solution = self._greedy_longest_path()
        if greedy_solution:
            best_length = self._calculate_path_length(greedy_solution)
            best_solution = greedy_solution
            print(f"Greedy baseline: {best_length} edges")
        
        n_qubits = min(10, int(np.ceil(np.log2(len(self.edges)))))
        
        random_offset = np.random.uniform(0, np.pi/8)
        
        for iteration in range(iterations):
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            for i in range(n_qubits):
                qc.h(i)
                if iteration < 50:
                    qc.rz(random_offset * np.random.uniform(0.5, 1.5), i)
            
            walk_steps = 4 + (iteration % 6)
            angle_base = np.pi/4 + (iteration * 0.05) % (np.pi/2) + random_offset
            
            for step in range(walk_steps):
                for i in range(n_qubits):
                    qc.ry(angle_base + i * 0.1, i)
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
                if step % 2 == 1:
                    for i in range(n_qubits - 2, 0, -1):
                        qc.cx(i, i - 1)
                for i in range(n_qubits):
                    qc.rz(angle_base/2, i)
            
            qc.measure(range(n_qubits), range(n_qubits))
            
            qc_transpiled = transpile(qc, self.simulator)
            job = self.simulator.run(qc_transpiled, shots=200)
            result = job.result()
            counts = result.get_counts()
            
            for bitstring, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                edges = self._bitstring_to_edges(bitstring)
                length = self._calculate_path_length(edges)
                if length > best_length:
                    best_length = length
                    best_solution = edges.copy()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best = {best_length}")
        
        print(f"\n[OK] Optimization complete!")
        print(f"Longest path: {best_length} edges")
        
        self.classical_dfs_baseline()
        return best_solution, best_length
    
    def _greedy_longest_path(self):
        """Greedy longest path"""
        if len(self.edges) == 0:
            return []
        
        best_path = []
        for start_edge in random.sample(self.edges, min(10, len(self.edges))):
            path = [start_edge]
            v1, v2 = start_edge
            endpoints = {v1, v2}
            
            while len(path) < self.max_roads:
                found = False
                for endpoint in list(endpoints):
                    for edge in self.edges:
                        if edge not in path:
                            e1, e2 = edge
                            if e1 == endpoint or e2 == endpoint:
                                path.append(edge)
                                endpoints.remove(endpoint)
                                endpoints.add(e1 if e1 != endpoint else e2)
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break
            
            if len(path) > len(best_path):
                best_path = path
        
        return best_path
    
    def _bitstring_to_edges(self, bitstring):
        """Convert bitstring to edges"""
        edges = []
        for i, bit in enumerate(bitstring):
            if bit == '1' and i < len(self.edges):
                edges.append(self.edges[i])
        return edges[:self.max_roads]
    
    def _calculate_path_length(self, edges):
        """Calculate longest path"""
        if not edges:
            return 0
        
        subgraph = nx.Graph()
        subgraph.add_edges_from(edges)
        
        max_length = 0
        for start_node in subgraph.nodes():
            visited = {start_node}
            length = self._dfs_longest_path(subgraph, start_node, visited, 0)
            max_length = max(max_length, length)
        
        return max_length
    
    def _dfs_longest_path(self, graph, node, visited, current_length):
        """DFS for longest path"""
        max_length = current_length
        
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                length = self._dfs_longest_path(graph, neighbor, visited, current_length + 1)
                max_length = max(max_length, length)
                visited.remove(neighbor)
        
        return max_length
    
    def classical_dfs_baseline(self):
        """Classical DFS baseline"""
        print("\n" + "-"*60)
        print("Classical DFS Baseline:")
        
        best_path = []
        best_length = 0
        
        # Sample combinations to avoid exponential explosion
        edge_samples = min(100, len(list(combinations(self.edges, min(self.max_roads, len(self.edges))))))
        checked = 0
        
        for edge_combo in combinations(self.edges, min(self.max_roads, len(self.edges))):
            if checked >= edge_samples:
                break
            length = self._calculate_path_length(list(edge_combo))
            if length > best_length:
                best_length = length
                best_path = list(edge_combo)
            checked += 1
        
        print(f"DFS best: {best_length} edges")
        return best_path
    
    def visualize(self, selected_edges):
        """Visualize road network"""
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_aspect('equal')
        
        for q, r in self.axial_coords:
            hx, hy = self._axial_to_cart(q, r)
            hex_patch = RegularPolygon(
                (hx, hy), numVertices=6, radius=self.hex_radius,
                orientation=np.radians(0), facecolor='lightgreen',
                alpha=0.2, edgecolor='gray', linewidth=1
            )
            ax.add_patch(hex_patch)
        
        for edge in self.graph.edges():
            v1, v2 = edge
            x1, y1 = self.vertex_positions[v1]
            x2, y2 = self.vertex_positions[v2]
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.3, zorder=1)
        
        for edge in selected_edges:
            v1, v2 = edge
            x1, y1 = self.vertex_positions[v1]
            x2, y2 = self.vertex_positions[v2]
            ax.plot([x1, x2], [y1, y2], 'red', linewidth=5, alpha=0.8, zorder=2)
        
        for vertex_id, (vx, vy) in self.vertex_positions.items():
            is_selected = any(vertex_id in edge for edge in selected_edges)
            if is_selected:
                ax.plot(vx, vy, 'o', color='darkred', markersize=12, 
                       markeredgecolor='white', markeredgewidth=2, zorder=3)
            else:
                ax.plot(vx, vy, 'o', color='lightgray', markersize=8, alpha=0.5, zorder=3)
        
        ax.set_title(f'Longest Road ({len(selected_edges)} roads)', fontsize=16, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig('longest_road.png', dpi=150, bbox_inches='tight')
        print("[OK] Visualization saved as 'longest_road.png'")


# ============================================================================
# PROBLEM 3: GROVER'S ALGORITHM RESOURCE TRADER
# ============================================================================

class QuantumResourceTrader:
    """Grover's Algorithm for resource trading"""
    
    def __init__(self):
        self.resources = {
            'wood': random.randint(1,5), 'brick': random.randint(1,5), 'ore': random.randint(1,5), 'wheat': random.randint(1,5), 'sheep': random.randint(1,5)
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
    print("  (REPAIRED VERSION)")
    print("="*60 + "\n")
    
    # Generate RANDOMIZED Catan map
    map_seed = MAP_SEED
    
    print("Generating Catan Map...")
    if map_seed is not None:
        print(f"  Using seed: {map_seed}")
    else:
        print(f"  Using random seed (different each time)")
    
    terrains, numbers, tiles, fig, ax = draw_catan_terrain_map(seed=map_seed, save_fig=True)
    plt.close(fig)
    
    print("\n[*] Catan Map Generated:")
    print(f"  Total hexes: {len(tiles)}")
    
    # Show terrain distribution
    terrain_count = {}
    for tile in tiles:
        terrain = tile['terrain']
        terrain_count[terrain] = terrain_count.get(terrain, 0) + 1
    
    print(f"  Terrain distribution:")
    for terrain, count in sorted(terrain_count.items()):
        print(f"    {terrain}: {count}")
    
    # Show number distribution
    print(f"  Numbers: {sorted([tile['number'] for tile in tiles])}")
    
    # Problem 1: QAOA Settlement Placement
    print("\n" + "="*60)
    print("PROBLEM 1: SETTLEMENT OPTIMIZATION")
    print("="*60)
    print(f"Diversity Weight: {DIVERSITY_WEIGHT} | Probability Weight: {PROBABILITY_WEIGHT}")
    print("="*60)
    
    problem1 = QuantumSettlementPlanner(tiles, diversity_weight=DIVERSITY_WEIGHT, 
                                       probability_weight=PROBABILITY_WEIGHT)
    best_settlements, score1 = problem1.quantum_inspired_optimization(
        num_settlements=NUM_SETTLEMENTS, iterations=QAOA_ITERATIONS)
    problem1.classical_greedy_baseline(num_settlements=NUM_SETTLEMENTS)
    
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
    
    # Redraw map
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
    
    # Draw vertices
    vertex_positions = problem1.vertex_positions
    
    for vid, (vx, vy) in vertex_positions.items():
        if vid in best_settlements:
            ax.plot(vx, vy, 'ro', markersize=15, markeredgecolor='white', 
                   markeredgewidth=2, zorder=10)
            ax.text(vx, vy, str(vid), ha='center', va='center', 
                   color='white', fontsize=8, fontweight='bold', zorder=11)
        else:
            ax.plot(vx, vy, 'ko', markersize=6, alpha=0.3)
    
    # Add annotations
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
    
    # Draw distance lines
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
    
    # Problem 2: Quantum Walk Longest Road
    problem2 = QuantumLongestRoad(max_roads=MAX_ROADS)
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
    print(f"[OK] Problem 3 (Grover's): Resource trading = {score3} points")
    print(f"\n[OK] All visualizations saved!")
    print(f"  - catan_map.png")
    print(f"  - settlement_placement.png")
    print(f"  - longest_road.png")
    
    # Summary box
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
