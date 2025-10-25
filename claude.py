"""
Quantum Catan Challenge - Complete Solution
Hackathon submission using quantum-inspired optimization

Requirements:
pip install numpy matplotlib networkx scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import networkx as nx
import random
from scipy.optimize import minimize
from itertools import combinations, product

# ============================================================================
# CATAN MAP GENERATOR (from main.py)
# ============================================================================

def draw_catan_terrain_map(seed=None, highlighted_vertices=None, save_fig=False):
    """Generate and draw a random Catan terrain map"""
    if seed is not None:
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

    # Build tiles data structure for optimization
    tiles = []
    for i, ((q, r), terrain, number) in enumerate(zip(axial_coords, terrain_list, dice_numbers)):
        # Simplified vertex mapping for 7-hex board
        vertex_map = {
            (0, 0): [0, 1, 2, 3, 4, 5],
            (1, 0): [1, 6, 7, 8, 2, 1],
            (1, -1): [8, 9, 10, 11, 6, 8],
            (0, -1): [5, 4, 11, 12, 13, 5],
            (-1, 0): [3, 14, 15, 16, 4, 3],
            (-1, 1): [14, 17, 18, 19, 15, 14],
            (0, 1): [0, 19, 20, 21, 1, 0]
        }
        
        tiles.append({
            'id': i,
            'axial': (q, r),
            'center': axial_to_cart(q, r),
            'terrain': terrain,
            'resource': terrain_resources[terrain],
            'number': number,
            'vertices': vertex_map.get((q, r), list(range(6)))
        })
    
    # Draw vertices if highlighting is requested
    if highlighted_vertices:
        vertex_positions = {}
        for tile in tiles:
            hx, hy = tile['center']
            for i, vid in enumerate(tile['vertices'][:6]):
                angle = np.pi / 3 * i - np.pi / 6
                vx = hx + hex_radius * np.cos(angle)
                vy = hy + hex_radius * np.sin(angle)
                if vid not in vertex_positions:
                    vertex_positions[vid] = (vx, vy)
        
        # Draw all vertices
        for vid, (vx, vy) in vertex_positions.items():
            if vid in highlighted_vertices:
                ax.plot(vx, vy, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2)
                ax.text(vx, vy, str(vid), ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
            else:
                ax.plot(vx, vy, 'ko', markersize=6, alpha=0.3)

    plt.title("Quantum Catan Challenge – Terrain Map", fontsize=16, fontweight='bold')
    
    if save_fig:
        plt.savefig('catan_map.png', dpi=150, bbox_inches='tight')
        print("✓ Map saved as 'catan_map.png'")
    
    return terrain_list, dice_numbers, tiles, fig, ax


# ============================================================================
# PROBLEM 1: QUANTUM-INSPIRED SETTLEMENT PLANNER
# ============================================================================

class QuantumSettlementPlanner:
    """
    Optimizes settlement placement using quantum-inspired optimization.
    Uses simulated annealing and iterative improvement.
    """
    
    def __init__(self, tiles):
        self.tiles = tiles
        
        # Probability of rolling each number
        self.roll_probabilities = {
            2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
            7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
        }
        
        # Resource values
        self.resource_values = {
            'wood': 1.0,
            'brick': 1.0,
            'ore': 1.2,
            'wheat': 1.1,
            'sheep': 0.9
        }
        
        # Get all unique vertices
        self.vertices = set()
        for tile in tiles:
            self.vertices.update(tile['vertices'])
        self.vertices = sorted(list(self.vertices))
        
    def calculate_vertex_score(self, vertex_id):
        """Calculate expected resource yield for a vertex"""
        score = 0
        resources_accessed = []
        
        for tile in self.tiles:
            if vertex_id in tile['vertices']:
                resource_val = self.resource_values[tile['resource']]
                prob = self.roll_probabilities[tile['number']]
                score += resource_val * prob
                resources_accessed.append(tile['resource'])
        
        # Bonus for resource diversity
        diversity_bonus = len(set(resources_accessed)) * 0.1
        
        return score + diversity_bonus, resources_accessed
    
    def is_valid_placement(self, vertices):
        """Check if settlements maintain minimum distance"""
        # For simplicity, we'll check if vertices are not in the same tile
        for v1, v2 in combinations(vertices, 2):
            # Check if they share any tiles
            tiles_v1 = [t for t in self.tiles if v1 in t['vertices']]
            tiles_v2 = [t for t in self.tiles if v2 in t['vertices']]
            
            # If they share a tile, they're too close
            if any(t in tiles_v2 for t in tiles_v1):
                return False
        
        return True
    
    def quantum_inspired_optimization(self, num_settlements=2, iterations=1000):
        """
        Quantum-inspired optimization using simulated annealing.
        Explores superposition of states through probabilistic sampling.
        """
        print("\n" + "="*60)
        print("PROBLEM 1: QUANTUM-INSPIRED SETTLEMENT PLANNER")
        print("="*60)
        print(f"\nSearching {len(self.vertices)} vertices for {num_settlements} settlements...")
        
        best_score = -np.inf
        best_solution = None
        
        # Temperature schedule (quantum annealing inspired)
        T_initial = 10.0
        T_final = 0.01
        
        # Start with random valid placement
        current_solution = self._random_valid_placement(num_settlements)
        current_score = self._evaluate_solution(current_solution)
        
        for iteration in range(iterations):
            # Temperature decay
            T = T_initial * (T_final / T_initial) ** (iteration / iterations)
            
            # Generate neighbor (quantum tunneling inspired)
            neighbor = self._generate_neighbor(current_solution)
            neighbor_score = self._evaluate_solution(neighbor)
            
            # Acceptance probability (Metropolis criterion)
            delta = neighbor_score - current_score
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_solution = neighbor
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = current_solution.copy()
            
            if iteration % 200 == 0:
                print(f"Iteration {iteration}: Best score = {best_score:.4f}")
        
        print(f"\n✓ Optimization complete!")
        print(f"Best score: {best_score:.4f}")
        print(f"Selected vertices: {best_solution}")
        
        # Show details
        for vid in best_solution:
            score, resources = self.calculate_vertex_score(vid)
            print(f"  Vertex {vid}: Score={score:.4f}, Resources={set(resources)}")
        
        return best_solution, best_score
    
    def _random_valid_placement(self, num_settlements):
        """Generate random valid settlement placement"""
        max_attempts = 1000
        for _ in range(max_attempts):
            placement = random.sample(self.vertices, num_settlements)
            if self.is_valid_placement(placement):
                return placement
        # Fallback: just pick distant vertices
        return self.vertices[:num_settlements]
    
    def _generate_neighbor(self, solution):
        """Generate neighboring solution (swap one settlement)"""
        neighbor = solution.copy()
        idx = random.randint(0, len(neighbor) - 1)
        available = [v for v in self.vertices if v not in neighbor]
        if available:
            neighbor[idx] = random.choice(available)
        return neighbor
    
    def _evaluate_solution(self, solution):
        """Evaluate solution quality"""
        if not self.is_valid_placement(solution):
            return -1000  # Invalid placement
        
        total_score = sum(self.calculate_vertex_score(v)[0] for v in solution)
        return total_score
    
    def classical_greedy_baseline(self, num_settlements=2):
        """Classical greedy approach for comparison"""
        print("\n" + "-"*60)
        print("Classical Greedy Baseline:")
        
        # Score all vertices
        vertex_scores = [(v, self.calculate_vertex_score(v)[0]) for v in self.vertices]
        vertex_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        for vertex, score in vertex_scores:
            test_placement = selected + [vertex]
            if self.is_valid_placement(test_placement):
                selected.append(vertex)
                if len(selected) == num_settlements:
                    break
        
        total_score = sum(self.calculate_vertex_score(v)[0] for v in selected)
        print(f"Greedy score: {total_score:.4f}")
        print(f"Selected vertices: {selected}")
        
        return selected, total_score


# ============================================================================
# PROBLEM 2: QUANTUM-INSPIRED LONGEST ROAD
# ============================================================================

class QuantumLongestRoad:
    """
    Finds longest connected path using quantum-inspired graph optimization.
    Uses a 7-hex Catan grid structure for road network.
    """
    
    def __init__(self, max_roads=6):
        self.max_roads = max_roads
        self.hex_radius = 1.0
        
        # Axial coordinates for 7-hex layout (same as map)
        self.axial_coords = [
            (0, 0),
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        
        # Create hex grid graph with vertices
        self.graph, self.vertex_positions = self._create_hex_grid()
        self.edges = list(self.graph.edges())
    
    def _axial_to_cart(self, q, r):
        """Convert axial coordinates to cartesian"""
        x = self.hex_radius * (np.sqrt(3) * q + np.sqrt(3)/2 * r)
        y = self.hex_radius * (1.5 * r)
        return (x, y)
    
    def _create_hex_grid(self):
        """Create a graph based on 7-hex Catan board structure"""
        graph = nx.Graph()
        vertex_positions = {}
        
        # For each hex, create 6 vertices at the corners
        vertex_id = 0
        hex_vertices = {}  # Map from hex to its vertex IDs
        
        for hex_id, (q, r) in enumerate(self.axial_coords):
            hx, hy = self._axial_to_cart(q, r)
            vertices = []
            
            # Create 6 vertices around the hex
            for i in range(6):
                angle = np.pi / 3 * i - np.pi / 6
                vx = hx + self.hex_radius * np.cos(angle)
                vy = hy + self.hex_radius * np.sin(angle)
                
                # Check if vertex already exists (shared between hexes)
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
        
        # Create edges between adjacent vertices (roads)
        for hex_id, vertices in hex_vertices.items():
            # Connect vertices around each hex
            for i in range(6):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % 6]
                graph.add_edge(v1, v2)
        
        return graph, vertex_positions
    
    def quantum_inspired_optimization(self, iterations=500):
        """Quantum-inspired path finding"""
        print("\n" + "="*60)
        print("PROBLEM 2: QUANTUM-INSPIRED LONGEST ROAD")
        print("="*60)
        print(f"\nNetwork: {len(self.graph.nodes())} nodes, {len(self.edges)} possible edges")
        print(f"Resource constraint: Max {self.max_roads} roads\n")
        
        best_length = 0
        best_solution = []
        
        # Quantum-inspired: explore multiple paths simultaneously
        for iteration in range(iterations):
            # Probabilistic edge selection (superposition collapse)
            selected_edges = self._quantum_sample_edges()
            path_length = self._calculate_path_length(selected_edges)
            
            if path_length > best_length:
                best_length = path_length
                best_solution = selected_edges.copy()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best path length = {best_length}")
        
        print(f"\n✓ Optimization complete!")
        print(f"Longest path: {best_length} edges")
        print(f"Selected roads: {best_solution}")
        
        # Classical comparison
        classical_solution = self.classical_dfs_baseline()
        
        return best_solution, best_length
    
    def _quantum_sample_edges(self):
        """Probabilistically sample edges (quantum-inspired)"""
        # Use quantum-like probability distribution
        probs = np.random.beta(2, 2, len(self.edges))  # Beta distribution
        probs = probs / probs.sum()
        
        num_edges = min(self.max_roads, random.randint(3, self.max_roads))
        selected_indices = np.random.choice(
            len(self.edges), 
            size=num_edges, 
            replace=False, 
            p=probs
        )
        
        return [self.edges[i] for i in selected_indices]
    
    def _calculate_path_length(self, edges):
        """Calculate longest connected path"""
        if not edges:
            return 0
        
        subgraph = nx.Graph()
        subgraph.add_edges_from(edges)
        
        max_length = 0
        for node in subgraph.nodes():
            visited = set()
            length = self._dfs_longest(subgraph, node, visited)
            max_length = max(max_length, length)
        
        return max_length
    
    def _dfs_longest(self, graph, node, visited):
        """DFS to find longest path from node"""
        visited.add(node)
        max_length = 0
        
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                length = 1 + self._dfs_longest(graph, neighbor, visited.copy())
                max_length = max(max_length, length)
        
        return max_length
    
    def classical_dfs_baseline(self):
        """Classical DFS baseline"""
        print("\n" + "-"*60)
        print("Classical DFS Baseline:")
        
        best_path = []
        best_length = 0
        
        # Try all combinations of max_roads edges
        for edge_combo in combinations(self.edges, self.max_roads):
            length = self._calculate_path_length(list(edge_combo))
            if length > best_length:
                best_length = length
                best_path = list(edge_combo)
        
        print(f"DFS best path length: {best_length}")
        print(f"Selected roads: {best_path}")
        
        return best_path
    
    def visualize(self, selected_edges):
        """Visualize road network on hex grid"""
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_aspect('equal')
        
        # Draw hex tiles in background
        for q, r in self.axial_coords:
            hx, hy = self._axial_to_cart(q, r)
            hex_patch = RegularPolygon(
                (hx, hy),
                numVertices=6,
                radius=self.hex_radius,
                orientation=np.radians(0),
                facecolor='lightgreen',
                alpha=0.2,
                edgecolor='gray',
                linewidth=1
            )
            ax.add_patch(hex_patch)
        
        # Draw all possible roads (edges) in light gray
        for edge in self.graph.edges():
            v1, v2 = edge
            x1, y1 = self.vertex_positions[v1]
            x2, y2 = self.vertex_positions[v2]
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.3, zorder=1)
        
        # Draw selected roads in red
        for edge in selected_edges:
            v1, v2 = edge
            x1, y1 = self.vertex_positions[v1]
            x2, y2 = self.vertex_positions[v2]
            ax.plot([x1, x2], [y1, y2], 'red', linewidth=5, alpha=0.8, zorder=2,
                   solid_capstyle='round')
        
        # Draw vertices
        for vertex_id, (vx, vy) in self.vertex_positions.items():
            # Check if this vertex is part of selected path
            is_selected = any(vertex_id in edge for edge in selected_edges)
            
            if is_selected:
                ax.plot(vx, vy, 'o', color='darkred', markersize=12, 
                       markeredgecolor='white', markeredgewidth=2, zorder=3)
            else:
                ax.plot(vx, vy, 'o', color='lightgray', markersize=8, 
                       alpha=0.5, zorder=3)
        
        # Add title and labels
        ax.set_title(f'Longest Road on 7-Hex Catan Board\n({len(selected_edges)} roads selected)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=2, alpha=0.3, label='Available Roads'),
            Line2D([0], [0], color='red', linewidth=5, alpha=0.8, label='Selected Roads'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
                   markersize=10, label='Active Vertices', markeredgecolor='white', markeredgewidth=2)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        ax.axis('off')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig('longest_road.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved as 'longest_road.png'")


# ============================================================================
# PROBLEM 3: QUANTUM-INSPIRED RESOURCE TRADER
# ============================================================================

class QuantumResourceTrader:
    """
    Resource trading optimization using quantum-inspired algorithms.
    """
    
    def __init__(self):
        self.resources = {
            'wood': 3,
            'brick': 2,
            'ore': 1,
            'wheat': 2,
            'sheep': 1
        }
        
        self.actions = [
            {'name': 'Build Road', 'cost': {'wood': 1, 'brick': 1}, 'points': 1},
            {'name': 'Build Settlement', 'cost': {'wood': 1, 'brick': 1, 'wheat': 1, 'sheep': 1}, 'points': 5},
            {'name': 'Build City', 'cost': {'ore': 3, 'wheat': 2}, 'points': 7},
            {'name': 'Buy Dev Card', 'cost': {'ore': 1, 'wheat': 1, 'sheep': 1}, 'points': 2},
        ]
    
    def quantum_inspired_optimization(self, iterations=1000):
        """Quantum-inspired knapsack optimization"""
        print("\n" + "="*60)
        print("PROBLEM 3: QUANTUM-INSPIRED RESOURCE TRADER")
        print("="*60)
        
        print("\nAvailable Resources:")
        for resource, count in self.resources.items():
            print(f"  {resource}: {count}")
        
        print(f"\nPossible Actions: {len(self.actions)}")
        
        best_score = 0
        best_solution = []
        
        # Quantum-inspired: probabilistic action selection
        for iteration in range(iterations):
            solution = self._quantum_sample_actions()
            score = self._evaluate_solution(solution)
            
            if score > best_score:
                best_score = score
                best_solution = solution.copy()
            
            if iteration % 200 == 0:
                print(f"Iteration {iteration}: Best score = {best_score}")
        
        print(f"\n✓ Optimization complete!")
        print(f"Best score: {best_score} points")
        print(f"Selected actions:")
        for action_idx in best_solution:
            print(f"  - {self.actions[action_idx]['name']}")
        
        # Classical comparison
        self.classical_greedy_baseline()
        
        return best_solution, best_score
    
    def _quantum_sample_actions(self):
        """Quantum-inspired probabilistic action sampling"""
        # Generate quantum superposition of action selections
        n = len(self.actions)
        amplitudes = np.random.random(n)
        probs = amplitudes ** 2
        probs = probs / probs.sum()
        
        # Sample actions based on quantum probabilities
        selected = []
        for i, prob in enumerate(probs):
            if np.random.random() < prob:
                selected.append(i)
        
        return selected
    
    def _evaluate_solution(self, action_indices):
        """Evaluate solution feasibility and score"""
        remaining = self.resources.copy()
        total_points = 0
        
        for idx in action_indices:
            action = self.actions[idx]
            # Check if we can afford it
            can_afford = all(
                remaining.get(res, 0) >= cost 
                for res, cost in action.get('cost', {}).items()
            )
            
            if can_afford:
                # Deduct costs
                for res, cost in action.get('cost', {}).items():
                    remaining[res] -= cost
                total_points += action['points']
        
        return total_points
    
    def classical_greedy_baseline(self):
        """Classical greedy baseline"""
        print("\n" + "-"*60)
        print("Classical Greedy Baseline:")
        
        # Sort by points per resource cost
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
    """Run all three quantum-inspired Catan challenges"""
    
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*8 + "QUANTUM CATAN CHALLENGE - SOLUTION" + " "*15 + "║")
    print("║" + " "*10 + "(Quantum-Inspired Algorithms)" + " "*18 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    # Generate random Catan map (no seed = random every time)
    print("Generating Random Catan Map...")
    terrains, numbers, tiles, fig, ax = draw_catan_terrain_map(seed=None, save_fig=True)
    plt.close(fig)
    
    print("\nCatan Map Generated:")
    for tile in tiles:
        print(f"  Tile {tile['id']}: {tile['terrain']} ({tile['number']}) -> {tile['resource']}")
    
    # Problem 1: Settlement Placement
    problem1 = QuantumSettlementPlanner(tiles)
    best_settlements, score1 = problem1.quantum_inspired_optimization(num_settlements=2, iterations=1000)
    problem1.classical_greedy_baseline(num_settlements=2)
    
    # Visualize with highlighted settlements (regenerate with same random state)
    # Note: We need to use the same tiles, so we'll redraw manually
    terrains_viz, numbers_viz, tiles_viz, fig, ax = draw_catan_terrain_map(
        seed=None,  # Will create new random, but we want same map
        highlighted_vertices=best_settlements,
        save_fig=False
    )
    
    # Better approach: redraw the same map we already generated
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
    
    # Redraw the map with settlements
    for tile in tiles:
        hx, hy = tile['center']
        color = terrain_types[tile['terrain']]
        
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
        
        ax.text(hx, hy, str(tile['number']), ha='center', va='center',
                fontsize=18, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle', facecolor='black', alpha=0.6))
        ax.text(hx, hy - 0.5, tile['terrain'], ha='center', va='center',
                fontsize=10, color='black', fontweight='bold')
    
    # Draw vertices with settlements highlighted
    vertex_positions = {}
    for tile in tiles:
        hx, hy = tile['center']
        for i, vid in enumerate(tile['vertices'][:6]):
            angle = np.pi / 3 * i - np.pi / 6
            vx = hx + hex_radius * np.cos(angle)
            vy = hy + hex_radius * np.sin(angle)
            if vid not in vertex_positions:
                vertex_positions[vid] = (vx, vy)
    
    for vid, (vx, vy) in vertex_positions.items():
        if vid in best_settlements:
            ax.plot(vx, vy, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2, zorder=10)
            ax.text(vx, vy, str(vid), ha='center', va='center', 
                   color='white', fontsize=8, fontweight='bold', zorder=11)
        else:
            ax.plot(vx, vy, 'ko', markersize=6, alpha=0.3)
    
    plt.title("Optimal Settlement Placement", fontsize=16, fontweight='bold')
    plt.savefig('settlement_placement.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n✓ Settlement visualization saved as 'settlement_placement.png'")
    
    # Problem 2: Longest Road
    problem2 = QuantumLongestRoad(max_roads=6)
    best_roads, length2 = problem2.quantum_inspired_optimization(iterations=500)
    problem2.visualize(best_roads)
    
    # Problem 3: Resource Trading
    problem3 = QuantumResourceTrader()
    best_trades, score3 = problem3.quantum_inspired_optimization(iterations=1000)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - QUANTUM-INSPIRED OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\n✓ Problem 1: Settlement score = {score1:.4f}")
    print(f"✓ Problem 2: Longest road = {length2} edges")
    print(f"✓ Problem 3: Resource trading = {score3} points")
    print(f"\n✓ All visualizations saved!")
    print("\n" + "="*60)
    print("Challenge Complete! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()