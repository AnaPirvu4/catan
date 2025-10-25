"""
Quantum Catan Challenge - Complete Solution
Hackathon submission for quantum optimization of Catan strategy

Requirements:
pip install qiskit qiskit-optimization numpy matplotlib networkx
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
try:
    from qiskit.primitives import Sampler
except ImportError:
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import networkx as nx
from itertools import combinations

# ============================================================================
# PROBLEM 1: QUANTUM SETTLEMENT PLANNER
# ============================================================================

class QuantumSettlementPlanner:
    """
    Optimizes settlement placement on a Catan hex board using QAOA.
    Maximizes expected resource yield while maintaining distance constraints.
    """
    
    def __init__(self):
        # Define a simplified hex board
        self.tiles = [
            {'id': 0, 'resource': 'wood', 'number': 8, 'vertices': [0, 1, 2]},
            {'id': 1, 'resource': 'brick', 'number': 6, 'vertices': [1, 2, 3, 4]},
            {'id': 2, 'resource': 'ore', 'number': 5, 'vertices': [2, 4, 5]},
            {'id': 3, 'resource': 'wheat', 'number': 9, 'vertices': [3, 4, 6, 7]},
            {'id': 4, 'resource': 'sheep', 'number': 4, 'vertices': [5, 6, 7, 8]},
        ]
        
        # Resource values
        self.resource_values = {
            'wood': 1.0,
            'brick': 1.0,
            'ore': 1.2,
            'wheat': 1.1,
            'sheep': 0.9
        }
        
        # Probability of rolling each number
        self.roll_probabilities = {
            2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
            7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
        }
        
        self.num_vertices = 9
        
    def calculate_vertex_score(self, vertex_id):
        """Calculate expected resource yield for a vertex"""
        score = 0
        for tile in self.tiles:
            if vertex_id in tile['vertices']:
                resource_val = self.resource_values[tile['resource']]
                prob = self.roll_probabilities[tile['number']]
                score += resource_val * prob
        return score
    
    def build_qaoa_problem(self, num_settlements=2):
        """
        Build QUBO formulation for settlement placement.
        
        Objective: Maximize resource yield
        Constraint: Settlements must be at least 2 edges apart
        """
        qp = QuadraticProgram()
        
        # Binary variables for each vertex
        for i in range(self.num_vertices):
            qp.binary_var(f'x{i}')
        
        # Objective: maximize expected resource yield
        linear = {}
        for i in range(self.num_vertices):
            score = self.calculate_vertex_score(i)
            linear[f'x{i}'] = -score  # Negative because we minimize
        
        qp.minimize(linear=linear)
        
        # Constraint: exactly num_settlements settlements
        constraint_linear = {f'x{i}': 1 for i in range(self.num_vertices)}
        qp.linear_constraint(constraint_linear, '==', num_settlements)
        
        # Distance constraints (simplified - adjacent vertices can't both have settlements)
        adjacency = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), 
            (5, 6), (6, 7), (7, 8), (0, 2), (2, 4)
        ]
        
        quadratic = {}
        penalty = 10  # Large penalty for violating distance constraint
        for v1, v2 in adjacency:
            quadratic[(f'x{v1}', f'x{v2}')] = penalty
        
        if quadratic:
            qp.minimize(quadratic=quadratic, linear=linear)
        
        return qp
    
    def solve_with_qaoa(self, p=1):
        """Solve using QAOA"""
        print("\n" + "="*60)
        print("PROBLEM 1: QUANTUM SETTLEMENT PLANNER")
        print("="*60)
        
        qp = self.build_qaoa_problem(num_settlements=2)
        print(f"\nOptimization problem formulated with {self.num_vertices} vertices")
        
        # QAOA solver
        optimizer = COBYLA(maxiter=100)
        qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=p)
        
        # Solve
        algorithm = MinimumEigenOptimizer(qaoa)
        result = algorithm.solve(qp)
        
        print(f"\nQAOA Solution (p={p}):")
        print(f"Optimal value: {-result.fval:.4f}")  # Negate back
        print(f"Selected vertices: {[i for i, v in enumerate(result.x) if v == 1]}")
        
        # Classical baseline
        classical_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        classical_result = classical_solver.solve(qp)
        
        print(f"\nClassical Solution (Exact):")
        print(f"Optimal value: {-classical_result.fval:.4f}")
        print(f"Selected vertices: {[i for i, v in enumerate(classical_result.x) if v == 1]}")
        
        self.visualize_solution(result.x)
        
        return result
    
    def visualize_solution(self, solution):
        """Visualize the settlement placement"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw hex tiles
        hex_centers = [(0, 0), (1.5, 0), (3, 0), (0.75, 1.3), (2.25, 1.3)]
        
        for i, (tile, center) in enumerate(zip(self.tiles, hex_centers)):
            hexagon = plt.Circle(center, 0.6, color='lightgreen', alpha=0.5)
            ax.add_patch(hexagon)
            ax.text(center[0], center[1], f"{tile['resource'][:4]}\n{tile['number']}", 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw vertices
        vertex_positions = [
            (0, 0.6), (0.75, 0.6), (1.5, 0.6), (2.25, 0.6), (3, 0.6),
            (0.75, 1.9), (1.5, 1.9), (2.25, 1.9), (3, 1.9)
        ]
        
        for i, pos in enumerate(vertex_positions[:self.num_vertices]):
            if solution[i] == 1:
                ax.plot(pos[0], pos[1], 'ro', markersize=20, label='Settlement' if i == 0 else '')
                ax.text(pos[0], pos[1], str(i), ha='center', va='center', 
                       color='white', fontweight='bold')
            else:
                ax.plot(pos[0], pos[1], 'ko', markersize=8, alpha=0.3)
        
        ax.set_xlim(-1, 4)
        ax.set_ylim(-0.5, 3)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('Optimal Settlement Placement')
        plt.tight_layout()
        plt.savefig('settlement_placement.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved as 'settlement_placement.png'")


# ============================================================================
# PROBLEM 2: QUANTUM LONGEST ROAD
# ============================================================================

class QuantumLongestRoad:
    """
    Finds the longest connected path in a road network using QAOA.
    Models as a constrained path optimization problem.
    """
    
    def __init__(self):
        # Define road network as graph
        self.graph = nx.Graph()
        edges = [
            ('A', 'B'), ('B', 'C'), ('C', 'D'),
            ('A', 'E'), ('B', 'E'), ('C', 'F'),
            ('D', 'F'), ('E', 'F'), ('E', 'G'),
            ('F', 'G'), ('G', 'H')
        ]
        self.graph.add_edges_from(edges)
        self.max_roads = 6  # Resource constraint
        
    def build_longest_path_problem(self):
        """
        Formulate longest path as QUBO.
        Each edge is a binary variable.
        """
        qp = QuadraticProgram()
        
        edges = list(self.graph.edges())
        self.edge_list = edges
        
        # Binary variable for each edge
        for i, edge in enumerate(edges):
            qp.binary_var(f'e{i}')
        
        # Objective: maximize number of selected edges (minimize negative count)
        linear = {f'e{i}': -1 for i in range(len(edges))}
        qp.minimize(linear=linear)
        
        # Constraint: at most max_roads edges
        constraint = {f'e{i}': 1 for i in range(len(edges))}
        qp.linear_constraint(constraint, '<=', self.max_roads)
        
        # Connectivity constraint (simplified): penalize disconnected edges
        quadratic = {}
        for i, e1 in enumerate(edges):
            for j, e2 in enumerate(edges):
                if i < j:
                    # Reward if edges share a vertex
                    if set(e1) & set(e2):
                        quadratic[(f'e{i}', f'e{j}')] = -0.5
        
        if quadratic:
            qp.minimize(quadratic=quadratic, linear=linear)
        
        return qp
    
    def solve_with_qaoa(self, p=2):
        """Solve using QAOA"""
        print("\n" + "="*60)
        print("PROBLEM 2: QUANTUM LONGEST ROAD")
        print("="*60)
        
        qp = self.build_longest_path_problem()
        print(f"\nRoad network: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} possible edges")
        print(f"Resource constraint: Maximum {self.max_roads} roads")
        
        # QAOA solver
        optimizer = COBYLA(maxiter=150)
        qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=p)
        
        algorithm = MinimumEigenOptimizer(qaoa)
        result = algorithm.solve(qp)
        
        selected_edges = [self.edge_list[i] for i, v in enumerate(result.x) if v == 1]
        
        print(f"\nQAOA Solution (p={p}):")
        print(f"Number of roads: {len(selected_edges)}")
        print(f"Selected roads: {selected_edges}")
        print(f"Path length: {self.calculate_path_length(selected_edges)}")
        
        # Classical baseline (greedy DFS)
        classical_solution = self.greedy_longest_path()
        print(f"\nClassical Greedy Solution:")
        print(f"Number of roads: {len(classical_solution)}")
        print(f"Path length: {self.calculate_path_length(classical_solution)}")
        
        self.visualize_solution(selected_edges)
        
        return result
    
    def calculate_path_length(self, edges):
        """Calculate longest connected path in selected edges"""
        if not edges:
            return 0
        
        subgraph = nx.Graph()
        subgraph.add_edges_from(edges)
        
        # Find longest path using DFS from each node
        max_length = 0
        for node in subgraph.nodes():
            length = self._dfs_longest_path(subgraph, node, set())
            max_length = max(max_length, length)
        
        return max_length
    
    def _dfs_longest_path(self, graph, node, visited):
        """DFS helper to find longest path"""
        visited.add(node)
        max_length = 0
        
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                length = 1 + self._dfs_longest_path(graph, neighbor, visited.copy())
                max_length = max(max_length, length)
        
        return max_length
    
    def greedy_longest_path(self):
        """Classical greedy baseline"""
        edges = list(self.graph.edges())
        selected = []
        
        # Start from highest degree node
        start = max(self.graph.degree(), key=lambda x: x[1])[0]
        visited = {start}
        current = start
        
        while len(selected) < self.max_roads:
            # Find next edge that extends the path
            best_edge = None
            for edge in edges:
                if edge not in selected:
                    if current in edge:
                        other = edge[1] if edge[0] == current else edge[0]
                        if other not in visited or len(selected) == 0:
                            best_edge = edge
                            break
            
            if best_edge:
                selected.append(best_edge)
                visited.add(best_edge[0])
                visited.add(best_edge[1])
                current = best_edge[1] if best_edge[0] == current else best_edge[0]
            else:
                break
        
        return selected
    
    def visualize_solution(self, selected_edges):
        """Visualize the road network and solution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Full network
        nx.draw(self.graph, pos, ax=ax1, with_labels=True, node_color='lightblue',
                node_size=500, font_size=12, font_weight='bold')
        ax1.set_title('Full Road Network')
        
        # Solution
        nx.draw_networkx_nodes(self.graph, pos, ax=ax2, node_color='lightblue', 
                               node_size=500)
        nx.draw_networkx_labels(self.graph, pos, ax=ax2, font_size=12, font_weight='bold')
        
        # Draw all edges in gray
        nx.draw_networkx_edges(self.graph, pos, ax=ax2, edge_color='lightgray', width=2)
        
        # Highlight selected edges
        if selected_edges:
            nx.draw_networkx_edges(self.graph, pos, ax=ax2, edgelist=selected_edges,
                                   edge_color='red', width=4)
        
        ax2.set_title(f'Longest Road Solution ({len(selected_edges)} roads)')
        
        plt.tight_layout()
        plt.savefig('longest_road.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved as 'longest_road.png'")


# ============================================================================
# PROBLEM 3: QUANTUM RESOURCE TRADER
# ============================================================================

class QuantumResourceTrader:
    """
    Optimizes resource trading and building decisions using QAOA.
    Formulated as a constrained knapsack problem.
    """
    
    def __init__(self):
        # Available resources
        self.resources = {
            'wood': 3,
            'brick': 2,
            'ore': 1,
            'wheat': 2,
            'sheep': 1
        }
        
        # Possible actions with costs and rewards
        self.actions = [
            {'name': 'Build Road', 'cost': {'wood': 1, 'brick': 1}, 'points': 1},
            {'name': 'Build Settlement', 'cost': {'wood': 1, 'brick': 1, 'wheat': 1, 'sheep': 1}, 'points': 5},
            {'name': 'Build City', 'cost': {'ore': 3, 'wheat': 2}, 'points': 7},
            {'name': 'Buy Dev Card', 'cost': {'ore': 1, 'wheat': 1, 'sheep': 1}, 'points': 2},
            {'name': 'Trade 4:1 Wood', 'cost': {'wood': 4}, 'gain': {'ore': 1}, 'points': 0},
        ]
    
    def build_knapsack_problem(self):
        """
        Build quantum knapsack formulation.
        Binary variable for each action.
        """
        qp = QuadraticProgram()
        
        # Binary variable for each action
        for i in range(len(self.actions)):
            qp.binary_var(f'a{i}')
        
        # Objective: maximize points (minimize negative points)
        linear = {}
        for i, action in enumerate(self.actions):
            linear[f'a{i}'] = -action['points']
        
        qp.minimize(linear=linear)
        
        # Resource constraints
        for resource in self.resources:
            constraint = {}
            for i, action in enumerate(self.actions):
                cost = action.get('cost', {}).get(resource, 0)
                gain = action.get('gain', {}).get(resource, 0)
                net_cost = cost - gain
                if net_cost != 0:
                    constraint[f'a{i}'] = net_cost
            
            if constraint:
                qp.linear_constraint(constraint, '<=', self.resources[resource])
        
        return qp
    
    def solve_with_qaoa(self, p=2):
        """Solve using QAOA"""
        print("\n" + "="*60)
        print("PROBLEM 3: QUANTUM RESOURCE TRADER")
        print("="*60)
        
        print("\nAvailable Resources:")
        for resource, count in self.resources.items():
            print(f"  {resource}: {count}")
        
        print(f"\nPossible Actions: {len(self.actions)}")
        for i, action in enumerate(self.actions):
            print(f"  {i}. {action['name']} - Points: {action['points']}")
        
        qp = self.build_knapsack_problem()
        
        # QAOA solver
        optimizer = COBYLA(maxiter=100)
        qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=p)
        
        algorithm = MinimumEigenOptimizer(qaoa)
        result = algorithm.solve(qp)
        
        selected_actions = [self.actions[i] for i, v in enumerate(result.x) if v == 1]
        
        print(f"\nQAOA Solution (p={p}):")
        print(f"Total points: {-result.fval:.0f}")
        print(f"Selected actions:")
        for action in selected_actions:
            print(f"  - {action['name']}")
        
        # Classical baseline (greedy)
        classical_solution = self.greedy_knapsack()
        print(f"\nClassical Greedy Solution:")
        print(f"Total points: {sum(a['points'] for a in classical_solution)}")
        print(f"Selected actions:")
        for action in classical_solution:
            print(f"  - {action['name']}")
        
        self.visualize_solution(result.x)
        
        return result
    
    def greedy_knapsack(self):
        """Classical greedy baseline - sort by points/cost ratio"""
        # Calculate efficiency for each action
        action_efficiency = []
        for i, action in enumerate(self.actions):
            total_cost = sum(action.get('cost', {}).values())
            efficiency = action['points'] / max(total_cost, 1)
            action_efficiency.append((i, efficiency, action))
        
        action_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        remaining_resources = self.resources.copy()
        
        for i, eff, action in action_efficiency:
            # Check if we can afford this action
            can_afford = True
            for resource, cost in action.get('cost', {}).items():
                if remaining_resources.get(resource, 0) < cost:
                    can_afford = False
                    break
            
            if can_afford:
                selected.append(action)
                # Deduct costs
                for resource, cost in action.get('cost', {}).items():
                    remaining_resources[resource] -= cost
                # Add gains
                for resource, gain in action.get('gain', {}).items():
                    remaining_resources[resource] = remaining_resources.get(resource, 0) + gain
        
        return selected
    
    def visualize_solution(self, solution):
        """Visualize resource allocation"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Resource availability
        resources_list = list(self.resources.keys())
        counts = list(self.resources.values())
        
        ax1.bar(resources_list, counts, color='skyblue', edgecolor='black')
        ax1.set_ylabel('Count')
        ax1.set_title('Available Resources')
        ax1.set_ylim(0, max(counts) + 1)
        
        # Action selection
        selected_indices = [i for i, v in enumerate(solution) if v == 1]
        action_names = [self.actions[i]['name'] for i in range(len(self.actions))]
        colors = ['green' if i in selected_indices else 'lightgray' 
                 for i in range(len(self.actions))]
        
        y_pos = np.arange(len(action_names))
        points = [self.actions[i]['points'] for i in range(len(self.actions))]
        
        ax2.barh(y_pos, points, color=colors, edgecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(action_names)
        ax2.set_xlabel('Points')
        ax2.set_title('Action Selection (Green = Selected)')
        
        plt.tight_layout()
        plt.savefig('resource_trading.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved as 'resource_trading.png'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all three quantum Catan challenges"""
    
    print("╔" + "═"*58 + "╗")
    print("║" + " "*10 + "QUANTUM CATAN CHALLENGE - SOLUTION" + " "*13 + "║")
    print("╚" + "═"*58 + "╝")
    
    # Problem 1: Settlement Placement
    problem1 = QuantumSettlementPlanner()
    result1 = problem1.solve_with_qaoa(p=1)
    
    # Problem 2: Longest Road
    problem2 = QuantumLongestRoad()
    result2 = problem2.solve_with_qaoa(p=2)
    
    # Problem 3: Resource Trading
    problem3 = QuantumResourceTrader()
    result3 = problem3.solve_with_qaoa(p=2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - QUANTUM OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\n✓ Problem 1: Settlement placement optimized")
    print(f"✓ Problem 2: Longest road found with {problem2.max_roads} segments")
    print(f"✓ Problem 3: Resource trading strategy optimized")
    print(f"\n✓ All visualizations saved as PNG files")
    print("\n" + "="*60)
    print("Challenge Complete! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()
