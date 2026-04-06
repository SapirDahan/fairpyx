"""
An implementation of the algorithm in:
"Fair Division Under Cardinality Constraints", by A. Biswas, S. Barman (2018), https://arxiv.org/abs/1804.09521
Programmer: Sapir Dahan
Date : 2026-04
"""

import math
import logging
import networkz as nz
from fairpyx import Instance, AllocationBuilder

logger = logging.getLogger(__name__)



def fair_division_under_cardinality_constraints(
    alloc: AllocationBuilder,
    item_categories: dict,
    category_capacities: dict,
    initial_agent_order: list = None,
):
    """
    Compute a feasible EF1 allocation under cardinality constraints (Algorithm 1).

    All agents share the same per-category threshold k_h. Valuations are additive and
    may differ across agents. The algorithm guarantees EF1: for every pair of agents
    (i, j) there exists some good g in j's bundle such that v_i(A_i) >= v_i(A_j \\ {g}).

    Procedure:
      1. Validate all inputs via validate_fair_division_inputs.
      2. For each category h in item_categories (in iteration order):
         a. Call greedy_round_robin to allocate the goods in h using the current agent order.
         b. Call eliminate_envy_cycles to remove all directed cycles from the envy graph and
            obtain an acyclic envy graph G_h.
         c. Derive the agent order for the next category as the topological sort of G_h
            (agents with no incoming envy edges pick first in the next round).

    :param alloc: an allocation builder, which tracks the allocation and the remaining
        capacity for items and agents.
    :param item_categories: a dictionary mapping each category name (str) to a list of
        item names. Every item in the instance must appear in exactly one category.
        Example: {'c1': ['m1', 'm2'], 'c2': ['m3', 'm4', 'm5']}.
    :param category_capacities: a dictionary mapping each category name (str) to a positive
        integer threshold k_h — the maximum number of goods each agent may receive from that
        category. All agents share the same k_h (required by the paper).
        Example: {'c1': 1, 'c2': 2}.
    :param initial_agent_order: a list of agent names specifying the initial picking order
        for the first category. Must contain each agent exactly once. If None (default),
        agents are sorted lexicographically, giving a fully deterministic result.

    >>> # Example 1: basic — 2 agents, 2 categories, 1 good each, k_h=1
    >>> # Agent1 prefers m1 and m3; Agent2 prefers m2 and m4.
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'Agent1': {'m1': 9, 'm2': 3, 'm3': 8, 'm4': 2},
    ...     'Agent2': {'m1': 3, 'm2': 9, 'm3': 2, 'm4': 8},
    ... }
    >>> item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3', 'm4']}
    >>> category_capacities = {'c1': 1, 'c2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])
    {'Agent1': ['m1', 'm3'], 'Agent2': ['m2', 'm4']}

    >>> # Example 1b: same setup, no initial_agent_order — defaults to sorted(agents) = ['Agent1', 'Agent2']
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities)
    {'Agent1': ['m1', 'm3'], 'Agent2': ['m2', 'm4']}

    >>> # Example 2: 3 agents, 2 categories of 3 goods each, k_h=1
    >>> # Each agent's top good in each category is unique, so the output is deterministic.
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'Agent1': {'m1': 9, 'm2': 6, 'm3': 3, 'm4': 8, 'm5': 5, 'm6': 2},
    ...     'Agent2': {'m1': 3, 'm2': 9, 'm3': 6, 'm4': 2, 'm5': 8, 'm6': 5},
    ...     'Agent3': {'m1': 6, 'm2': 3, 'm3': 9, 'm4': 5, 'm5': 2, 'm6': 8},
    ... }
    >>> item_categories = {'c1': ['m1', 'm2', 'm3'], 'c2': ['m4', 'm5', 'm6']}
    >>> category_capacities = {'c1': 1, 'c2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2', 'Agent3'])
    {'Agent1': ['m1', 'm4'], 'Agent2': ['m2', 'm5'], 'Agent3': ['m3', 'm6']}

    >>> # Example 3: single agent — trivially receives all goods (up to its capacity)
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Alice': {'m1': 10, 'm2': 7, 'm3': 4}}
    >>> item_categories = {'c1': ['m1', 'm2', 'm3']}
    >>> category_capacities = {'c1': 3}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Alice'])
    {'Alice': ['m1', 'm2', 'm3']}

    >>> # Example 4: single category — degenerates to a single greedy_round_robin call
    >>> from fairpyx import Instance, divide
    >>> valuations = {'A': {'x': 10, 'y': 5, 'z': 1}, 'B': {'x': 1, 'y': 5, 'z': 10}}
    >>> item_categories = {'c1': ['x', 'y', 'z']}
    >>> category_capacities = {'c1': 2}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['A', 'B'])
    {'A': ['x', 'y'], 'B': ['z']}

    >>> # Example 5: single good per category — each agent can receive at most one good per category
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': 8, 'm2': 5}, 'Agent2': {'m1': 5, 'm2': 8}}
    >>> item_categories = {'c1': ['m1'], 'c2': ['m2']}
    >>> category_capacities = {'c1': 1, 'c2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])
    {'Agent1': ['m1'], 'Agent2': ['m2']}

    >>> # Example 6: asymmetric category sizes — c1 has 4 goods (k_h=2), c2 has 1 good (k_h=1)
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'Agent1': {'m1': 10, 'm2': 8, 'm3': 6, 'm4': 4, 'm5': 9},
    ...     'Agent2': {'m1': 4,  'm2': 6, 'm3': 8, 'm4': 10, 'm5': 7},
    ... }
    >>> item_categories = {'c1': ['m1', 'm2', 'm3', 'm4'], 'c2': ['m5']}
    >>> category_capacities = {'c1': 2, 'c2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])
    {'Agent1': ['m1', 'm2', 'm5'], 'Agent2': ['m3', 'm4']}

    >>> # Example 7: minimal base case — 1 agent, 1 good, 1 category.
    >>> # σ = ['a1'].  C1 round-robin: a1 picks g1 (only good). No envy possible.
    >>> # Final allocation: {a1: [g1]}.
    >>> from fairpyx import Instance, divide
    >>> valuations = {'a1': {'g1': 2}}
    >>> item_categories = {'C1': ['g1']}
    >>> category_capacities = {'C1': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1'])
    {'a1': ['g1']}

    >>> # Example 8: 2 agents, 2 goods, 1 category, k_h=1 - full example explanation
    >>> # σ = ['a1','a2'].
    >>> # C1 round-robin:
    >>> #   a1 picks best of {g1,g2}: values 2,3 → picks g2 (value 3).
    >>> #   a2 picks best of {g1}:    value  1   → picks g1 (only remaining).
    >>> # After C1: {a1:[g2], a2:[g1]}.
    >>> #
    >>> # Envy table (row = whose eyes, col = whose bundle):
    >>> #              a1's bundle   a2's bundle
    >>> #   a1 values:    3             2        → a1 does NOT envy a2 (3 >= 2).
    >>> #   a2 values:    4             1        → a2 ENVIES a1 (4 > 1). Edge: a2→a1.
    >>> #
    >>> # Envy graph: a2→a1. No cycle. Topo sort: ['a2','a1'].
    >>> # (Only one category, so updated σ is never used.)
    >>> # Final allocation: {a1:[g2], a2:[g1]}.
    >>> from fairpyx import Instance, divide
    >>> valuations = {'a1': {'g1': 2, 'g2': 3}, 'a2': {'g1': 1, 'g2': 4}}
    >>> item_categories = {'C1': ['g1', 'g2']}
    >>> category_capacities = {'C1': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1', 'a2'])
    {'a1': ['g2'], 'a2': ['g1']}

    >>> # Example 9: 2 agents, 6 goods, 2 categories, k_h=2 - full example explanation
    >>> # Key feature: a2 envies a1 after C1, so the agent order REVERSES for C2,
    >>> # giving a2 the first pick in the second category.
    >>> #
    >>> # σ = ['a1','a2'].
    >>> # C1 round-robin (each agent picks at most k_h=2 goods):
    >>> #   a1 picks best of {g1,g2,g3}: values 9,6,3 → g1 (value 9).
    >>> #   a2 picks best of {g2,g3}:    values 7,8   → g3 (value 8).
    >>> #   a1 picks best of {g2}:        value  6    → g2 (only remaining).
    >>> # After C1: {a1:[g1,g2], a2:[g3]}.
    >>> #
    >>> # Envy table after C1:
    >>> #              a1's bundle {g1,g2}   a2's bundle {g3}
    >>> #   a1 values:     9+6=15                3         → a1 does NOT envy a2.
    >>> #   a2 values:     4+7=11                8         → a2 ENVIES a1 (11>8). Edge: a2→a1.
    >>> #
    >>> # Envy graph: a2→a1. No cycle. Topo sort: a2 has no incoming edge → first.
    >>> # σ updated to ['a2','a1'] for C2.
    >>> #
    >>> # C2 round-robin with σ=['a2','a1']:
    >>> #   a2 picks best of {g4,g5,g6}: values 3,9,6 → g5 (value 9).
    >>> #   a1 picks best of {g4,g6}:    values 8,2   → g4 (value 8).
    >>> #   a2 picks best of {g6}:        value  6    → g6 (only remaining).
    >>> # After C2: {a1:[g1,g2,g4], a2:[g3,g5,g6]}.
    >>> #
    >>> # Final envy table:
    >>> #              a1's bundle {g1,g2,g4}   a2's bundle {g3,g5,g6}
    >>> #   a1 values:    9+6+8=23                  3+5+2=10   → a1 does NOT envy a2.
    >>> #   a2 values:    4+7+3=14                  8+9+6=23   → a2 does NOT envy a1.
    >>> # No envy. Final allocation below.
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'a1': {'g1': 9, 'g2': 6, 'g3': 3, 'g4': 8, 'g5': 5, 'g6': 2},
    ...     'a2': {'g1': 4, 'g2': 7, 'g3': 8, 'g4': 3, 'g5': 9, 'g6': 6},
    ... }
    >>> item_categories = {'C1': ['g1', 'g2', 'g3'], 'C2': ['g4', 'g5', 'g6']}
    >>> category_capacities = {'C1': 2, 'C2': 2}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1', 'a2'])
    {'a1': ['g1', 'g2', 'g4'], 'a2': ['g3', 'g5', 'g6']}

    >>> # Example 10: 3 agents, 6 goods, 2 categories, k_h=1 - full example explanation
    >>> # Key feature: after C2, the envy graph contains MULTIPLE OVERLAPPING CYCLES,
    >>> # requiring eliminate_envy_cycles to resolve them iteratively.
    >>> #
    >>> # σ = ['a1','a2','a3'].
    >>> # C1 round-robin:
    >>> #   a1 picks best of {g1,g2,g3}: values 9,2,1 → g1.
    >>> #   a2 picks best of {g2,g3}:    values 8,1   → g2.
    >>> #   a3 picks best of {g3}:        value  5    → g3.
    >>> # After C1: {a1:[g1], a2:[g2], a3:[g3]}.
    >>> #
    >>> # Envy table after C1:
    >>> #              a1 {g1}   a2 {g2}   a3 {g3}
    >>> #   a1 values:    9         2         1      → a1 does NOT envy anyone.
    >>> #   a2 values:   10         8         1      → a2 ENVIES a1 (10>8). Edge: a2→a1.
    >>> #   a3 values:   10         9         5      → a3 ENVIES a1 and a2. Edges: a3→a1, a3→a2.
    >>> #
    >>> # Envy graph: a2→a1, a3→a1, a3→a2. No cycle (DAG). Topo sort: ['a3','a2','a1'].
    >>> # σ updated to ['a3','a2','a1'] for C2.
    >>> #
    >>> # C2 round-robin with σ=['a3','a2','a1']:
    >>> #   a3 picks best of {g4,g5,g6}: values 8,1,7 → g4.
    >>> #   a2 picks best of {g5,g6}:    values 1,0   → g5.
    >>> #   a1 picks best of {g6}:        value  0    → g6.
    >>> # After C2: {a1:[g1,g6], a2:[g2,g5], a3:[g3,g4]}.
    >>> #
    >>> # Envy table after C2:
    >>> #              a1 {g1,g6}   a2 {g2,g5}   a3 {g3,g4}
    >>> #   a1 values:  9+0=9       2+10=12       1+1=2      → a1 ENVIES a2. Edge: a1→a2.
    >>> #   a2 values: 10+0=10      8+1=9        1+10=11     → a2 ENVIES a1 and a3. Edges: a2→a1, a2→a3.
    >>> #   a3 values: 10+7=17      9+1=10        5+8=13     → a3 ENVIES a1. Edge: a3→a1.
    >>> #
    >>> # Envy graph edges: a1→a2, a2→a1, a2→a3, a3→a1.
    >>> # This graph contains TWO overlapping simple cycles:
    >>> #   - 2-cycle: a1↔a2        (edges a1→a2 and a2→a1)
    >>> #   - 3-cycle: a1→a2→a3→a1 (edges a1→a2, a2→a3, a3→a1)
    >>> #
    >>> # Both elimination orderings are valid and both lead to the SAME final allocation
    >>> # (verified by tracing both paths):
    >>> #
    >>> #   PATH A — eliminate 3-cycle first:
    >>> #     Rotate: a1 ← a2's bundle, a2 ← a3's bundle, a3 ← a1's bundle.
    >>> #     After rotation: {a1:[g2,g5], a2:[g3,g4], a3:[g1,g6]}.
    >>> #     Verify: a1 own=12, a2's=9, a3's=9 → no envy.
    >>> #             a2 own=11, a1's=9, a3's=10 → no envy.
    >>> #             a3 own=17, a1's=10, a2's=13 → no envy. Done.
    >>> #
    >>> #   PATH B — eliminate 2-cycle first, then the resulting 2-cycle:
    >>> #     Step 1: swap a1↔a2 → {a1:[g2,g5], a2:[g1,g6], a3:[g3,g4]}.
    >>> #       New envy: a2 values a3's bundle at 11 > own 10, and a3 values a2's at 17 > own 13.
    >>> #       New 2-cycle: a2↔a3.
    >>> #     Step 2: swap a2↔a3 → {a1:[g2,g5], a2:[g3,g4], a3:[g1,g6]}.
    >>> #       No remaining envy. Done.
    >>> #
    >>> # Both paths produce the SAME EF1 allocation for this specific instance — verified above.
    >>> # Note: in general, different cycle orderings CAN lead to different valid EF1 allocations
    >>> # (the paper guarantees EF1 for any order, but not uniqueness). For instances where the
    >>> # two orderings diverge, use: assert result in [answer_path_A, answer_path_B].
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'a1': {'g1': 9, 'g2': 2, 'g3': 1, 'g4': 1, 'g5': 10, 'g6': 0},
    ...     'a2': {'g1': 10, 'g2': 8, 'g3': 1, 'g4': 10, 'g5': 1, 'g6': 0},
    ...     'a3': {'g1': 10, 'g2': 9, 'g3': 5, 'g4': 8, 'g5': 1, 'g6': 7},
    ... }
    >>> item_categories = {'C1': ['g1', 'g2', 'g3'], 'C2': ['g4', 'g5', 'g6']}
    >>> category_capacities = {'C1': 1, 'C2': 1}
    >>> instance = Instance(valuations=valuations)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1', 'a2', 'a3'])
    {'a1': ['g2', 'g5'], 'a2': ['g3', 'g4'], 'a3': ['g1', 'g6']}

    >>> # Example 11: 4 agents, 26 goods, 4 categories (5/6/7/8 goods), k_h=2 - large example
    >>> # This is the largest and most complex run example. Two order changes occur:
    >>> #   - After C1: a3 envies a1 → σ becomes ['a3','a1','a2','a4'].
    >>> #   - After C2: a2 envies a3 → σ becomes ['a2','a1','a3','a4'].
    >>> #   - After C3: no envy     → σ remains  ['a1','a2','a3','a4'].
    >>> #
    >>> # C1=[g1..g5], C2=[g6..g11], C3=[g12..g18], C4=[g19..g26]; k_h=2 for all.
    >>> #
    >>> # C1 round-robin σ=['a1','a2','a3','a4'] (5 goods, 2 picks each — only 4 agents × 2=8 slots but only 5 goods):
    >>> #   a1 picks g1 (9), a2 picks g2 (10), a3 picks g4 (9), a4 picks g3 (10), a1 picks g5 (6).
    >>> # After C1: {a1:[g1,g5], a2:[g2], a3:[g4], a4:[g3]}.
    >>> #
    >>> # Envy table after C1 (values shown as agent_i sees bundle_j):
    >>> #       a1{g1,g5}  a2{g2}  a3{g4}  a4{g3}
    >>> #   a1:    15        4       1       8     → a1 does not envy anyone (15 is max).
    >>> #   a2:     8       10       7       2     → a2 does not envy anyone (10 is max).
    >>> #   a3:    10        1       9       5     → a3 ENVIES a1 (10>9). Edge: a3→a1.
    >>> #   a4:     9        6       3      10     → a4 does not envy anyone (10 is max).
    >>> # No cycle. Topo sort: a3 first. σ = ['a3','a1','a2','a4'].
    >>> #
    >>> # C2 round-robin σ=['a3','a1','a2','a4'] (6 goods):
    >>> #   a3 picks g8 (10), a1 picks g7 (10), a2 picks g6 (9), a4 picks g9 (10),
    >>> #   a3 picks g10 (7), a1 picks g11 (7).
    >>> # After C2: {a1:[g1,g5,g7,g11], a2:[g2,g6], a3:[g4,g8,g10], a4:[g3,g9]}.
    >>> #
    >>> # Envy table after C2:
    >>> #       a1{..}  a2{g2,g6}  a3{g4,g8,g10}  a4{g3,g9}
    >>> #   a1:   32       6          5              13      → a1 does not envy (32 is max).
    >>> #   a2:   11      19         21               6      → a2 ENVIES a3 (21>19). Edge: a2→a3.
    >>> #   a3:   19       5         26              12      → a3 does not envy (26 is max).
    >>> #   a4:   15      11          6              20      → a4 does not envy (20 is max).
    >>> # No cycle. Topo sort: a2 first (only outgoing edge). σ = ['a2','a1','a3','a4'].
    >>> #
    >>> # C3 round-robin σ=['a2','a1','a3','a4'] (7 goods):
    >>> #   a2 picks g15 (10), a1 picks g14 (9), a3 picks g17 (10), a4 picks g18 (10),
    >>> #   a2 picks g13 (7), a1 picks g16 (8), a3 picks g12 (9).
    >>> # After C3: {a1:[g1,g5,g7,g11,g14,g16], a2:[g2,g6,g15,g13], a3:[g4,g8,g10,g17,g12], a4:[g3,g9,g18]}.
    >>> #
    >>> # Envy table after C3: no agent envies another. σ remains ['a1','a2','a3','a4'].
    >>> #
    >>> # C4 round-robin σ=['a1','a2','a3','a4'] (8 goods):
    >>> #   a1 picks g22 (10), a2 picks g25 (10), a3 picks g23 (9), a4 picks g24 (10),
    >>> #   a1 picks g26 (8), a2 picks g21 (9), a3 picks g19 (8), a4 picks g20 (8).
    >>> # After C4: {a1:[g1,g5,g7,g11,g14,g16,g22,g26], a2:[g2,g6,g15,g13,g25,g21],
    >>> #            a3:[g4,g8,g10,g17,g12,g23,g19], a4:[g3,g9,g18,g24,g20]}.
    >>> #
    >>> # Final envy table: no envy among any agents. EF1 holds.
    >>> # NOTE: marked SKIP — the exact output depends on lex tie-breaking across 26 goods.
    >>> from fairpyx import Instance, divide  # doctest: +SKIP
    >>> all_goods = [f'g{i}' for i in range(1, 27)]  # doctest: +SKIP
    >>> valuations = {  # doctest: +SKIP
    ...     'a1': {'g1':9,'g2':4,'g3':8,'g4':1,'g5':6,'g6':2,'g7':10,'g8':3,'g9':5,'g10':1,'g11':7,
    ...            'g12':6,'g13':2,'g14':9,'g15':4,'g16':8,'g17':1,'g18':3,
    ...            'g19':5,'g20':7,'g21':2,'g22':10,'g23':4,'g24':6,'g25':1,'g26':8},
    ...     'a2': {'g1':3,'g2':10,'g3':2,'g4':7,'g5':5,'g6':9,'g7':1,'g8':8,'g9':4,'g10':6,'g11':2,
    ...            'g12':1,'g13':7,'g14':3,'g15':10,'g16':5,'g17':8,'g18':4,
    ...            'g19':6,'g20':2,'g21':9,'g22':1,'g23':7,'g24':3,'g25':10,'g26':5},
    ...     'a3': {'g1':8,'g2':1,'g3':5,'g4':9,'g5':2,'g6':4,'g7':6,'g8':10,'g9':1,'g10':7,'g11':3,
    ...            'g12':9,'g13':5,'g14':2,'g15':6,'g16':1,'g17':10,'g18':4,
    ...            'g19':8,'g20':3,'g21':6,'g22':2,'g23':9,'g24':1,'g25':5,'g26':7},
    ...     'a4': {'g1':2,'g2':6,'g3':10,'g4':3,'g5':7,'g6':5,'g7':2,'g8':1,'g9':10,'g10':8,'g11':4,
    ...            'g12':3,'g13':9,'g14':6,'g15':2,'g16':7,'g17':5,'g18':10,
    ...            'g19':1,'g20':8,'g21':4,'g22':6,'g23':2,'g24':10,'g25':3,'g26':9},
    ... }
    >>> item_categories = {  # doctest: +SKIP
    ...     'C1': ['g1','g2','g3','g4','g5'],
    ...     'C2': ['g6','g7','g8','g9','g10','g11'],
    ...     'C3': ['g12','g13','g14','g15','g16','g17','g18'],
    ...     'C4': ['g19','g20','g21','g22','g23','g24','g25','g26'],
    ... }
    >>> category_capacities = {'C1': 2, 'C2': 2, 'C3': 2, 'C4': 2}  # doctest: +SKIP
    >>> instance = Instance(valuations=valuations)  # doctest: +SKIP
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,  # doctest: +SKIP
    ...        item_categories=item_categories, category_capacities=category_capacities,
    ...        initial_agent_order=['a1', 'a2', 'a3', 'a4'])
    {'a1': ['g1', 'g11', 'g14', 'g16', 'g22', 'g26', 'g5', 'g7'], 'a2': ['g13', 'g15', 'g2', 'g21', 'g25', 'g6'], 'a3': ['g10', 'g12', 'g17', 'g19', 'g23', 'g4', 'g8'], 'a4': ['g18', 'g20', 'g24', 'g3', 'g9']}

    # Invalid-input tests (negative valuations, k_h too small, duplicate goods across
    # categories, non-positive k_h, empty/duplicate initial_agent_order, uncategorised
    # goods) are covered in full by the doctests of validate_fair_division_inputs.
    """


    pass


def greedy_round_robin(
    alloc: AllocationBuilder,
    items_in_category: list,
    agent_order: list,
    category_capacities: dict,
    category: str,
) -> None:
    """
    Allocate all goods from a single category using Greedy Round-Robin (Algorithm 2).

    Agents pick in round-robin order, cycling repeatedly. On each turn an agent greedily
    selects the remaining good in the category it values most, subject to its per-category
    capacity k_h. An agent is skipped once it has received k_h goods from this category.
    The procedure terminates when all goods are allocated or all agents have reached capacity.

    :param alloc: an allocation builder, which tracks the allocation and the remaining
        capacity for items and agents.
    :param items_in_category: the list of items belonging to the category being processed
        in this round.
    :param agent_order: the ordered list of agents specifying the picking sequence for this
        category (either initial_agent_order or the topological sort from the previous category).
    :param category_capacities: a dictionary mapping each category name (str) to the shared
        integer threshold k_h for that category.
    :param category: the name (str) of the category currently being allocated.

    >>> # Example 1: 2 agents, 2 goods, k_h=1 — each agent picks their top good
    >>> from fairpyx import Instance, AllocationBuilder
    >>> valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 8}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> category_capacities = {'c1': 1}
    >>> greedy_round_robin(alloc, ['m1', 'm2'], ['Alice', 'Bob'], category_capacities, 'c1')
    >>> alloc.sorted()
    {'Alice': ['m1'], 'Bob': ['m2']}

    >>> # Example 2: 3 agents, 3 goods, k_h=1 — each picks their unique top good in order
    >>> valuations = {'A': {'m1': 9, 'm2': 5, 'm3': 1},
    ...               'B': {'m1': 3, 'm2': 9, 'm3': 2},
    ...               'C': {'m1': 2, 'm2': 4, 'm3': 8}}
    >>> category_capacities = {'c1': 1}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> greedy_round_robin(alloc, ['m1', 'm2', 'm3'], ['A', 'B', 'C'], category_capacities, 'c1')
    >>> alloc.sorted()
    {'A': ['m1'], 'B': ['m2'], 'C': ['m3']}

    >>> # Example 3: 2 agents, 4 goods, k_h=2 — each agent picks 2 goods
    >>> valuations = {'Alice': {'m1': 10, 'm2': 8, 'm3': 5, 'm4': 3},
    ...               'Bob':   {'m1': 3,  'm2': 5, 'm3': 8, 'm4': 10}}
    >>> category_capacities = {'c1': 2}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> greedy_round_robin(alloc, ['m1', 'm2', 'm3', 'm4'], ['Alice', 'Bob'], category_capacities, 'c1')
    >>> alloc.sorted()
    {'Alice': ['m1', 'm2'], 'Bob': ['m3', 'm4']}
    """

    pass


def eliminate_envy_cycles(alloc: AllocationBuilder) -> nz.DiGraph:
    """
    Build the envy graph for the current (partial) allocation, eliminate all directed cycles
    by rotating bundles along each cycle, and return the resulting acyclic envy graph (Lemma 1).

    Envy relation: agent i envies agent j if
        sum_{g in A_j} v_i(g) > sum_{g in A_i} v_i(g).
    A directed edge i -> j is added to the graph when i envies j.

    Cycle elimination: for a detected cycle (a_1, a_2, ..., a_r), rotate bundles so that
    a_1 receives a_2's bundle, a_2 receives a_3's bundle, ..., a_r receives a_1's bundle.
    The paper proves that no agent's value decreases under such a rotation. This is repeated
    until the graph contains no directed cycles (i.e., it is a DAG).

    The returned DAG is passed to nz.topological_sort to derive the agent order for the next
    category: agents with no incoming envy edges (nobody envies them) pick first.

    :param alloc: an allocation builder whose alloc.bundles reflect the current partial
        allocation after the most recent greedy_round_robin call.
    :return: a networkz DiGraph representing the acyclic envy graph after all cycles have
        been eliminated.

    >>> # Example 1: no envy — graph is already a DAG, bundles are unchanged
    >>> from fairpyx import Instance, AllocationBuilder
    >>> import networkz as nz
    >>> valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 9}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('Alice', 'm1')
    >>> alloc.give('Bob', 'm2')
    >>> G = eliminate_envy_cycles(alloc)
    >>> list(nz.simple_cycles(G))
    []
    >>> sorted(alloc.bundles['Alice'])
    ['m1']
    >>> sorted(alloc.bundles['Bob'])
    ['m2']

    >>> # Example 2: 2-agent envy cycle — bundles are swapped to eliminate the cycle
    >>> # Alice has m1 (she values at 3), Bob has m2 (he values at 3).
    >>> # Alice envies Bob (values m2=7 > m1=3) and Bob envies Alice (values m1=7 > m2=3).
    >>> valuations = {'Alice': {'m1': 3, 'm2': 7}, 'Bob': {'m1': 7, 'm2': 3}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('Alice', 'm1')
    >>> alloc.give('Bob', 'm2')
    >>> G = eliminate_envy_cycles(alloc)
    >>> list(nz.simple_cycles(G))
    []
    >>> sorted(alloc.bundles['Alice'])
    ['m2']
    >>> sorted(alloc.bundles['Bob'])
    ['m1']

    >>> # Example 3: 3-agent cycle — bundle rotation resolves all envy
    >>> # A has m1, B has m2, C has m3.
    >>> # A envies B (5>3), B envies C (5>3), C envies A (5>3) — a 3-cycle.
    >>> # After rotation: A gets m2, B gets m3, C gets m1 — no agent envies another.
    >>> valuations = {'A': {'m1': 3, 'm2': 5, 'm3': 1},
    ...               'B': {'m1': 1, 'm2': 3, 'm3': 5},
    ...               'C': {'m1': 5, 'm2': 1, 'm3': 3}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('A', 'm1')
    >>> alloc.give('B', 'm2')
    >>> alloc.give('C', 'm3')
    >>> G = eliminate_envy_cycles(alloc)
    >>> list(nz.simple_cycles(G))
    []
    >>> sorted(alloc.bundles['A'])
    ['m2']
    >>> sorted(alloc.bundles['B'])
    ['m3']
    >>> sorted(alloc.bundles['C'])
    ['m1']
    """

    return nz.DiGraph()  # placeholder so callers see a DiGraph, not None



def validate_fair_division_inputs(
    alloc: AllocationBuilder,
    item_categories: dict,
    category_capacities: dict,
    initial_agent_order: list,
):
    """
    Validate all inputs for fair_division_under_cardinality_constraints before the algorithm runs.

    Checks performed (in order):

    Type checks (correct input format):
    1. initial_agent_order is either None (use sorted agents as default) or a list.
    2. item_categories is a dict (correct type).
    3. category_capacities is a dict (correct type).
    4. item_categories values are all lists (correct type).

    Existence/non-empty checks:
    5. At least one agent exists in the instance.
    6. At least one category exists in item_categories.
    7. Every category in item_categories has at least one item (no empty category lists).

    Consistency checks (inputs match each other):
    8. If initial_agent_order is not None, it must be a valid permutation of the agents —
       contains each agent exactly once (no duplicates, no missing agents, no extra agents
       not in the instance). If None, this check is skipped.
    9. category_capacities keys match item_categories keys exactly (every category has a
       threshold, no threshold for a non-existent category).
    10. Every item listed in item_categories appears in the instance valuations.
    11. Every item in the instance valuations appears in at least one category (no uncategorised goods).
    12. No item appears in more than one category.

    Mathematical/feasibility checks:
    13. All valuation values are non-negative (>= 0).
    14. All thresholds k_h in category_capacities are positive integers.
    15. Each k_h satisfies k_h >= ceil(|C_h| / n) — the feasibility condition from the paper.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity
        for items and agents.
    :param item_categories: a dictionary mapping each category name (str) to a list of item
        names belonging to that category. Example: {'c1': ['m1', 'm2'], 'c2': ['m3']}.
    :param category_capacities: a dictionary mapping each category name (str) to the shared
        integer threshold k_h (max goods any agent may receive from that category).
        Example: {'c1': 1, 'c2': 2}.
    :param initial_agent_order: a list of agent names specifying the initial picking order,
        or None to use sorted(agents) as the default order.
        If a list, must contain each agent exactly once.
    :raises ValueError: if any of the above conditions are violated.

    # -------------------------------------------------------------------
    # Setup: base instance used across most tests
    # -------------------------------------------------------------------

    >>> from fairpyx import Instance, AllocationBuilder
    >>> valuations = {'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}}
    >>> instance = Instance(valuations=valuations)
    >>> alloc = AllocationBuilder(instance)
    >>> item_categories = {'c1': ['m1', 'm2']}
    >>> category_capacities = {'c1': 1}

    >>> # Valid inputs with explicit agent order — no exception raised
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Bob'])

    >>> # Valid inputs with None — uses sorted agents as default, no exception raised
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, None)

    # -------------------------------------------------------------------
    # Check 1: initial_agent_order is either None or a list (correct type).
    # -------------------------------------------------------------------

    >>> # [check 1] invalid — initial_agent_order is a tuple instead of a list raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ('Alice', 'Bob'))
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 1] invalid — initial_agent_order is a set instead of a list raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, {'Alice', 'Bob'})
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 2: item_categories is a dict (correct type).
    # -------------------------------------------------------------------

    >>> # [check 2] invalid — item_categories is a list instead of a dict raises ValueError
    >>> validate_fair_division_inputs(alloc, [('c1', ['m1', 'm2'])], category_capacities, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 2] invalid — item_categories is None instead of a dict raises ValueError
    >>> validate_fair_division_inputs(alloc, None, category_capacities, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 3: category_capacities is a dict (correct type).
    # -------------------------------------------------------------------

    >>> # [check 3] invalid — category_capacities is a list instead of a dict raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, [('c1', 1)], ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 3] invalid — category_capacities is None instead of a dict raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, None, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 4: item_categories values are all lists (correct type).
    # -------------------------------------------------------------------

    >>> # [check 4] invalid — a category value is a tuple instead of a list raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ('m1', 'm2')}, category_capacities, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 4] invalid — a category value is a set instead of a list raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': {'m1', 'm2'}}, category_capacities, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 5] invalid — initial_agent_order is empty list (no agents given) raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, [])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 5] invalid — instance has empty valuations (no agents, no items), order is None raises ValueError
    >>> empty_instance = Instance(valuations={})
    >>> empty_alloc = AllocationBuilder(empty_instance)
    >>> validate_fair_division_inputs(empty_alloc, item_categories, category_capacities, None)
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 6: At least one category exists in item_categories.
    # -------------------------------------------------------------------

    >>> # [check 6] invalid — item_categories is empty (no categories at all) raises ValueError
    >>> validate_fair_division_inputs(alloc, {}, {}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 7: Every category in item_categories has at least one item.
    # -------------------------------------------------------------------

    >>> # [check 7] invalid — category 'c1' has an empty item list raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': []}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 7] invalid — one category is empty, one is not raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1', 'm2'], 'c2': []}, {'c1': 1, 'c2': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 8: initial_agent_order is a valid permutation of the agents (skipped if None).
    # -------------------------------------------------------------------

    >>> # [check 8] valid — None skips the permutation check entirely, no exception raised
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, None)

    >>> # [check 8] invalid — agent missing from initial_agent_order raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 8] invalid — duplicate agent in initial_agent_order raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Alice'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 8] invalid — agent in initial_agent_order not in instance raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Charlie'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 8] invalid — agent in initial_agent_order not in instance raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, category_capacities, ['Alice', 'Bob', 'Charlie'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 9: category_capacities keys match item_categories keys exactly.
    # -------------------------------------------------------------------

    >>> # [check 9] invalid — category_capacities missing a threshold for an existing category raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1'], 'c2': ['m2']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 9] invalid — category_capacities has a key for a category not in item_categories raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': 1, 'c99': 2}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 9] invalid — category_capacities has a key for a category not in item_categories raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': 1, 'c2': 2, 'c99': 2}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 10: Every item listed in item_categories appears in the instance valuations.
    # -------------------------------------------------------------------

    >>> # [check 10] invalid — item 'm99' is in item_categories but not in instance valuations raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1', 'm99']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 11: Every item in the instance valuations appears in at least one category.
    # -------------------------------------------------------------------

    >>> # [check 11] invalid — item 'm2' is in the instance valuations but missing from all categories raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 12: No item appears in more than one category.
    # -------------------------------------------------------------------

    >>> # [check 12] invalid — item 'm1' appears in two categories raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1', 'm2'], 'c2': ['m1']}, {'c1': 1, 'c2': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 13: All valuation values are non-negative (>= 0).
    # -------------------------------------------------------------------

    >>> # [check 13] valid — valuation of exactly 0 is allowed, no exception raised
    >>> valuations_zero = {'Alice': {'m1': 0, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}}
    >>> instance_zero = Instance(valuations=valuations_zero)
    >>> alloc_zero = AllocationBuilder(instance_zero)
    >>> validate_fair_division_inputs(alloc_zero, {'c1': ['m1', 'm2']}, {'c1': 1}, ['Alice', 'Bob'])

    >>> # [check 13] invalid — negative valuation raises ValueError
    >>> valuations_neg = {'Alice': {'m1': -1, 'm2': 3}, 'Bob': {'m1': 4, 'm2': 6}}
    >>> instance_neg = Instance(valuations=valuations_neg)
    >>> alloc_neg = AllocationBuilder(instance_neg)
    >>> validate_fair_division_inputs(alloc_neg, {'c1': ['m1', 'm2']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 14: All thresholds k_h in category_capacities are positive integers.
    # -------------------------------------------------------------------

    >>> # [check 14] invalid — k_h=0 is not a positive integer raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': 0}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 14] invalid — k_h=-1 is not a positive integer raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': -1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 14] invalid — k_h=1.5 is not an integer raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': 1.5}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 14] invalid — k_h='1' is a string not an integer raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'c1': '1'}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    # -------------------------------------------------------------------
    # Check 15: Each k_h satisfies k_h >= ceil(|C_h| / n) — feasibility condition.
    # -------------------------------------------------------------------

    >>> # [check 15] invalid — k_h=1 < ceil(3/2)=2 for 3 items and 2 agents raises ValueError
    >>> valuations_3 = {'Alice': {'m1': 5, 'm2': 3, 'm3': 1}, 'Bob': {'m1': 1, 'm2': 3, 'm3': 5}}
    >>> instance_3 = Instance(valuations=valuations_3)
    >>> alloc_3 = AllocationBuilder(instance_3)
    >>> validate_fair_division_inputs(alloc_3, {'c1': ['m1', 'm2', 'm3']}, {'c1': 1}, ['Alice', 'Bob'])
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # [check 15] valid — k_h=2 == ceil(3/2)=2, exactly on the boundary — no exception raised
    >>> validate_fair_division_inputs(alloc_3, {'c1': ['m1', 'm2', 'm3']}, {'c1': 2}, ['Alice', 'Bob'])
    """

    pass


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())