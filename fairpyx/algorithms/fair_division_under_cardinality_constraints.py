"""
An implementation of the algorithm in:
"Fair Division Under Cardinality Constraints", by A. Biswas, S. Barman (2018), https://arxiv.org/abs/1804.09521
Programmer: Sapir Dahan
Date : 2026-04
"""

import math
import logging

import networkz as nx

from fairpyx import Instance, AllocationBuilder

logger = logging.getLogger(__name__)



def fair_division_under_cardinality_constraints(
    alloc: AllocationBuilder,
    item_categories: dict,
    agent_category_capacities: dict,
    initial_agent_order: list,
):
    """
    Compute a feasible EF1 allocation under cardinality constraints (Algorithm 1,
    Biswas & Barman 2018).

    All agents share the same per-category threshold k_h. Valuations are additive and
    may differ across agents. The algorithm guarantees EF1: for every pair of agents
    (i, j) there exists some good g in j's bundle such that
        v_i(A_i) >= v_i(A_j \\ {g}).

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
    :param agent_category_capacities: a dictionary mapping each agent name (str) to a
        dictionary of per-category integer capacities (k_h). All agents must have the same
        k_h for each category. The sum of each agent's k_h values across all categories
        should equal the agent_capacity passed to Instance.
        Example: {'Agent1': {'c1': 1, 'c2': 2}, 'Agent2': {'c1': 1, 'c2': 2}}.
    :param initial_agent_order: a list of agent names specifying the initial picking order
        for the first category. Must contain each agent exactly once.

    >>> # Example 1: basic — 2 agents, 2 categories, 1 good each, k_h=1
    >>> # Agent1 prefers m1 and m3; Agent2 prefers m2 and m4.
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'Agent1': {'m1': 9, 'm2': 3, 'm3': 8, 'm4': 2},
    ...     'Agent2': {'m1': 3, 'm2': 9, 'm3': 2, 'm4': 8},
    ... }
    >>> item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3', 'm4']}
    >>> agent_category_capacities = {'Agent1': {'c1': 1, 'c2': 1}, 'Agent2': {'c1': 1, 'c2': 1}}
    >>> sum_caps = {a: sum(v.values()) for a, v in agent_category_capacities.items()}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2', 'm3', 'm4'], agent_capacities=sum_caps)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])
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
    >>> agent_category_capacities = {
    ...     'Agent1': {'c1': 1, 'c2': 1},
    ...     'Agent2': {'c1': 1, 'c2': 1},
    ...     'Agent3': {'c1': 1, 'c2': 1},
    ... }
    >>> sum_caps = {a: sum(v.values()) for a, v in agent_category_capacities.items()}
    >>> instance = Instance(valuations=valuations, items=['m1','m2','m3','m4','m5','m6'], agent_capacities=sum_caps)
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2', 'Agent3'])
    {'Agent1': ['m1', 'm4'], 'Agent2': ['m2', 'm5'], 'Agent3': ['m3', 'm6']}

    >>> # Example 3: single agent — trivially receives all goods (up to its capacity)
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Alice': {'m1': 10, 'm2': 7, 'm3': 4}}
    >>> item_categories = {'c1': ['m1', 'm2', 'm3']}
    >>> agent_category_capacities = {'Alice': {'c1': 3}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2', 'm3'], agent_capacities={'Alice': 3})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Alice'])
    {'Alice': ['m1', 'm2', 'm3']}

    >>> # Example 4: single category — degenerates to a single greedy_round_robin call
    >>> from fairpyx import Instance, divide
    >>> valuations = {'A': {'x': 10, 'y': 5, 'z': 1}, 'B': {'x': 1, 'y': 5, 'z': 10}}
    >>> item_categories = {'c1': ['x', 'y', 'z']}
    >>> agent_category_capacities = {'A': {'c1': 2}, 'B': {'c1': 1}}
    >>> instance = Instance(valuations=valuations, items=['x', 'y', 'z'], agent_capacities={'A': 2, 'B': 1})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['A', 'B'])
    {'A': ['x', 'y'], 'B': ['z']}

    >>> # Example 5: single good per category — each agent can receive at most one good per category
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': 8, 'm2': 5}, 'Agent2': {'m1': 5, 'm2': 8}}
    >>> item_categories = {'c1': ['m1'], 'c2': ['m2']}
    >>> agent_category_capacities = {'Agent1': {'c1': 1, 'c2': 1}, 'Agent2': {'c1': 1, 'c2': 1}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2'], agent_capacities={'Agent1': 2, 'Agent2': 2})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])
    {'Agent1': ['m1'], 'Agent2': ['m2']}

    >>> # Example 6: boundary case — k_h = ceil(|C_h| / n), exactly the minimum feasible threshold
    >>> # 3 agents, 3 goods in c1, k_h = ceil(3/3) = 1.
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'A': {'g1': 9, 'g2': 6, 'g3': 3},
    ...     'B': {'g1': 3, 'g2': 9, 'g3': 6},
    ...     'C': {'g1': 6, 'g2': 3, 'g3': 9},
    ... }
    >>> item_categories = {'c1': ['g1', 'g2', 'g3']}
    >>> agent_category_capacities = {'A': {'c1': 1}, 'B': {'c1': 1}, 'C': {'c1': 1}}
    >>> instance = Instance(valuations=valuations, items=['g1', 'g2', 'g3'], agent_capacities={'A': 1, 'B': 1, 'C': 1})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['A', 'B', 'C'])
    {'A': ['g1'], 'B': ['g2'], 'C': ['g3']}

    >>> # Example 7: asymmetric category sizes — c1 has 4 goods (k_h=2), c2 has 1 good (k_h=1)
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'Agent1': {'m1': 10, 'm2': 8, 'm3': 6, 'm4': 4, 'm5': 9},
    ...     'Agent2': {'m1': 4,  'm2': 6, 'm3': 8, 'm4': 10, 'm5': 7},
    ... }
    >>> item_categories = {'c1': ['m1', 'm2', 'm3', 'm4'], 'c2': ['m5']}
    >>> agent_category_capacities = {'Agent1': {'c1': 2, 'c2': 1}, 'Agent2': {'c1': 2, 'c2': 1}}
    >>> instance = Instance(valuations=valuations, items=['m1','m2','m3','m4','m5'],
    ...                     agent_capacities={'Agent1': 3, 'Agent2': 3})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])
    {'Agent1': ['m1', 'm2', 'm5'], 'Agent2': ['m3', 'm4']}

    >>> # Example 8: EF1 property verification — for the result, no agent envies another
    >>> # by more than the removal of one good from the other's bundle.
    >>> from fairpyx import Instance, divide
    >>> valuations = {
    ...     'Agent1': {'m1': 10, 'm2': 8, 'm3': 6, 'm4': 4},
    ...     'Agent2': {'m1': 4,  'm2': 6, 'm3': 8, 'm4': 10},
    ... }
    >>> item_categories = {'c1': ['m1', 'm2'], 'c2': ['m3', 'm4']}
    >>> agent_category_capacities = {'Agent1': {'c1': 1, 'c2': 1}, 'Agent2': {'c1': 1, 'c2': 1}}
    >>> sum_caps = {a: sum(v.values()) for a, v in agent_category_capacities.items()}
    >>> instance = Instance(valuations=valuations, items=['m1','m2','m3','m4'], agent_capacities=sum_caps)
    >>> result = divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...                 item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...                 initial_agent_order=['Agent1', 'Agent2'])
    >>> # EF1 check for Agent1: there exists a good in Agent2's bundle whose removal ends the envy
    >>> v1 = valuations['Agent1']
    >>> val1_own   = sum(v1[g] for g in result['Agent1'])
    >>> val1_other = sum(v1[g] for g in result['Agent2'])
    >>> val1_own >= val1_other - max(v1[g] for g in result['Agent2'])
    True

    >>> # Example 9: invalid — negative valuation raises ValueError
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': -1, 'm2': 3}, 'Agent2': {'m1': 4, 'm2': 6}}
    >>> item_categories = {'c1': ['m1', 'm2']}
    >>> agent_category_capacities = {'Agent1': {'c1': 1}, 'Agent2': {'c1': 1}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2'], agent_capacities={'Agent1': 1, 'Agent2': 1})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 10: invalid — k_h too small (k_h=1 < ceil(3/2)=2) raises ValueError
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': 5, 'm2': 3, 'm3': 1}, 'Agent2': {'m1': 1, 'm2': 3, 'm3': 5}}
    >>> item_categories = {'c1': ['m1', 'm2', 'm3']}
    >>> agent_category_capacities = {'Agent1': {'c1': 1}, 'Agent2': {'c1': 1}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2', 'm3'], agent_capacities={'Agent1': 1, 'Agent2': 1})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 11: invalid — a good appears in two categories raises ValueError
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': 5, 'm2': 3}, 'Agent2': {'m1': 3, 'm2': 5}}
    >>> item_categories_dup = {'c1': ['m1', 'm2'], 'c2': ['m1']}
    >>> agent_category_capacities = {'Agent1': {'c1': 1, 'c2': 1}, 'Agent2': {'c1': 1, 'c2': 1}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2'], agent_capacities={'Agent1': 2, 'Agent2': 2})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories_dup, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 12: invalid — an agent is missing a threshold for a category raises ValueError
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': 5}, 'Agent2': {'m1': 3}}
    >>> item_categories = {'c1': ['m1']}
    >>> agent_category_capacities_bad = {'Agent1': {'c1': 1}, 'Agent2': {}}
    >>> instance = Instance(valuations=valuations, items=['m1'], agent_capacities={'Agent1': 1, 'Agent2': 0})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities_bad,
    ...        initial_agent_order=['Agent1', 'Agent2'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 13: invalid — initial_agent_order is empty while instance has agents raises ValueError
    >>> # (Instance(valuations={}) cannot be constructed; test zero-order case instead)
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': 5}, 'Agent2': {'m1': 3}}
    >>> item_categories = {'c1': ['m1']}
    >>> agent_category_capacities = {'Agent1': {'c1': 1}, 'Agent2': {'c1': 0}}
    >>> instance = Instance(valuations=valuations, items=['m1'], agent_capacities={'Agent1': 1, 'Agent2': 0})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=[])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 14: invalid — duplicate agent in initial_agent_order raises ValueError
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': 5}, 'Agent2': {'m1': 3}}
    >>> item_categories = {'c1': ['m1']}
    >>> agent_category_capacities = {'Agent1': {'c1': 1}, 'Agent2': {'c1': 0}}
    >>> instance = Instance(valuations=valuations, items=['m1'], agent_capacities={'Agent1': 1, 'Agent2': 0})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent1'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 15: invalid — a good is in the instance but missing from all categories raises ValueError
    >>> from fairpyx import Instance, divide
    >>> valuations = {'Agent1': {'m1': 5, 'm2': 3}, 'Agent2': {'m1': 3, 'm2': 5}}
    >>> item_categories_incomplete = {'c1': ['m1']}  # m2 is not in any category
    >>> agent_category_capacities = {'Agent1': {'c1': 1}, 'Agent2': {'c1': 1}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2'], agent_capacities={'Agent1': 1, 'Agent2': 1})
    >>> divide(algorithm=fair_division_under_cardinality_constraints, instance=instance,
    ...        item_categories=item_categories_incomplete, agent_category_capacities=agent_category_capacities,
    ...        initial_agent_order=['Agent1', 'Agent2'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...
    """
    # --- Section: validate all inputs before proceeding ---
    # validate_fair_division_inputs(alloc, item_categories, agent_category_capacities, initial_agent_order)

    # --- Section: initialise current agent order ---
    # current_order = initial_agent_order
    # logger.info("Starting fair_division_under_cardinality_constraints. Initial order: %s", current_order)

    # --- Section: main loop — process one category at a time (Algorithm 1, step 3) ---
    # for category in item_categories:
    #     logger.info("Processing category '%s' with agent order %s", category, current_order)
    #
    #     # Step 4: allocate goods in this category using Greedy Round-Robin (Algorithm 2)
    #     greedy_round_robin(alloc, item_categories[category], current_order,
    #                        agent_category_capacities, category)
    #     logger.info("Bundles after greedy_round_robin for '%s': %s", category, alloc.bundles)
    #
    #     # Step 6: eliminate envy cycles and obtain an acyclic envy graph (Lemma 1)
    #     envy_graph = eliminate_envy_cycles(alloc)
    #     logger.info("Envy graph after cycle elimination: %s", list(envy_graph.edges))
    #
    #     # Step 7: update agent order to the topological sort of the acyclic envy graph
    #     current_order = list(nx.topological_sort(envy_graph))
    #     logger.info("Updated agent order: %s", current_order)

    # logger.info("Final allocation: %s", alloc.bundles)
    pass


def greedy_round_robin(
    alloc: AllocationBuilder,
    items_in_category: list,
    agent_order: list,
    agent_category_capacities: dict,
    category: str,
) -> None:
    """
    Allocate all goods from a single category using Greedy Round-Robin (Algorithm 2,
    Biswas & Barman 2018).

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
    :param agent_category_capacities: a dictionary mapping each agent name (str) to a
        dictionary of per-category integer capacities.
    :param category: the name (str) of the category currently being allocated.

    >>> # Example 1: 2 agents, 2 goods, k_h=1 — each agent picks their top good
    >>> from fairpyx import Instance, AllocationBuilder
    >>> valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 8}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2'], agent_capacities={'Alice': 1, 'Bob': 1})
    >>> alloc = AllocationBuilder(instance)
    >>> agent_category_capacities = {'Alice': {'c1': 1}, 'Bob': {'c1': 1}}
    >>> greedy_round_robin(alloc, ['m1', 'm2'], ['Alice', 'Bob'], agent_category_capacities, 'c1')
    >>> alloc.sorted()
    {'Alice': ['m1'], 'Bob': ['m2']}

    >>> # Example 2: 3 agents, 3 goods, k_h=1 — each picks their unique top good in order
    >>> valuations = {'A': {'m1': 9, 'm2': 5, 'm3': 1},
    ...               'B': {'m1': 3, 'm2': 9, 'm3': 2},
    ...               'C': {'m1': 2, 'm2': 4, 'm3': 8}}
    >>> agent_category_capacities = {'A': {'c1': 1}, 'B': {'c1': 1}, 'C': {'c1': 1}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2', 'm3'],
    ...                     agent_capacities={'A': 1, 'B': 1, 'C': 1})
    >>> alloc = AllocationBuilder(instance)
    >>> greedy_round_robin(alloc, ['m1', 'm2', 'm3'], ['A', 'B', 'C'], agent_category_capacities, 'c1')
    >>> alloc.sorted()
    {'A': ['m1'], 'B': ['m2'], 'C': ['m3']}

    >>> # Example 3: 2 agents, 4 goods, k_h=2 — each agent picks 2 goods
    >>> valuations = {'Alice': {'m1': 10, 'm2': 8, 'm3': 5, 'm4': 3},
    ...               'Bob':   {'m1': 3,  'm2': 5, 'm3': 8, 'm4': 10}}
    >>> agent_category_capacities = {'Alice': {'c1': 2}, 'Bob': {'c1': 2}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2', 'm3', 'm4'],
    ...                     agent_capacities={'Alice': 2, 'Bob': 2})
    >>> alloc = AllocationBuilder(instance)
    >>> greedy_round_robin(alloc, ['m1', 'm2', 'm3', 'm4'], ['Alice', 'Bob'], agent_category_capacities, 'c1')
    >>> alloc.sorted()
    {'Alice': ['m1', 'm2'], 'Bob': ['m3', 'm4']}
    """
    # --- Section: initialise per-category remaining capacities ---
    # remaining_caps = {agent: agent_category_capacities[agent][category] for agent in agent_order}

    # --- Section: keep track of which agents still have capacity and which goods remain ---
    # remaining_goods = [g for g in alloc.remaining_items() if g in items_in_category]
    # agents_with_capacity = set(agent for agent in agent_order if remaining_caps[agent] > 0)

    # --- Section: cycle through agent_order, each picks their best remaining good ---
    # for agent in cycle(agent_order):
    #     if not remaining_goods or not agents_with_capacity:
    #         break
    #     if agent not in agents_with_capacity:
    #         continue
    #     best_good = max(remaining_goods, key=lambda g: alloc.instance.agent_item_value(agent, g))
    #     alloc.give(agent, best_good, logger)
    #     remaining_goods.remove(best_good)
    #     remaining_caps[agent] -= 1
    #     if remaining_caps[agent] == 0:
    #         agents_with_capacity.discard(agent)
    pass


def eliminate_envy_cycles(alloc: AllocationBuilder) -> nx.DiGraph:
    """
    Build the envy graph for the current (partial) allocation, eliminate all directed cycles
    by rotating bundles along each cycle, and return the resulting acyclic envy graph
    (Lemma 1, Biswas & Barman 2018).

    Envy relation: agent i envies agent j if
        sum_{g in A_j} v_i(g) > sum_{g in A_i} v_i(g).
    A directed edge i -> j is added to the graph when i envies j.

    Cycle elimination: for a detected cycle (a_1, a_2, ..., a_r), rotate bundles so that
    a_1 receives a_2's bundle, a_2 receives a_3's bundle, ..., a_r receives a_1's bundle.
    The paper proves that no agent's value decreases under such a rotation. This is repeated
    until the graph contains no directed cycles (i.e., it is a DAG).

    The returned DAG is passed to nx.topological_sort to derive the agent order for the next
    category: agents with no incoming envy edges (nobody envies them) pick first.

    :param alloc: an allocation builder whose alloc.bundles reflect the current partial
        allocation after the most recent greedy_round_robin call.
    :return: a networkz DiGraph representing the acyclic envy graph after all cycles have
        been eliminated.

    >>> # Example 1: no envy — graph is already a DAG, bundles are unchanged
    >>> from fairpyx import Instance, AllocationBuilder
    >>> import networkz as nx
    >>> valuations = {'Alice': {'m1': 9, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 9}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2'])
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('Alice', 'm1')
    >>> alloc.give('Bob', 'm2')
    >>> G = eliminate_envy_cycles(alloc)
    >>> list(nx.simple_cycles(G))
    []
    >>> sorted(alloc.bundles['Alice'])
    ['m1']
    >>> sorted(alloc.bundles['Bob'])
    ['m2']

    >>> # Example 2: 2-agent envy cycle — bundles are swapped to eliminate the cycle
    >>> # Alice has m1 (she values at 3), Bob has m2 (he values at 3).
    >>> # Alice envies Bob (values m2=7 > m1=3) and Bob envies Alice (values m1=7 > m2=3).
    >>> valuations = {'Alice': {'m1': 3, 'm2': 7}, 'Bob': {'m1': 7, 'm2': 3}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2'])
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('Alice', 'm1')
    >>> alloc.give('Bob', 'm2')
    >>> G = eliminate_envy_cycles(alloc)
    >>> list(nx.simple_cycles(G))
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
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2', 'm3'])
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give('A', 'm1')
    >>> alloc.give('B', 'm2')
    >>> alloc.give('C', 'm3')
    >>> G = eliminate_envy_cycles(alloc)
    >>> list(nx.simple_cycles(G))
    []
    >>> sorted(alloc.bundles['A'])
    ['m2']
    >>> sorted(alloc.bundles['B'])
    ['m3']
    >>> sorted(alloc.bundles['C'])
    ['m1']
    """
    # --- Section: build the envy graph ---
    # envy_graph = nx.DiGraph()
    # agents = list(alloc.bundles.keys())
    # envy_graph.add_nodes_from(agents)
    # for agent_i in agents:
    #     val_i_own = sum(alloc.instance.agent_item_value(agent_i, g) for g in alloc.bundles[agent_i])
    #     for agent_j in agents:
    #         if agent_i == agent_j:
    #             continue
    #         val_i_other = sum(alloc.instance.agent_item_value(agent_i, g) for g in alloc.bundles[agent_j])
    #         if val_i_other > val_i_own:
    #             envy_graph.add_edge(agent_i, agent_j)

    # --- Section: eliminate cycles iteratively ---
    # while not nx.is_directed_acyclic_graph(envy_graph):
    #     cycle_nodes = next(nx.simple_cycles(envy_graph))  # list of agents forming one cycle
    #     # rotate bundles: a_0 gets a_1's bundle, a_1 gets a_2's bundle, ..., a_{r-1} gets a_0's bundle
    #     first_bundle = alloc.bundles[cycle_nodes[0]].copy()
    #     for i in range(len(cycle_nodes) - 1):
    #         alloc.bundles[cycle_nodes[i]] = alloc.bundles[cycle_nodes[i + 1]].copy()
    #     alloc.bundles[cycle_nodes[-1]] = first_bundle
    #     # rebuild the envy graph with updated bundles
    #     envy_graph = ... (repeat edge construction above)

    # --- Section: return the acyclic envy graph ---
    # return envy_graph
    return nx.DiGraph()  # placeholder so callers see a DiGraph, not None


def validate_fair_division_inputs(
    alloc: AllocationBuilder,
    item_categories: dict,
    agent_category_capacities: dict,
    initial_agent_order: list,
):
    """
    Validate all inputs for fair_division_under_cardinality_constraints before the algorithm runs.

    Checks performed (in order):
    - At least one agent exists.
    - Keys of agent_category_capacities match the set of agents in alloc.instance exactly.
    - initial_agent_order contains each agent exactly once (no duplicates, no missing agents).
    - No item appears in more than one category.
    - Every item listed in item_categories appears in the instance valuations.
    - Every item in the instance valuations appears in exactly one category (no uncategorised goods).
    - All thresholds k_h are positive integers.
    - All agents share the same k_h for each category (uniformity required by the paper).
    - Each k_h satisfies k_h >= ceil(|C_h| / n) — the feasibility condition from the paper.
    - All valuation values are non-negative.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity
        for items and agents.
    :param item_categories: a dictionary mapping each category name (str) to a list of item
        names belonging to that category. Example: {'c1': ['m1', 'm2'], 'c2': ['m3']}.
    :param agent_category_capacities: a dictionary mapping each agent name (str) to a dictionary
        of per-category integer capacities (the thresholds k_h). All agents must share the same
        k_h for each category h. Example: {'Agent1': {'c1': 1}, 'Agent2': {'c1': 1}}.
    :param initial_agent_order: a list of agent names specifying the initial picking order.
        Must contain each agent exactly once.
    :raises ValueError: if any of the above conditions are violated.

    >>> # Example 1: valid inputs — no exception raised
    >>> from fairpyx import Instance, AllocationBuilder
    >>> valuations = {'Alice': {'m1': 5, 'm2': 3}, 'Bob': {'m1': 3, 'm2': 7}}
    >>> instance = Instance(valuations=valuations, items=['m1', 'm2'], agent_capacities={'Alice': 1, 'Bob': 1})
    >>> alloc = AllocationBuilder(instance)
    >>> item_categories = {'c1': ['m1', 'm2']}
    >>> agent_category_capacities = {'Alice': {'c1': 1}, 'Bob': {'c1': 1}}
    >>> validate_fair_division_inputs(alloc, item_categories, agent_category_capacities, ['Alice', 'Bob'])

    >>> # Example 2: invalid — initial_agent_order is empty while instance has agents raises ValueError
    >>> # (Instance(valuations={}) cannot be constructed; test zero-order case instead)
    >>> validate_fair_division_inputs(alloc, item_categories, agent_category_capacities, [])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 3: invalid — negative valuation raises ValueError
    >>> valuations_neg = {'Alice': {'m1': -1, 'm2': 3}, 'Bob': {'m1': 4, 'm2': 6}}
    >>> instance_neg = Instance(valuations=valuations_neg, items=['m1', 'm2'], agent_capacities={'Alice': 1, 'Bob': 1})
    >>> alloc_neg = AllocationBuilder(instance_neg)
    >>> validate_fair_division_inputs(alloc_neg, {'c1': ['m1', 'm2']}, {'Alice': {'c1': 1}, 'Bob': {'c1': 1}}, ['Alice', 'Bob'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 4: invalid — k_h too small (k_h=1 < ceil(3/2)=2) raises ValueError
    >>> valuations_3 = {'Alice': {'m1': 5, 'm2': 3, 'm3': 1}, 'Bob': {'m1': 1, 'm2': 3, 'm3': 5}}
    >>> instance_3 = Instance(valuations=valuations_3, items=['m1', 'm2', 'm3'], agent_capacities={'Alice': 1, 'Bob': 1})
    >>> alloc_3 = AllocationBuilder(instance_3)
    >>> validate_fair_division_inputs(alloc_3, {'c1': ['m1', 'm2', 'm3']}, {'Alice': {'c1': 1}, 'Bob': {'c1': 1}}, ['Alice', 'Bob'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 5: invalid — item appears in two categories raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1', 'm2'], 'c2': ['m1']}, {'Alice': {'c1': 1, 'c2': 1}, 'Bob': {'c1': 1, 'c2': 1}}, ['Alice', 'Bob'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 6: invalid — an agent is missing a threshold for one of the categories raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, {'Alice': {'c1': 1}, 'Bob': {}}, ['Alice', 'Bob'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 7: invalid — duplicate agent in initial_agent_order raises ValueError
    >>> validate_fair_division_inputs(alloc, item_categories, agent_category_capacities, ['Alice', 'Alice'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...

    >>> # Example 8: invalid — item present in valuations but missing from all categories raises ValueError
    >>> validate_fair_division_inputs(alloc, {'c1': ['m1']}, {'Alice': {'c1': 1}, 'Bob': {'c1': 1}}, ['Alice', 'Bob'])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ...
    """
    # --- Section: check at least one agent exists ---
    # agents = list(alloc.instance.agents)
    # if len(agents) == 0:
    #     raise ValueError("No agents found in the instance.")

    # --- Section: check agent_category_capacities keys match instance agents exactly ---
    # if set(agent_category_capacities.keys()) != set(agents):
    #     raise ValueError("agent_category_capacities agents do not match the instance agents.")

    # --- Section: check initial_agent_order has no duplicates and covers all agents ---
    # if len(initial_agent_order) != len(set(initial_agent_order)):
    #     raise ValueError("initial_agent_order contains duplicate agents.")
    # if set(initial_agent_order) != set(agents):
    #     raise ValueError("initial_agent_order does not match the instance agents.")

    # --- Section: check no item appears in more than one category ---
    # all_items_in_categories = [item for items in item_categories.values() for item in items]
    # if len(all_items_in_categories) != len(set(all_items_in_categories)):
    #     raise ValueError("An item appears in more than one category.")

    # --- Section: check every categorised item is known to the instance ---
    # instance_items = set(alloc.instance.items)
    # for item in all_items_in_categories:
    #     if item not in instance_items:
    #         raise ValueError(f"Item '{item}' in item_categories is not present in the instance.")

    # --- Section: check every instance item appears in exactly one category ---
    # categorised_items = set(all_items_in_categories)
    # for item in instance_items:
    #     if item not in categorised_items:
    #         raise ValueError(f"Item '{item}' is in the instance but not assigned to any category.")

    # --- Section: check all k_h values are positive integers and uniform across agents ---
    # for category in item_categories:
    #     capacities_for_category = []
    #     for agent in agents:
    #         if category not in agent_category_capacities[agent]:
    #             raise ValueError(f"Agent '{agent}' has no threshold for category '{category}'.")
    #         k = agent_category_capacities[agent][category]
    #         if not isinstance(k, int) or k < 1:
    #             raise ValueError(f"Threshold for agent '{agent}', category '{category}' must be a positive integer.")
    #         capacities_for_category.append(k)
    #     if len(set(capacities_for_category)) > 1:
    #         raise ValueError(f"Thresholds for category '{category}' differ across agents (paper requires uniform k_h).")

    # --- Section: check k_h >= ceil(|C_h| / n) for feasibility ---
    # n = len(agents)
    # for category, items in item_categories.items():
    #     k_h = agent_category_capacities[agents[0]][category]
    #     required = math.ceil(len(items) / n)
    #     if k_h < required:
    #         raise ValueError(
    #             f"Threshold k_{category}={k_h} is too small for {len(items)} goods and {n} agents "
    #             f"(need k_h >= ceil({len(items)}/{n}) = {required})."
    #         )

    # --- Section: check all valuations are non-negative ---
    # for agent in agents:
    #     for item in alloc.instance.items:
    #         val = alloc.instance.agent_item_value(agent, item)
    #         if val < 0:
    #             raise ValueError(f"Valuation of agent '{agent}' for item '{item}' is negative ({val}).")
    pass



if __name__ == "__main__":
    import doctest
    print(doctest.testmod())