"""
Test the iterated-matching algorithm.

Programmer: Erel Segal-Halevi
Since:  2023-07
"""

import pytest

import fairpyx
import numpy as np

NUM_OF_RANDOM_INSTANCES=10

def test_feasibility():
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        instance = fairpyx.Instance.random_uniform(
            num_of_agents=70, num_of_items=10, normalized_sum_of_values=1000,
            agent_capacity_bounds=[1,10],
            agent_target_weight_bounds=[2,20],
            item_capacity_bounds=[20,40], 
            item_weight_bounds=[2,4],
            item_base_value_bounds=[1,1000],
            item_subjective_ratio_bounds=[0.5, 1.5]
            )
        allocation = fairpyx.divide(fairpyx.algorithms.iterated_maximum_matching_unadjusted, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, unadjusted")
        allocation = fairpyx.divide(fairpyx.algorithms.iterated_maximum_matching_adjusted, instance=instance)
        fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, adjusted")


def test_weights():
    instance = fairpyx.Instance(
        valuations={
            "alon":   {"c1": 300, "c2": 200, "c4": 150, "c5": 150, "c3": 100, "c6": 100},
            "ruti":   {"c3": 300, "c2": 200, "c4": 150, "c6": 150, "c1": 100, "c5": 100},
            "sigalit":{"c3": 250, "c1": 200, "c4": 200, "c2": 150, "c5": 100, "c6": 100},
            "uri":    {"c4": 300, "c3": 200, "c1": 150, "c5": 150, "c2": 100, "c6": 100},
            "ron":    {"c1": 250, "c4": 200, "c5": 200, "c3": 150, "c2": 100, "c6": 100},
        },

        agent_capacities={
            "alon": 2,
            "ruti": 2,
            "sigalit": 3,
            "uri": 2,
            "ron": 2
        },

        agent_target_weights={
            "alon": 6,
            "ruti": 5,
            "sigalit": 8,
            "uri": 4,
            "ron": 2
        },

        item_capacities={
            "c1": 2,
            "c2": 3,
            "c3": 1,
            "c4": 2,
            "c5": 4,
            "c6": 2
        },

        item_weights={
            "c1": 2,
            "c2": 3,
            "c3": 4,
            "c4": 2,
            "c5": 3,
            "c6": 4
        },

    )

    from fairpyx.explanations import StringsExplanationLogger
    string_explanation_logger = StringsExplanationLogger(agents=[name for name in instance.agents], language='he')
    allocation = fairpyx.divide(fairpyx.algorithms.iterated_maximum_matching_adjusted, instance=instance, explanation_logger = string_explanation_logger)
    fairpyx.validate_allocation(instance, allocation, title=f"adjusted")
    with open('explanation.txt','w') as f:
        print(string_explanation_logger.map_agent_to_explanation()['ron'],file=f)


if __name__ == "__main__":
     pytest.main(["-vs",__file__])

