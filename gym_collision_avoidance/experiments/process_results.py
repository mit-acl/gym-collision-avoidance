import numpy as np
import pickle
from functools import reduce
import os

policies = ['PPOCADRLPolicy', 'RVOPolicy', 'CADRLPolicy']
num_agents_to_test = [2,3,4]
test_cases = 8

print('\n\n#######################')
print('#######################')
print('#######################\n\n')
for num_agents in num_agents_to_test:
    print("Num agents: {}\n".format(num_agents))
    stats = {}
    for policy in policies:
        stats[policy] = pickle.load(open('{dir}/../logs/test_case_stats/{num_agents}_agents/{policy}.p'.format(dir=os.path.dirname(os.path.realpath(__file__)), num_agents=num_agents, policy=policy),'rb'))

    non_collision_inds = reduce(np.intersect1d, (stats[policy]['non_collision_inds'] for policy in policies))
    all_at_goal_inds = reduce(np.intersect1d, (stats[policy]['all_at_goal_inds'] for policy in policies))
    no_funny_business_inds = np.intersect1d(non_collision_inds, all_at_goal_inds)
    for policy in policies:
        print('---')
        print("Policy: {}".format(policy))
        num_collisions = test_cases-len(stats[policy]['non_collision_inds'])
        num_stuck = len(stats[policy]['stuck_inds'])
        print("Total # test cases with collision: %i of %i (%.2f%%)" %(num_collisions,test_cases,(100.0*num_collisions/(test_cases))))
        print("Total # test cases where agent got stuck: %i of %i (%.2f%%)" %(num_stuck,test_cases,(100.0*num_stuck/(test_cases))))
        # time_to_goal_sum = 0.0
        mean_extra_time_to_goal_list = []
        for ind in no_funny_business_inds:
            # time_to_goal_sum += stats[alg][ind]['total_time_to_goal']
            mean_extra_time_to_goal_list.append(stats[policy][ind]['mean_extra_time_to_goal'])
        print("Extra time to goal [50th, 75th, 90th] percentile (non-collision/non-stuck cases):")
        print(np.percentile(np.array(mean_extra_time_to_goal_list),[50,75,90]))
    print('\n----------\n')

print('\n\n#######################')
print('#######################')
print('#######################')