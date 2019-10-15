import numpy as np
import pickle
from functools import reduce
import os
import sys
from gym_collision_avoidance.envs.config import Config

from gym_collision_avoidance.envs.policies.PPOCADRLPolicy import PPOCADRLPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy

from datetime import datetime

wandb_dir = '/home/mfe/code/'

##########################
# Set up logging as print to terminal and file
# 
log_filename = '{dir}/results/full_test_suites/full_test_suite_results_{datetime}.txt'.format(
                    dir=os.path.dirname(os.path.realpath(__file__)),
                    datetime=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                    ) 
log = open(log_filename, "w")
terminal = sys.stdout

# verbose = True
verbose = False

def write(message):
    terminal.write("{}\n".format(message))
    log.write("{}\n".format(message))
# 
##########################


##########################
# Define experiment parameters
#
policies = {
            'GA3C-CADRL-10': {
                'order': 4
                },
            # 'GA3C-CADRL-10-AWS': {
            #     'order': 4
            #     },
            # 'GA3C-CADRL-4-AWS': {
            #     'order': 2
            #     },
            'CADRL': {
                'order': 1
                },
            'RVO': {
                'order': 0
                },
            'DRL-Long': {
                'order': 3
                },
            }
ordered_policies = [key for key,value in sorted(policies.items(), key=lambda item: item[1]['order'])]
num_agents_to_test = [2,3,4]
# num_agents_to_test = [2,3,4, 5, 6, 8, 10]
num_test_cases = 100
# num_test_cases = 500
# num_test_cases = Config.NUM_TEST_CASES

vpref1 = True
radius_bounds = [0.2, 0.2]
if vpref1:
    vpref1_str = 'vpref1.0_r{}-{}/'.format(radius_bounds[0], radius_bounds[1])
else:
    vpref1_str = ''
# 
##########################


##########################
# Evaluate
#
write('\n\n#######################')
write('#######################')
write('#######################\n\n')
for num_agents in num_agents_to_test:
    write("Num agents: {}\n".format(num_agents))
    stats = {}
    for policy in policies:
        pickle_filename = '{dir}/results/full_test_suites/{vpref1_str}{num_agents}_agents/stats/{policy}.p'.format(vpref1_str=vpref1_str, dir=os.path.dirname(os.path.realpath(__file__)), num_agents=num_agents, policy=policy) 
        stats[policy] = pickle.load(open(pickle_filename, 'rb'))

    non_collision_inds = reduce(np.intersect1d, (stats[policy]['non_collision_inds'] for policy in policies))
    all_at_goal_inds = reduce(np.intersect1d, (stats[policy]['all_at_goal_inds'] for policy in policies))
    no_funny_business_inds = np.intersect1d(non_collision_inds, all_at_goal_inds)
    for policy in ordered_policies:
        write('---')
        write("Policy: {}".format(policy))
        num_collisions = num_test_cases-len(stats[policy]['non_collision_inds'])
        num_stuck = len(stats[policy]['stuck_inds'])
        pct_collisions = 100.0*num_collisions/num_test_cases
        pct_stuck = 100.0*num_stuck/num_test_cases
        mean_extra_time_to_goal_list = []
        for ind in no_funny_business_inds:
            mean_extra_time_to_goal_list.append(stats[policy][ind]['mean_extra_time_to_goal'])
        pctls = np.percentile(np.array(mean_extra_time_to_goal_list),[50,75,90])
        pctls = [round(pctl, 2) for pctl in pctls]

        if verbose:
            write("Total # test cases with collision: %i of %i (%.2f%%)" %(num_collisions,num_test_cases,pct_collisions))
            write("Total # test cases where agent got stuck: %i of %i (%.2f%%)" %(num_stuck,num_test_cases,pct_stuck))
            write("Extra time to goal [50th, 75th, 90th] percentile (non-collision/non-stuck cases):")
            write(pctls)
        else:        
            write("{total:.2f} ({pct_collisions:.2f} / {pct_stuck:.2f})".format(
                total=pct_collisions+pct_stuck,
                pct_collisions=pct_collisions,
                pct_stuck=pct_stuck))
            write(str(pctls).strip('[]').replace(',',' /'))

    write('\n----------\n')

write('\n\n#######################')
write('#######################')
write('#######################')
# 
##########################

