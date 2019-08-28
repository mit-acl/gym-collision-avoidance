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

wandb_dir = '/home/mfe/code/'

##########################
# Set up logging as print to terminal and file
# 
log_filename = '{dir}/results/full_test_suites/log.txt'.format(dir=os.path.dirname(os.path.realpath(__file__))) 
log = open(log_filename, "w")
terminal = sys.stdout

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
                'policy': GA3CCADRLPolicy,
                'checkpt_name': 'network_01900000'
                },
            'GA3C-CADRL-10-AWS': {
                'policy': GA3CCADRLPolicy,
                'checkpt_dir': wandb_dir,
                'checkpt_name': 'network_01900000'
                },
            'GA3C-CADRL-4-AWS': {
                'policy': GA3CCADRLPolicy,
                'checkpt_dir': wandb_dir+"run-20190727_015942-jzuhlntn/",
                'checkpt_name': 'network_01490000'
                },
            'CADRL': {
                'policy': CADRLPolicy,
                },
            'RVO': {
                'policy': RVOPolicy,
                },
            }
num_agents_to_test = [2]
# num_agents_to_test = [2,3,4, 5, 6, 8, 10]
num_test_cases = 500
# num_test_cases = Config.NUM_TEST_CASES
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
        pickle_filename = '{dir}/results/full_test_suites/{num_agents}_agents/stats/{policy}.p'.format(dir=os.path.dirname(os.path.realpath(__file__)), num_agents=num_agents, policy=policy) 
        stats[policy] = pickle.load(open(pickle_filename, 'rb'))

    non_collision_inds = reduce(np.intersect1d, (stats[policy]['non_collision_inds'] for policy in policies))
    all_at_goal_inds = reduce(np.intersect1d, (stats[policy]['all_at_goal_inds'] for policy in policies))
    no_funny_business_inds = np.intersect1d(non_collision_inds, all_at_goal_inds)
    for policy in policies:
        write('---')
        write("Policy: {}".format(policy))
        num_collisions = num_test_cases-len(stats[policy]['non_collision_inds'])
        num_stuck = len(stats[policy]['stuck_inds'])
        write("Total # test cases with collision: %i of %i (%.2f%%)" %(num_collisions,num_test_cases,(100.0*num_collisions/(num_test_cases))))
        write("Total # test cases where agent got stuck: %i of %i (%.2f%%)" %(num_stuck,num_test_cases,(100.0*num_stuck/(num_test_cases))))
        mean_extra_time_to_goal_list = []
        for ind in no_funny_business_inds:
            mean_extra_time_to_goal_list.append(stats[policy][ind]['mean_extra_time_to_goal'])
        write("Extra time to goal [50th, 75th, 90th] percentile (non-collision/non-stuck cases):")
        pctls = np.percentile(np.array(mean_extra_time_to_goal_list),[50,75,90])
        write(pctls)
    write('\n----------\n')

write('\n\n#######################')
write('#######################')
write('#######################')
# 
##########################

