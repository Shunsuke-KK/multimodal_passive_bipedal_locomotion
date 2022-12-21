#from vertical_mvt_pendulum import VerticalMvtPendulumEnv
import os
from spinup import sk_sac,sac_core
from spinup.utils.test_policy import load_policy_and_env, run_policy
import gym
import os
import torch
from custom_env import PassiveBipedal_2dEnv

path=os.getcwd()+'/assets'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    ### Common parameters###
    parser.add_argument('run_type', type=str, choices=('run, test'))
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--itr', type=int, default=-1)

    ### MODEL FREE parameters###

    args = parser.parse_args()


    save_folder=os.path.join(os.getcwd(),'data',args.exp_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    logger_kwargs = dict(output_dir=save_folder, exp_name=args.exp_name)

    VPenv = PassiveBipedal_2dEnv(path=path)

    if args.run_type=='run':
        sk_sac(lambda : VPenv, actor_critic=sac_core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[100]*2), gamma=0.99,
            seed=args.seed, steps_per_epoch=1000, epochs=10000,max_ep_len=1000,
            batch_size=256,save_freq=100, logger_kwargs=logger_kwargs,env_type= 'PassiveBipedal')

    elif args.run_type=='test':
        load_path=os.path.join(os.getcwd(),'data',args.exp_name)
        _, get_action = load_policy_and_env(load_path,10000,True)
        run_policy(VPenv, get_action)