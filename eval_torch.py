import time

import torch
import panda_gym
import numpy as np
import gym
import os, sys
from torch_arguments import get_args
from algos.pytorch.sac_sp.sac_per_her import SACTorch
#from kuka_real_gym import KukaReachEnv
from test_jaka_evolution import KukaReachEnv
#env = gym.make("PandaReach-v2")
#obs = env.reset()
#print(obs)

#ac=torch.load("actor.pth")
#print('ac={}'.format(ac))

def launch(net, args):
    env = KukaReachEnv(is_good_view=True, is_render=True)
    #env = gym.make(args.env_name,render = True)
    env.seed(args.seed)
    np.random.seed(args.seed)

    try:
        s_dim = env.observation_space.shape[0]
    except:
        s_dim = env.observation_space.spaces['observation'].shape[0] + \
                env.observation_space.spaces['desired_goal'].shape[0]
    s_dim = 6
    act_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.gpu_id != -1 else 'cpu')
    print("gpu_id:", args.gpu_id,
          "device:", device)
    print(net)
    net = net(act_dim=act_dim,
              obs_dim=s_dim,
              a_bound=a_bound,
              per_flag=args.per,
              her_flag=args.her,
              action_l2=args.action_l2,
              state_norm=args.state_norm,
              gamma=args.gamma,
              sess_opt=args.sess_opt,
              seed=args.seed,
              clip_return=args.clip_return,
              device=device,
              )


    net.ac.load_state_dict(torch.load('test_jaka_evolution/actor.pth'))
    net.load_norm("test_jaka_evolution/norm.pkl")
    #net.ac_targ.load_state_dict(torch.load("actor20.pth"))
    obs = env.reset()
    key_list = ['observation', 'desired_goal']
    obs = np.concatenate(([obs[key] for key in key_list]
                       ))
    #obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    noise_scale = args.noise_ps
    #print("s is what:",s)

    #mean = np.array([-0.00227926,-0.00372945 , 0.14144202 ,-0.00530948, -0.00139599 ,-0.00567219 ,-0.00257795 ,-0.00279574 , 0.15915842])
    #std = np.array([0.08629387,0.09107588 ,0.08918328, 0.3763083 , 0.5050184 , 0.5104836, 0.08784693 ,0.08897089 ,0.08646375])
    for i in range(10000):
        #obs = np.clip((obs - mean) / std,S-5, 5)
        a = net.get_action(obs)
        obs_next, reward, done, info = env.step(a)
        obs = np.concatenate(([obs_next[key] for key in key_list]
                             ))
        #obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        #print(reward,done,info["is_success"])
        time.sleep(0.05)
        # env.render()
        if (reward>0) | done:
            time.sleep(1)
            obs = env.reset()
            obs = np.concatenate(([obs[key] for key in key_list]
                                 ))
            #obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            print('get it')
if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # get the params
    args = get_args()
    from algos.tf1.td3_sp.TD3_per_her import TD3
    from algos.tf1.ddpg_sp.DDPG_per_her import DDPG
    from algos.tf1.sac_sp.SAC_per_her import SAC
    from algos.tf1.sac_auto.sac_auto_per_her import SAC_AUTO

    from algos.pytorch.td3_sp.td3_per_her import TD3Torch
    from algos.pytorch.ddpg_sp.ddpg_per_her import DDPGTorch
    from algos.pytorch.sac_sp.sac_per_her import SACTorch

    RL_list = [TD3, DDPG, SAC, SAC_AUTO, TD3Torch, DDPGTorch, SACTorch]

    [launch(net=net, args=args) for net in RL_list if net.__name__ == args.RL_name]