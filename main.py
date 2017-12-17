import torch
import torch.optim as optim
import argparse, pdb, numpy as np

from model import DQN, Dueling_DQN, DQN2
from learn import dqn_learning, OptimizerSpec
from schedules import *
from env import Grid, Agent
# Global Variables
# Extended data table 1 of nature paper
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
FRAME_HISTORY_LEN = 1
TARGET_UPDATE_FREQ = 1000
GAMMA = 0.99
LEARNING_FREQ = 2
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(100000, 0.1)
LEARNING_STARTS = 500
HARD_RESET_FREQ = 2500

def agent_learn(env, env_id, num_timesteps, double_dqn, dueling_dqn, varyenv):

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return env.get_total_steps() >= num_timesteps

    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )
    self_channel = False
    if dueling_dqn:
        dqn_learning(
            env=env,
            env_id=env_id,
            q_func=Dueling_DQN,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn,
            self_channel=self_channel,
            varyenv=varyenv,
            hard_reset_freq=HARD_RESET_FREQ
        )
    else:
        dqn_learning(
            env=env,
            env_id=env_id,
            q_func=DQN if self_channel else DQN2,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn,
            self_channel=self_channel,
            varyenv=varyenv,
            hard_reset_freq=HARD_RESET_FREQ
        )
    #env.close()



def main():
    parser = argparse.ArgumentParser(description='RL agents for atari')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train an RL agent for atari games")
    train_parser.add_argument("--task-id", type=int, required=True, help="0 = BeamRider, 1 = Breakout, 2 = Enduro, 3 = Pong, 4 = Qbert, 5 = Seaquest, 6 = Spaceinvaders")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--double-dqn", type=int, default=0, help="double dqn - 0 = No, 1 = Yes")
    train_parser.add_argument("--dueling-dqn", type=int, default=0, help="dueling dqn - 0 = No, 1 = Yes")

    args = parser.parse_args()

    # command
    if (args.gpu != None):
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print("CUDA Device: %d" %torch.cuda.current_device())



    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    double_dqn = (args.double_dqn == 1)
    dueling_dqn = (args.dueling_dqn == 1)
    #env = get_env(task, seed, task.env_id, double_dqn, dueling_dqn)
    env_id = "MyEnv1_varyenv"
    print("Training on %s, double_dqn %d, dueling_dqn %d" %(env_id, double_dqn, dueling_dqn))
    gridmap = np.zeros([100,100])
    gridmap[40:60, 40:60] = 2
    gridmap[10:13,20:23] = 1
    gridmap[80:83,30:33] = 1
    gridmap[75:78,90:93] = 1
    env = Grid(100,100, gridmap)#, grid_state_fn=lambda x:x.get_grid_img())  #the 3rd argument tells what the state this learner needs from grid
    #agentlist = [DQNSeeker((0,0,0), (255,255,255)) for k in range(50)]
    #env.init_agents(agentlist)
    agentlist = [Agent((0,0,0), (255,255,255)) for k in range(25)]
    env.init_agents(agentlist)
    agent_learn(env, env_id, num_timesteps=1000000, double_dqn=double_dqn, dueling_dqn=dueling_dqn, varyenv=True)


if __name__ == '__main__':
    main()
    #python main.py train --task-id 0 --gpu 0