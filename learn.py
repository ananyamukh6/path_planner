"""
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

import torch, pdb, random
from torch.autograd import Variable
import sys
import os
import itertools
import numpy as np
import random
from collections import namedtuple
from replay_buffer import *
from schedules import *
from logger import Logger
import time
import matplotlib.pyplot as plt

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Set the logger
logger = Logger('./logs')
def to_np(x):
    return x.data.cpu().numpy()




def sample_from_buffers(replay_buffer_list, batch_size): # to replace: replay_buffer.sample(batch_size)
    draws_from_buffer = [0]*len(replay_buffer_list)
    for k in range(batch_size):
        draws_from_buffer[random.choice(range(len(replay_buffer_list)))] += 1   #select one
    obs_t_list = []
    act_t_list = []
    rew_t_list = []
    obs_tp1_list = []
    done_mask_list = []
    for idx, replay_buffer in enumerate(replay_buffer_list):
        if draws_from_buffer[idx]>0:
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(draws_from_buffer[idx])
            obs_t_list += [obs_t]
            act_t_list += [act_t]
            rew_t_list += [rew_t]
            obs_tp1_list += [obs_tp1]
            done_mask_list += [done_mask]
    return np.concatenate(obs_t_list, axis=0), np.concatenate(act_t_list, axis=0), np.concatenate(rew_t_list, axis=0), np.concatenate(obs_tp1_list, axis=0), np.concatenate(done_mask_list, axis=0)


def dqn_learning(env,
          env_id,
          q_func,
          optimizer_spec,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=100000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          double_dqn=False,
          dueling_dqn=False,
          self_channel=True):
    """Run Deep Q-learning algorithm.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    env_id: string
        gym environment id for model saving.
    q_func: function
        Model to use for computing the q function.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """


    ###############
    # BUILD MODEL #
    ###############


    img_h, img_w = env.shape
    img_c = 3
    input_shape = (img_h, img_w, frame_history_len * img_c)
    assert frame_history_len==1  #For now dont consider history. easily extensible by having agent class keep track of its locations
    in_channels = input_shape[2]
    num_actions = env.n

    # define Q target and Q
    Q = q_func(in_channels if self_channel else in_channels-frame_history_len, num_actions).type(dtype)
    Q_target = q_func(in_channels if self_channel else in_channels-frame_history_len, num_actions).type(dtype)

    # initialize optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # create replay buffer
    replay_buffer_list = [ReplayBuffer(replay_buffer_size, frame_history_len) for kk in range(len(env.agentlist))]

    ######

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.get_obs_for_qlearning()
    LOG_EVERY_N_STEPS = 1000
    SAVE_MODEL_EVERY_N_STEPS = 5000

    rewardslist = []
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            pdb.set_trace()
            break

        ### 2. Step the env and store the transition
        # store last frame, returned idx used later
        last_stored_frame_idx = [replay_buffer_list[k].store_frame(last_obs[k,:,:,:]) for k in range(last_obs.shape[0])]
        #last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        observations = env.get_obs_for_qlearning()
        assert all([kk.alive for kk in env.agentlist])  #assert everyones alive

        # before learning starts, choose actions randomly
        if t < learning_starts:
            action_list = [np.random.randint(num_actions) for i in range(len(env.agentlist))]
            threshold = 1
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs_ = np.swapaxes(np.swapaxes(observations, 2, 3), 1, 2)
                obs = torch.from_numpy(obs_).type(dtype) / 255.0
                #TODO: checkkk
                q_value_all_actions = Q(Variable(obs, volatile=True)).cpu()
                action_list = ((q_value_all_actions).data.max(1)[1]).numpy().squeeze().tolist()
            else:
                #action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]
                action_list = [np.random.randint(num_actions) for i in range(len(env.agentlist))]

        #pdb.set_trace()
        obs, reward, done = env.step(action_list)
        rewardslist += [reward]
        if t%20 == 0:
            print t, learning_starts, np.mean(reward), threshold, self_channel

        # clipping the reward, noted in nature paper
        reward = np.clip(reward, -1.0, 1.0)

        # store effect of action
        for idx, last_st_idx in enumerate(last_stored_frame_idx):
            replay_buffer_list[idx].store_effect(last_st_idx, action_list[idx], reward[idx], done[idx])

        # reset env if reached episode boundary
        for idx, dd in enumerate(done):
            if dd:
                currag = env.agentlist[idx]
                env.agentlist[idx] = currag.__class__((0,0,0), (255,255,255), (random.randint(0,img_h-1),random.randint(0,img_w-1)))

        # update last_obs
        last_obs = obs

        ### 3. Perform experience replay and train the network.
        # if the replay buffer contains enough samples...
        if (t > learning_starts and
                t % learning_freq == 0):
                #and replay_buffer.can_sample(batch_size)):

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = sample_from_buffers(replay_buffer_list, batch_size)# replay_buffer.sample(batch_size)
            obs_t = Variable(torch.from_numpy(obs_t)).type(dtype) / 255.0
            act_t = Variable(torch.from_numpy(act_t)).type(dlongtype)
            rew_t = Variable(torch.from_numpy(rew_t)).type(dtype)
            obs_tp1 = Variable(torch.from_numpy(obs_tp1)).type(dtype) / 255.0
            done_mask = Variable(torch.from_numpy(done_mask)).type(dtype)

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            #pdb.set_trace()
            q_values = Q(obs_t)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()
            if t%20==0:
                print(torch.max(q_values, 1)[1].cpu().data.numpy().squeeze().tolist())

            if (double_dqn):
                # ---------------
                #   double DQN
                # ---------------

                # get the Q values for best actions in obs_tp1
                # based off the current Q network
                # max(Q(s', a', theta_i)) wrt a'
                q_tp1_values = Q(obs_tp1).detach()
                _, a_prime = q_tp1_values.max(1)

                # get Q values from frozen network for next state and chosen action
                # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
                q_target_tp1_values = Q_target(obs_tp1).detach()
                q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()

                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime

                error = rew_t + gamma * q_target_s_a_prime - q_s_a
            else:
                # ---------------
                #   regular DQN
                # ---------------

                # get the Q values for best actions in obs_tp1
                # based off frozen Q network
                # max(Q(s', a', theta_i_frozen)) wrt a'
                q_tp1_values = Q_target(obs_tp1).detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)

                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done_mask) * q_s_a_prime

                # Compute Bellman error
                # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
                error = rew_t + gamma * q_s_a_prime - q_s_a

            # clip the error and flip
            clipped_error = -1.0 * error.clamp(-1, 1)

            # backwards pass
            optimizer.zero_grad()
            q_s_a.backward(clipped_error.data.unsqueeze(1))

            # update
            optimizer.step()
            num_param_updates += 1

            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

            # (2) Log values and gradients of the parameters (histogram)
            if t % LOG_EVERY_N_STEPS == 0:
                for tag, value in Q.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), t+1)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)
            #####

        ### 4. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            add_str = ''
            if (double_dqn):
                add_str = 'double'
            if (dueling_dqn):
                add_str = 'dueling'
            model_save_path = "models/%s_%s_%d_%s.model" %(str(env_id), add_str, t, str(time.ctime()).replace(' ', '_'))
            torch.save(Q.state_dict(), model_save_path)

        #episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        episode_rewards = rewardslist
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush()

            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'learning_started': (t > learning_starts),
                'num_episodes': len(episode_rewards),
                'exploration': exploration.value(t),
                'learning_rate': optimizer_spec.kwargs['lr'],
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, t+1)

            if len(episode_rewards) > 0:
                info = {
                    'last_episode_rewards': np.mean(episode_rewards[-1]),
                }

                #pdb.set_trace()
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)

            if (best_mean_episode_reward != -float('inf')):
                info = {
                    'mean_episode_reward_last_100': mean_episode_reward,
                    'best_mean_episode_reward': best_mean_episode_reward
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)