import gym 
import time
import argparse
import numpy as np
import torch

from lib import wrappers
from lib import dqn

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 400

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    parser.add_argument("-e", "--env", default = DEFAULT_ENV_NAME,
                        help="Environment name to use, default="
                        +DEFAULT_ENV_NAME
                       )
    parser.add_argument("-r", "--record", help="Directory for video")
    args = parser.parse_args()
    
    
    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = dqn.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1

        state, reward, done, _ = env.step(action)
        total_reward =+ reward
        if done:
            break

        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
        
        env.env.ale.saveScreenPNG('test_image2.png')
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()


