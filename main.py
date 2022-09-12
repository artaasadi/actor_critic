from fileinput import filename
import pybullet
import gym
import numpy as np
from sac_algo import Agent

if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = Agent(env= env, input_dims= env.observation_space.shape, n_actions= env.action_space.shape[0])
    n_games = 250
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observations = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.act(observations=observations)
            observations_, reward, done, info = env.step(action)
            score += reward
            agent.store(state= observations, action= action, reward= reward, state_= observations_, done= done)
            if not load_checkpoint:
                agent.learn()
            observations = observations_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score is ', score, 'and average score is : ', avg_score)
