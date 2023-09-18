import matplotlib.pyplot as plt 
import pickle
import numpy as np

roll = 50

vpg = pickle.load(open('./train_logs/vpg_rewards.pkl', 'rb'))
a2c = pickle.load(open('./train_logs/a2c_rewards.pkl', 'rb'))
ppo = pickle.load(open('./train_logs/ppo_rewards.pkl', 'rb'))
dqn = pickle.load(open('./train_logs/dqn_rewards.pkl', 'rb'))

plt.style.use('seaborn-v0_8-pastel')

plt.plot(np.convolve(vpg, np.ones(roll)/roll, mode='valid'), label='VPG')
plt.plot(np.convolve(a2c, np.ones(roll)/roll, mode='valid'), label='A2C')
plt.plot(np.convolve(ppo, np.ones(roll)/roll, mode='valid'), label='PPO')
plt.plot(np.convolve(dqn, np.ones(roll)/roll, mode='valid'), label='DQN')
plt.legend(loc='upper left')
plt.xlabel('Matches')
plt.ylabel(f'Rolling Avg Reward Past {roll} Matches')
plt.savefig('rewards.png')
plt.show()

