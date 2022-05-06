"""
the binomial tree model:
N: steps
R: risk-free growth factor
u: stock up-move rate
    default: 1 + N**0.5 * (R-1)
        after trying some value of u, I find it suitable to choose N**0.5 as parameter so that p is close to 0.5 when N is large
d: stock down-move rate
    d = 1 / u
p: risk-neutral probability
    p = (R - d) / (u - d)
q: probability of up-move
    default: p
V_u: value of option at t+1 if up-move
V_d: value of option at t+1 if down-move
V_t: value of Eur-option at t
    V_t = (q * V_u + (1 - q) * V_d) / R
delta_t: ratio of stock under delta-hedging
    delta_t = (V_u - V_d) / ((u - d) * S_t)


the ATM-European call option:
tenor: T = 1 year
one-year risk-free interest rate: r_f = 5%
initial stock price: S_0 = 50 $
strike price: K = S_0
stock price at maturity: S_T
value at maturity V_T = max(S_T - K, 0)


hedging of call seller:
pai_t: value of assets of seller
    pai_0 = delta_0 * S_0 + M_0 - V_0 = 0
    pai_t = delta_t_1 * S_t + M_t_1 * R - V_t


policy-gradient model:
state: (t, stock_price)
reward: pai_t
action: choose delta_t
    it is continuous (chapter 13.7 in textbook)
    assume normal distribution N(miu, sigma)
    policy parameter vector: theta = [theta_miu, theta_sigma]
    state feature vector: X_miu, X_sigma
        in our problem, each state have 2 features (S_t, t), but S_t is much more important than t, because R is so small
            that under the same S_t, pai_t differs just a little tiny from different t.
        so we assume X_miu and X_sigma are the function of S_t
        when S_t is much higher than K, we are sure to hold 1 share, so X_miu should be large and X_sigma should be small.
        when much lower, sure to hold 0.
        when close, not sure, means X_sigma should be large, and X_miu should be close to 0.5
        as a result, set
        X_miu = sigmoid((S_t / K - 1) * k)
            where sigmoid(x) = 1 / (1 + exp(-x))
            k is a constant parameter, maybe 10 is good
        X_sigma = 1 / (10 + absolute(S_t / K - 1) * k)
            k is a constant parameter, maybe 100 is good
    miu = theta_miu * X_miu
    sigma = theta_sigma * X_sigma
R = (1+r_f)**(1/N)
"""
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 365
r_f = 0.05
S_0 = 50
K = S_0
R = (1 + r_f) ** (1 / N)
u = 1 + N ** 0.5 * (R - 1)
d = 1 / u
q = (R - d) / (u - d)
sigma_min = 0.02

def test_q(q=q, n=1000):
    l = np.zeros((n, N))
    for i in tqdm(range(n)):
        s = 1
        for j in range(N):
            if np.random.binomial(1, q) == 1:
                s = s * u
            else:
                s = s * d
            l[i, j] = s
    return l[:, -1].mean(), l.max()


# print(test_q(q), q, u)


stock_prices = np.zeros((N, N))  # (time-step, number of up-move) 行往下是时间顺序，列往右是上涨次数
stock_prices[0, 0] = S_0
for i in range(1, N):
    for j in range(i + 1):
        stock_prices[i, j] = stock_prices[0, 0] * u ** j * d ** (i - j)

V_t = np.zeros((N, N))
V_t[-1, :] = np.max((stock_prices[-1, :] - K, np.zeros(N)), axis=0)
for i in range(N - 2, -1, -1):
    for j in range(i + 1):
        V_t[i, j] = (q * V_t[i + 1, j + 1] + (1 - q) * V_t[i + 1, j]) / R

delta_hedging_ratio = np.zeros((N, N))
for i in range(N - 1):
    for j in range(i + 1):
        delta_hedging_ratio[i, j] = (V_t[i + 1, j + 1] - V_t[i + 1, j]) / ((u - d) * stock_prices[i, j])
print('ok')


def sigmoid(x):
    return 1/(1+np.exp(-x))


class BinomialModel:
    """

    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = 0
        self.s = S_0
        self.pai = 0
        self.pai_delta_hedging = 0
        self.up = 0
        self.M = 0  # money market account
        self.M_delta_hedging = 0
        self.actions_delta_hedging = []
    def step(self, actions):
        """
        Args:
            actions(list)
        Returns:
            tuple of (reward, reward_delta_hedging, stock_price, episode terminated?)
        """
        # delta hedging
        action_delta_hedging = delta_hedging_ratio[self.state, self.up]
        self.actions_delta_hedging.append(action_delta_hedging)

        if self.state ==0 :
            self.M = V_t[0,0] - actions[0] * stock_prices[0,0]
            self.M_delta_hedging = V_t[0, 0] - self.actions_delta_hedging[0] * stock_prices[0, 0]
        else:
            self.pai = actions[self.state - 1] * self.s + self.M * R - V_t[self.state, self.up]
            self.pai_delta_hedging=self.actions_delta_hedging[self.state - 1] * self.s + self.M_delta_hedging * R - V_t[self.state, self.up]
            self.M = (actions[self.state - 1] - actions[self.state]) * self.s + self.M * R
            self.M_delta_hedging = (self.actions_delta_hedging[self.state - 1] - self.actions_delta_hedging[self.state]) * self.s + self.M_delta_hedging * R
        self.state+=1
        self.up += np.random.uniform()<=q
        self.s = S_0 * u**self.up * d**(self.state-self.up)
        if self.state == N-1:
            # terminal state
            self.pai = actions[self.state - 1] * self.s + self.M * R - V_t[self.state, self.up]
            self.pai_delta_hedging = self.actions_delta_hedging[self.state - 1] * self.s + self.M_delta_hedging * R - \
                                     V_t[self.state, self.up]
            return self.pai, self.pai_delta_hedging, self.s, True
        else:
            return 0, 0, self.s, False



class ReinforceAgent:
    """
    ReinforceAgent that follows algorithm
    'REINFORNCE Monte-Carlo Policy-Gradient Control (episodic)'
    """
    def __init__(self, alpha, gamma=1/R):
        self.theta_miu = 1
        self.theta_sigma = 1
        self.alpha = alpha
        self.gamma = gamma
        self.s = S_0
        self.x_miu = sigmoid((self.s / K - 1) * 10)
        # self.x_sigma = 1 / (10 + np.abs(self.s / K - 1) * 100)
        # self.x_sigma = np.abs(self.s / K - 1) * 2
        self.x_sigma = np.log(0.1)
        self.rewards = []
        self.actions = []
        self.actions_delta_hedging = []
        self.miu = self.theta_miu * self.x_miu
        self.sigma = np.exp(self.theta_sigma * self.x_sigma)

    def choose_action(self, reward):
        if reward is not None:
            self.rewards.append(reward)

        action = np.random.normal(self.miu, self.sigma)
        if action <0:
            action=0
        elif action >1:
            action =1
        self.actions.append(action)

        return action

    def episode_end(self, last_reward, S_t):
        self.rewards.append(last_reward)

        # learn theta
        G = np.zeros(len(self.rewards))
        G[-1] = self.rewards[-1]

        for i in range(2, len(G) + 1):
            G[-i] = self.gamma * G[-i + 1] + self.rewards[-i]

        gamma_pow = 1

        for i in range(len(G)):
            # print(i,self.miu,self.sigma)
            self.s = S_t[i]
            self.x_miu = sigmoid((self.s / K - 1) * 10)
            # self.x_sigma = np.log(1 / (10 + np.abs(self.s / K - 1) * 100))
            # self.x_sigma = np.abs(self.s / K - 1) * 2
            grad_ln_pi_miu = 1 / self.sigma ** 2 * (self.actions[i] - self.miu) * self.x_miu
            grad_ln_pi_sigma = ((self.actions[i]-self.miu)**2/self.sigma**2-1)*self.x_sigma
            update_miu = self.alpha * gamma_pow * G[i] * grad_ln_pi_miu
            update_sigma = self.alpha * gamma_pow * G[i] * grad_ln_pi_sigma
            self.theta_miu += update_miu
            self.theta_sigma += update_sigma
            self.miu = self.theta_miu * self.x_miu
            self.sigma = np.exp(self.theta_sigma * self.x_sigma)
            if self.sigma < sigma_min:
                self.sigma = sigma_min
            if self.miu > 0.95:
                self.miu = 0.95
            elif self.miu < 0.05:
                self.miu = 0.05
            gamma_pow *= self.gamma

        self.rewards = []
        self.actions = []


def trial(num_episodes, agent_generator):
    # np.random.seed(123)
    env = BinomialModel()
    agent = agent_generator()

    rewards = np.zeros(num_episodes)
    rewards_delta_hedging = np.zeros(num_episodes)
    for episode_idx in range(num_episodes):
        reward = None
        env.reset()
        actions = []
        S_t = []

        while True:
            action = agent.choose_action(reward)
            actions.append(action)
            reward, reward_delta_hedging, stock_price, episode_end = env.step(actions)
            S_t.append(stock_price)

            if episode_end:
                agent.episode_end(reward, S_t)
                break

        rewards[episode_idx] = reward
        rewards_delta_hedging[episode_idx] = reward_delta_hedging

    return rewards, rewards_delta_hedging

# print(trial(1000,lambda :ReinforceAgent(alpha=2e-4, gamma=1/R)))


def figure_13_1():
    import pandas as pd

    num_trials = 100
    num_episodes = 1000
    gamma = 1/R
    agent_generators = [lambda : ReinforceAgent(alpha=2e-4, gamma=gamma),
                        lambda : ReinforceAgent(alpha=2e-5, gamma=gamma),
                        lambda : ReinforceAgent(alpha=2e-3, gamma=gamma)]
    labels = ['alpha = 2e-4',
              'alpha = 2e-5',
              'alpha = 2e-3']
    result = pd.DataFrame(columns=labels.append('delta_hedging'))

    rewards = np.zeros((len(agent_generators), num_trials, num_episodes))
    rewards_delta_hedging = np.zeros((len(agent_generators), num_trials, num_episodes))

    for agent_index, agent_generator in enumerate(agent_generators):
        for i in tqdm(range(num_trials)):
            reward, reward_delta_hedging = trial(num_episodes, agent_generator)
            rewards[agent_index, i, :] = reward
            rewards_delta_hedging[agent_index, i, :] = reward_delta_hedging

    # plt.plot(np.arange(num_episodes) + 1, np.zeros(num_episodes), ls='dashed', color='red', label='0')
    for i, label in enumerate(labels):
        plt.plot(np.arange(num_episodes) + 1, rewards[i].mean(axis=0), label=label)
        result.iloc[:,i] = rewards[i].mean(axis=0)

    plt.plot(np.arange(num_episodes) + 1, rewards_delta_hedging[0].mean(axis=0), label='delta_hedging')
    result.iloc[:, 3] = rewards_delta_hedging[0].mean(axis=0)
    result.to_csv('results.csv')
    plt.ylabel('total reward on episode')
    plt.xlabel('episode')
    plt.legend(loc='lower right')

    plt.savefig('hw3_1.png')
    plt.close()
figure_13_1()