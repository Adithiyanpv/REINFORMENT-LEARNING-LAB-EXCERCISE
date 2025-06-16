import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)   # Number of times each arm was pulled
        self.values = np.zeros(n_arms)   # Estimated value (CTR) for each arm
        self.total_reward = 0
        self.actions = []
        self.rewards = []

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)  # Explore
        else:
            return np.argmax(self.values)  # Exploit

    def update(self, action, reward):
        self.counts[action] += 1
        self.values[action] += (reward - self.values[action]) / self.counts[action]
        self.total_reward += reward
        self.actions.append(action)
        self.rewards.append(reward)

def simulate_bandit(true_ctrs, epsilon, n_rounds=1000):
    n_arms = len(true_ctrs)
    agent = EpsilonGreedyAgent(n_arms, epsilon)
    optimal_arm = np.argmax(true_ctrs)
    regrets = []

    for t in range(n_rounds):
        action = agent.select_action()
        reward = np.random.rand() < true_ctrs[action]  # Simulated Bernoulli reward
        agent.update(action, reward)
        regret = true_ctrs[optimal_arm] - true_ctrs[action]
        regrets.append(regret)

    return agent, np.cumsum(regrets)

# ---------- Main Experiment ----------
np.random.seed(42)
n_arms = 10
true_ctrs = np.random.uniform(0.05, 0.5, n_arms)
print("True Click-Through Rates (CTR) per Ad:", np.round(true_ctrs, 2))

n_rounds = 1000
epsilons = [0.01, 0.1, 0.3]
agents = {}
regret_curves = {}

for epsilon in epsilons:
    agent, regrets = simulate_bandit(true_ctrs, epsilon, n_rounds)
    agents[epsilon] = agent
    regret_curves[epsilon] = regrets

# ---------- Plotting Results ----------
plt.figure(figsize=(12, 5))

# Plot cumulative regret
plt.subplot(1, 2, 1)
for epsilon in epsilons:
    plt.plot(regret_curves[epsilon], label=f'ε={epsilon}')
plt.title("Cumulative Regret")
plt.xlabel("Rounds")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.grid(True)

# Plot estimated CTRs vs true CTRs
plt.subplot(1, 2, 2)
bar_width = 0.25
x = np.arange(n_arms)
for i, epsilon in enumerate(epsilons):
    plt.bar(x + i * bar_width,
            agents[epsilon].values,
            width=bar_width,
            label=f'ε={epsilon}')

plt.axhline(np.max(true_ctrs), color='r', linestyle='--', label='Optimal CTR')
plt.xticks(x + bar_width, [f'Ad {i}' for i in range(n_arms)])
plt.ylabel("Estimated CTR")
plt.title("Estimated CTRs vs True CTR")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
