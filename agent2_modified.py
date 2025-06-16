import numpy as np
import matplotlib.pyplot as plt

class ContextualBanditAgent:
    def __init__(self, n_arms, n_contexts, budgets, epsilon=0.1, strategy='epsilon_greedy'):
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.epsilon = epsilon
        self.strategy = strategy
        self.budgets = np.array(budgets)
        self.displays = np.zeros(n_arms)  # Tracks how many times each ad was shown
        self.counts = np.zeros((n_contexts, n_arms))
        self.values = np.zeros((n_contexts, n_arms))
        self.total_reward = 0
        self.actions = []
        self.rewards = []

    def select_action(self, context):
        if self.strategy == 'epsilon_greedy':
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_arms)
            return np.argmax(self.values[context])

        elif self.strategy == 'ucb':
            total_counts = np.sum(self.counts[context])
            if total_counts == 0:
                return np.random.randint(self.n_arms)
            confidence_bounds = self.values[context] + np.sqrt(2 * np.log(total_counts + 1) / (self.counts[context] + 1e-5))
            return np.argmax(confidence_bounds)

        elif self.strategy == 'softmax':
            preferences = self.values[context]
            exp_preferences = np.exp(preferences - np.max(preferences))  # stability
            probs = exp_preferences / np.sum(exp_preferences)
            return np.random.choice(self.n_arms, p=probs)

    def update(self, context, action, reward):
        if self.displays[action] < self.budgets[action]:
            self.displays[action] += 1
        else:
            reward = max(0, reward - 0.5)  # Penalty for over-budget
        self.counts[context, action] += 1
        self.values[context, action] += (reward - self.values[context, action]) / self.counts[context, action]
        self.total_reward += reward
        self.actions.append(action)
        self.rewards.append(reward)


def simulate_bandit(true_ctrs, budgets, epsilon, strategy, n_rounds=1000):
    n_contexts = len(true_ctrs)
    n_arms = len(true_ctrs[0])
    agent = ContextualBanditAgent(n_arms, n_contexts, budgets, epsilon, strategy)
    regrets = []

    for t in range(n_rounds):
        context = np.random.randint(n_contexts)
        action = agent.select_action(context)
        reward = np.random.rand() < true_ctrs[context][action]
        agent.update(context, action, reward)
        regret = np.max(true_ctrs[context]) - true_ctrs[context][action]
        regrets.append(regret)

    return agent, np.cumsum(regrets)


# ---------- Experiment Configuration ----------
np.random.seed(42)
n_contexts = 3  # Teenagers, Adults, Seniors
n_arms = 10     # Number of ads
n_rounds = 1000
budgets = [100] * n_arms
true_ctrs = np.random.uniform(0.05, 0.5, size=(n_contexts, n_arms))

print("True CTRs per context:")
for i, group in enumerate(["Teenagers", "Adults", "Seniors"]):
    print(f"{group}: {np.round(true_ctrs[i], 2)}")

strategies = ['epsilon_greedy', 'ucb', 'softmax']
regret_curves = {}

# ---------- Run Simulations ----------
for strategy in strategies:
    agent, regrets = simulate_bandit(true_ctrs, budgets, epsilon=0.1, strategy=strategy, n_rounds=n_rounds)
    regret_curves[strategy] = regrets

# ---------- Plot Results ----------
plt.figure(figsize=(10, 5))
for strategy in strategies:
    plt.plot(regret_curves[strategy], label=strategy)
plt.xlabel("Rounds")
plt.ylabel("Cumulative Regret")
plt.title("Strategy Comparison on Contextual Bandit with Budgets")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
