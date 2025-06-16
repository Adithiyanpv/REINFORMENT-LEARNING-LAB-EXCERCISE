import numpy as np
import matplotlib.pyplot as plt

# --- Configurable Gridworld Setup ---
GRID_SIZE = 8  # Change to 6 or 8 as needed
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_DICT = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

GOAL_STATE = (GRID_SIZE - 1, GRID_SIZE - 1)
BONUS_STATES = [(2, 3), (5, 5)]
TRAP_STATES = [(1, 1), (3, 4)]

MAX_STEPS = 100

def is_valid_state(state):
    return 0 <= state[0] < GRID_SIZE and 0 <= state[1] < GRID_SIZE

def step(state, action, visited):
    move = ACTION_DICT[action]
    new_state = (state[0] + move[0], state[1] + move[1])

    reward = 0
    if not is_valid_state(new_state):
        new_state = state  # Wall hit: stay in place
        reward -= 1        # Penalty for bumping into wall
    elif new_state in visited:
        reward += 0        # No bonus for revisiting
    else:
        reward += 1        # Bonus for first-time visit

    if new_state == GOAL_STATE:
        reward += 10
        done = True
    elif new_state in BONUS_STATES:
        reward += 5
        done = False
    elif new_state in TRAP_STATES:
        reward -= 10
        done = False
    else:
        done = False

    return new_state, reward, done

# --- Agent Logic ---
def run_episode():
    state = (0, 0)
    total_reward = 0
    trajectory = [state]
    visited = set()
    visited.add(state)
    for _ in range(MAX_STEPS):
        action = np.random.choice(ACTIONS)
        next_state, reward, done = step(state, action, visited)
        visited.add(next_state)
        trajectory.append(next_state)
        total_reward += reward
        state = next_state
        if done:
            return trajectory, total_reward, True
    return trajectory, total_reward, False

# --- Run Multiple Episodes ---
success_count = 0
all_trajectories = []

for ep in range(100):
    traj, reward, reached_goal = run_episode()
    all_trajectories.append(traj)
    if reached_goal:
        success_count += 1
    print(f"Episode {ep + 1}: Reward = {reward}, Steps = {len(traj)}, Goal Reached = {reached_goal}")

# --- Summary ---
print(f"\nOver 100 episodes, agent reached the goal {success_count} times.")

# --- Heatmap of Last Episode ---
def plot_trajectory(trajectory):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for (x, y) in trajectory:
        grid[x, y] += 1

    plt.imshow(grid, cmap='Blues', origin='upper')
    plt.title("Agent Trajectory Heatmap (Last Episode)")
    plt.colorbar(label="Visits")
    plt.scatter(0, 0, c='green', s=100, label='Start')
    plt.scatter(GOAL_STATE[1], GOAL_STATE[0], c='red', s=100, label='Goal')
    for x, y in BONUS_STATES:
        plt.scatter(y, x, c='gold', s=80, marker='*', label='Bonus')
    for x, y in TRAP_STATES:
        plt.scatter(y, x, c='black', s=80, marker='x', label='Trap')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

# Plot last episode trajectory
plot_trajectory(all_trajectories[-1])
