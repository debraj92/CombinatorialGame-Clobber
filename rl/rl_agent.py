import random
import torch
import torch.nn as nn
from collections import namedtuple, deque
import sys
import copy
import pickle
from tqdm import tqdm
import numpy as np

sys.path.append("../")
from rl.clobber_environment import ClobberEnvironment
from rl.models import DQN

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "action_mask")
)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(
        self,
        board_size=10,
        memory_size=10000,
        gamma=0.95,
        epsilon_start=0.95,
        epsilon_end=0.05,
        epsilon_decay=200,
        learning_rate=1e-3,
        evaluation_db=None,
    ):
        # Setup Environment
        self.board_size = board_size
        self.environment = ClobberEnvironment(self.board_size)
        self.action_map = self.environment.get_action_map()
        self.reverse_action_map = {value: key for key, value in self.action_map.items()}

        # Setup policy & target networks
        # Input Size = Board Size + 1 to indicate current player
        self.policy_network = DQN(
            output_size=len(self.action_map), max_board_size=self.board_size
        )
        self.target_network = DQN(
            output_size=len(self.action_map), max_board_size=self.board_size
        )
        # We train only the policy network;
        # target network uses the policy network's weights
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.policy_network.train()
        self.target_network.eval()

        # Setup optimizer & memory
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=learning_rate
        )
        self.memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        # We multiply decay * board size to ensure significant exploration
        self.epsilon_decay = epsilon_decay * self.board_size
        self.iterations = 0

        # Setup device
        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")

        # Move models to GPU if possible
        self.policy_network.to(self.device)
        self.target_network.to(self.device)

        # Load an evaluation DB if provided
        if evaluation_db:
            with open(evaluation_db, "rb") as fp:
                self.evaluation_db = pickle.load(fp)
        else:
            self.evaluation_db = None

    def train(
        self,
        num_episodes,
        target_model_update=100,
        batch_size=32,
        evaluation_interval=1000,
        evaluation_episodes=100,
    ):
        all_rewards = []
        average_rewards = []
        all_losses = []
        last_print = 0

        for _ in tqdm(range(num_episodes)):
            reward, loss = self.train_on_one_episode(
                target_model_update=target_model_update,
                batch_size=batch_size,
            )
            all_rewards.append(reward)
            all_losses.extend(loss)
            average_reward = np.array(all_rewards).mean()
            average_rewards.append(average_reward)
            if self.iterations - last_print >= evaluation_interval:
                self.policy_network.eval()
                last_print = self.iterations
                (
                    random_v_random_rewards,
                    agent_v_random_rewards,
                    agent_v_agent_rewards,
                    optimal_play_rewards,
                ) = self.evaluate(evaluation_episodes)
                tqdm.write(
                    f"Iteration {self.iterations}:\n[Random vs Random] {random_v_random_rewards}\t[Agent vs Random] {agent_v_random_rewards}\t[Agent vs Agent] {agent_v_agent_rewards}\t[Optimal Play] {optimal_play_rewards}"
                )
                self.policy_network.train()
        return self.iterations, average_rewards, all_losses

    def compute_action_mask(self, legal_moves, mask_value=-1e9):
        return torch.tensor(
            [0 if action in legal_moves else mask_value for action in self.action_map]
        ).to(self.device)

    def train_on_one_episode(self, target_model_update, batch_size):
        board, player = self.environment.reset()
        all_losses = []
        done = False
        state = torch.tensor(board + [player]).unsqueeze(0).float()

        while not done:
            # Get legal actions in this state
            legal_actions = self.environment.get_legal_moves()

            # Make action mask
            action_mask = self.compute_action_mask(legal_actions)

            ## Use Epsilon-Greedy Policy
            epsilon_threshold = self.epsilon_end + (
                self.epsilon_start - self.epsilon_end
            ) * np.exp(-1.0 * self.iterations / self.epsilon_decay)

            # Choose random action
            if random.random() > epsilon_threshold:
                action = torch.tensor(self.action_map[random.choice(legal_actions)])
            # Use network for action selection
            else:
                # Get action probabilities from model
                with torch.no_grad():
                    action = self.policy_network(state.unsqueeze(0).to(self.device))
                    # Mask Moves & pick greedy action
                    action = (action + action_mask).argmax().cpu()

            # Play move
            _, _, reward, done = self.environment.step(
                self.reverse_action_map[int(action)]
            )

            # Else simulate opponent move as part of the environment
            if not done:
                # Get legal actions in this state
                legal_actions = self.environment.get_legal_moves()

                # Pick random move
                opponent_action = random.choice(legal_actions)

                # Play move
                board, player, _, done = self.environment.step(opponent_action)
                next_state = torch.tensor(board + [player]).unsqueeze(0).float()

            # If we're done the next state is the terminal state -> None
            if done:
                next_state = None

            # Save transition in memory
            self.memory.push(state, action, next_state, reward, action_mask)
            state = next_state

            # Train model on one batch of data
            loss = self.train_model(batch_size)
            if loss:
                all_losses.append(loss)

            # Update target network every target_model_update steps
            if self.iterations % target_model_update == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

            self.iterations += 1

        return reward, all_losses

    def train_model(self, batch_size):
        # Only train if we have enough samples for a batch of data
        if len(self.memory) < batch_size:
            return

        # Create batch from memory
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))  # (state, action, next_state, reward)

        # Separate out data into separate batches
        state_batch = torch.stack(batch.state).float().to(self.device)
        action_batch = torch.stack(batch.action).unsqueeze(dim=1).to(self.device)
        reward_batch = torch.tensor(batch.reward).to(self.device)

        # Final states would be None, so we mask those out as the value is 0 there
        non_final_mask = torch.tensor(
            tuple(map(lambda state: state is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        # Handle edge cases where all the samples are final states
        if not any(non_final_mask):
            return

        # Compute the next state transitions for non-final states
        non_final_next_states = []
        non_final_action_masks = []
        for mask, state in zip(batch.action_mask, batch.next_state):
            if state is not None:
                non_final_next_states.append(state)
                non_final_action_masks.append(mask)
        non_final_next_states = (
            torch.stack(non_final_next_states).float().to(self.device)
        )
        non_final_action_masks = (
            torch.stack(non_final_action_masks).float().to(self.device)
        )

        # Compute Q(s_t, a)
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Which is just max_a Q(s'_t+1, a)
        # We compute this using the target_network for stability.
        # V(s) = 0 for all final states
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            (self.target_network(non_final_next_states) + non_final_action_masks)
            .max(1)[0]
            .detach()
        )

        # Compute the expected Q values: (max_a Q(s_t+1, a) * gamma) + reward
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = nn.functional.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().cpu()

    def predict(self, state, action_mask):
        with torch.no_grad():
            action = self.policy_network(state.unsqueeze(0))
            action = (action + action_mask).argmax().cpu()
        return self.reverse_action_map[int(action)]

    def save_for_deployment(self, save_path):
        # Save policy_network, action_map and reverse_action_map
        torch.save(
            {
                "policy_network": self.policy_network.state_dict(),
                "action_map": self.action_map,
                "reverse_action_map": self.reverse_action_map,
                "board_size": self.board_size,
            },
            save_path,
        )

    def play_one_episode(self, first_agent="model", second_agent="random", state={}):
        if state:
            board, player = self.environment.reset_to_board(**state)
        else:
            board, player = self.environment.reset()
        done = False

        while not done:
            # Get legal actions in this state
            legal_actions = self.environment.get_legal_moves()

            # Choose move
            if first_agent == "random":
                action = random.choice(legal_actions)
            else:
                state = torch.tensor(board + [player]).unsqueeze(0).float()
                action_mask = self.compute_action_mask(legal_actions)
                action = self.predict(state, action_mask)

            # Play move
            board, player, reward, done = self.environment.step(action)

            ## Opponent Move
            # Make sure game is not already over
            if not done:
                # Get legal actions in this state
                legal_actions = self.environment.get_legal_moves()

                # Choose move
                if second_agent == "random":
                    action = random.choice(legal_actions)
                else:
                    state = torch.tensor(board + [player]).unsqueeze(0).float()
                    action_mask = self.compute_action_mask(legal_actions)
                    action = self.predict(state, action_mask)

                # Play move
                board, player, reward, done = self.environment.step(action)

        return reward

    def evaluate(self, num_episodes):
        random_v_random = 0
        agent_v_random = 0
        agent_v_agent = 0
        optimal_play_reward = 0
        # Play the same N episodes for each comparison
        for _ in range(num_episodes):
            board, first_player = self.environment.reset(hard_reset=True)
            state = {
                "board": board,
                "first_player": first_player,
            }
            random_v_random_reward = self.play_one_episode(
                first_agent="random", second_agent="random", state=copy.deepcopy(state)
            )
            agent_v_random_reward = self.play_one_episode(
                first_agent="model", second_agent="random", state=copy.deepcopy(state)
            )
            agent_v_agent_reward = self.play_one_episode(
                first_agent="model", second_agent="model", state=copy.deepcopy(state)
            )

            random_v_random += random_v_random_reward if random_v_random_reward > 0 else 0
            agent_v_random += agent_v_random_reward if agent_v_random_reward > 0 else 0
            agent_v_agent += agent_v_agent_reward if agent_v_agent_reward > 0 else 0

            if self.evaluation_db:
                outcome = self.evaluation_db[tuple(board)]
                if outcome[first_player - 1] == "1":
                    optimal_play_reward += 1
        return (
            (random_v_random * 100)/ num_episodes,
            (agent_v_random * 100)/ num_episodes,
            (agent_v_agent * 100)/ num_episodes,
            (optimal_play_reward * 100)/ num_episodes,
        )
