import random
import torch
import torch.nn as nn
from collections import namedtuple, deque
import sys
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
        hidden_size=64,
        memory_size=10000,
        gamma=0.95,
        epsilon_start=0.95,
        epsilon_end=0.05,
        epsilon_decay=200,
    ):
        # Setup Environment
        self.environment = ClobberEnvironment(board_size)
        self.action_map = self.environment.get_action_map()
        self.reverse_action_map = {value: key for key, value in self.action_map.items()}

        # Setup policy & target networks
        # Input Size = Board Size + 1 to indicate current player
        self.policy_network = DQN(len(self.action_map))
        self.target_network = DQN(len(self.action_map))
        # We train only the policy network;
        # target network uses the policy network's weights
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.policy_network.train()
        self.target_network.eval()

        # Setup optimizer & memory
        self.optimizer = torch.optim.RMSprop(self.policy_network.parameters())
        self.memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Setup device
        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")

        # Move models to GPU if possible
        self.policy_network.to(self.device)
        self.target_network.to(self.device)

    def train(
        self,
        num_episodes,
        print_every=1000,
        target_model_update=100,
        batch_size=32,
    ):
        iterations = 0
        all_rewards = []
        all_losses = []
        last_print = 0

        for _ in tqdm(range(num_episodes)):
            iterations, reward, loss = self.play_one_episode(
                iterations=iterations,
                target_model_update=target_model_update,
                batch_size=batch_size,
            )
            all_rewards.append(reward)
            all_losses.append(loss)
            if iterations - last_print >= print_every:
                average_reward = np.array(all_rewards).mean()
                last_print = iterations
                tqdm.write(f"Average Reward: {average_reward}")
        return iterations, all_rewards, all_losses

    def play_one_episode(self, target_model_update, iterations, batch_size):
        board, player = self.environment.reset()
        state = torch.tensor(board + [player]).unsqueeze(0).float()
        all_losses = []
        done = False

        while not done:
            # Get legal actions in this state
            legal_actions = self.environment.get_legal_moves()

            ## Use Epsilon-Greedy Policy
            eps_threshold = self.epsilon_end + (
                self.epsilon_start - self.epsilon_end
            ) * np.exp(-1.0 * iterations / self.epsilon_decay)

            # Choose random action
            if random.random() > eps_threshold:
                action = torch.tensor(self.action_map[random.choice(legal_actions)])
                action_mask = torch.zeros((len(self.action_map)))
            # Use network for action selection
            else:
                # Make action mask
                action_mask = torch.tensor(
                    [
                        0 if action in legal_actions else -1e9
                        for action in self.action_map
                    ]
                )

                # Get action probabilities from model
                with torch.no_grad():
                    action = self.policy_network(state.to(self.device))

                # Mask Moves & pick greedy action
                action = (action + action_mask).argmax()

            # Play move
            board, player, reward, done = self.environment.step(
                self.reverse_action_map[int(action)]
            )
            next_state = torch.tensor(board + [player]).unsqueeze(0).float()

            # If we're done the next state is the terminal state -> None
            if done:
                next_state = None

            # Save transition in memory
            self.memory.push(state, action, next_state, reward, action_mask)

            # Train model on one batch
            loss = self.train_model(batch_size)
            if loss:
                all_losses.append(loss)

            ## Opponent Move
            # Make sure game is not already over
            if not done:
                # Get legal actions in this state
                legal_actions = self.environment.get_legal_moves()

                # Pick Random move
                action = random.choice(legal_actions)

                # Play move
                board, player, reward, done = self.environment.step(action)
                state = torch.tensor(board + [player]).unsqueeze(0).float()

            # Update target network every target_model_update steps
            if iterations % target_model_update == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

            iterations += 1

        return iterations, reward, np.mean(all_losses)

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

    def predict(self, board, current_player, legal_moves):
        state = torch.tensor(board + [current_player]).to(self.device)
        action_mask = torch.tensor(
            [0 if action in legal_moves else -1e9 for action in self.action_map]
        ).to(self.device)
        with torch.no_grad():
            action = self.policy_network(state.unsqueeze(0).float())
        action += action_mask
        return self.reverse_action_map[int(action.argmax())]

    def policy_model_to_disk(self, save_path):
        # Save policy_network, action_map and reverse_action_map
        pass

    def validate(self):
        pass
