import torch
import sys

sys.path.append("../")
from rl.models import DQN


class DeployableAgent:
    def __init__(self, model_directory):
        # Setup device
        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu else "cpu")

        # Load data from disk
        saved_model = torch.load(model_directory, map_location=self.device)

        # Initialize Action Maps
        self.action_map = saved_model["action_map"]
        self.reverse_action_map = saved_model["reverse_action_map"]

        # Create model & restore weights
        self.policy_network = DQN(len(self.action_map)).to(self.device)
        self.policy_network.load_state_dict(saved_model["policy_network"])
        self.policy_network.eval()

    def compute_action_mask(self, legal_moves, mask_value=-1e9):
        return torch.tensor(
            [0 if action in legal_moves else mask_value for action in self.action_map]
        ).to(self.device)

    def predict(self, board, current_player, legal_moves):
        with torch.no_grad():
            state = torch.tensor(board + [current_player]).unsqueeze(0).float()
            action_mask = self.compute_action_mask(legal_moves)
            action = self.policy_network(state)
            action = int((action + action_mask).argmax().cpu())
        return self.reverse_action_map[action]
