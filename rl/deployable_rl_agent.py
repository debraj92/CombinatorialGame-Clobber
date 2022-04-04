import pickle
import sys
import onnxruntime
import os

sys.path.append("../")


class DeployableAgent:
    def __init__(self, model_directory):

        with open(os.path.join(model_directory, "model_config.pkl"), "rb") as fp:
            config = pickle.load(fp)

        # Initialize Action Maps
        self.action_map = config["action_map"]
        self.reverse_action_map = config["reverse_action_map"]
        self.board_size = config["board_size"]
        self.session = onnxruntime.InferenceSession(
            os.path.join(model_directory, "model.onnx")
        )

    def compute_legal_move_ids(self, legal_moves):
        return set(self.action_map[legal_move] for legal_move in legal_moves)

    def move_ordering(self, board, current_player, legal_moves):
        """
        Returns an ordered list of moves to play.
        """
        state = [[[float(x) for x in board] + [float(current_player)]]]
        preds = self.session.run(None, {"input.1": state})
        legal_moves = self.compute_legal_move_ids(legal_moves)
        preds = preds[0][0].argsort().tolist()[::-1]
        return [
            self.reverse_action_map[action] for action in preds if action in legal_moves
        ]
