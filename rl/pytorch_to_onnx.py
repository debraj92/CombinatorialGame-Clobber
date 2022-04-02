import torch
import pickle
import onnx
import sys
import os

sys.path.append("./")
from models import DQN

MODEL_DIR = "model_size_25_v2/model.pt"
OUTPUT_DIR = "model_size_25_v2"

gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
saved_model = torch.load(MODEL_DIR, map_location=device)
board_size = saved_model["board_size"]
action_map = saved_model["action_map"]
reverse_action_map = saved_model["reverse_action_map"]
model = DQN(output_size=len(action_map), max_board_size=board_size)
model.load_state_dict(saved_model["policy_network"])
model.eval()
dummy = torch.tensor([1] * (board_size + 1)).unsqueeze(0).float()
torch.onnx.export(
    model, dummy.unsqueeze(0), os.path.join(OUTPUT_DIR, "model.onnx")
)

with open(os.path.join(OUTPUT_DIR, "model_config.pkl"), "wb") as fp:
    pickle.dump(
        {"action_map": action_map, "reverse_action_map": reverse_action_map, "board_size": board_size}, fp
    )

onnx_model = onnx.load(os.path.join(OUTPUT_DIR, "model.onnx"))
onnx.checker.check_model(onnx_model)
