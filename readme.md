# Better Move Ordering in Combinatorial Games via Learnable Heuristics in a 1D Clobber Solver
## Akash Saravanan (asaravan), Debraj Ray (debraj1)
---

## Inference

1. Install Python 3.8.
2. Install required libraries: `pip install -r requirements.txt`
3. Run the code: `python git_good.py BOARD PLAYER TIMEOUT --MOVE_ORDERING`

   Where, `--MOVE_ORDERING` can be one of `[--cnn_move_ordering, --rl_move_ordering, --default_move_ordering, --no_move_ordering]` 

   For example:
   ```
    python .\git_good.py BWBWBWBWBWBWBWBWBWBWBWBWBW W 10 --cnn_move_ordering
    W 23-24 4.413733477554321 32887

    python .\git_good.py BWBWBWBWBWBWBWBWBWBWBWBWBW W 10 --rl_move_ordering
    W 23-22 4.388262097702026 25444
   ```

## Training

We recommend training the models using a GPU as the training process would otherwise be slower.

### Training - CNN

### Training - RL

1. Install Python 3.8.
2. Install required libraries (Not required if you installed the requirements from the inference section): `pip install -r rl/requirements.txt`
3. Modify the config in `rl/main.py` to suit your requirements.
4. Run `rl/main.py`
5. Run `rl/pytorch_to_onnx.py` to convert the model to the ONNX format for faster inference.
6. Set the RL model path [line 35] in `boolean_negamax_tt.py` to this new model.
