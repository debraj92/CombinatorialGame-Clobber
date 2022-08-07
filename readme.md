# Better Move Ordering in Combinatorial Games via Learnable Heuristics in a 1D Clobber Solver
## Akash Saravanan (asaravan), Debraj Ray (debraj1)
---

## Run Clobber Solver with preferred move ordering

1. Install Python 3.8.
2. Install required libraries: `pip install -r requirements.txt`
3. Run the code: `python git_good.py BOARD PLAYER TIMEOUT --MOVE_ORDERING`

   Where, `--MOVE_ORDERING` can be one of `[--cnn_move_ordering, --rl_move_ordering, --default_move_ordering, --no_move_ordering]` 

   For example:
   ```
    python3 git_good.py BWBWBWBWBWBWBWBWBWBWBWBWBWBW W 100 --cnn_move_ordering
    W 21-22 4.279016017913818 90713

    python3 git_good.py BWBWBWBWBWBWBWBWBWBWBWBWBWBW W 100 --rl_move_ordering
    W 13-14 4.321457862854004 74800
    
    python3 git_good.py BWBWBWBWBWBWBWBWBWBWBWBWBWBW W 100 --default_move_ordering
    W 21-22 7.818457841873169 173465
    
    python3 git_good.py BWBWBWBWBWBWBWBWBWBWBWBWBWBW W 100 --no_move_ordering
    W 21-22 54.99504590034485 1200074
   ```
   The input is formatted as follows:
   [python3] [file to execute.py] [starting 1D clobber board position] [player to start (Black / White)] [Maximum time to execute] [--type of move ordering]
   
   The output is formatted as follows:
   [Winner (Black / White)] [First winning move] [Time taken to run the heuristic search] [number of nodes of the game tree expanded]

## Model generation

We recommend training the models using a GPU as the training process would otherwise be slower.

### Training - CNN

1. Install Python 3.8.
2. Install required libraries (Not required if you installed the requirements from the inference section): `pip install -r rl/requirements.txt`
3. Modify `cnn_control.py` and choose training a fresh model or re-training a trained model.
4. The number of samples, sparsity of board and method of generation of synthetic samples can also be controlled.
5. Run training by running the `cnn_control.py` file.

### Inference - CNN

1. Modify `inference_tflite.py` file. Choose whether accuracy is tested for the black-cnn, white-cnn or both.
2. Control board size and sparsity (distribution of black, white pieces) and number of samples to be used for the accuracy check.
3. Run `inference_tflite.py`


### Training - RL

1. Install Python 3.8.
2. Install required libraries (Not required if you installed the requirements from the inference section): `pip install -r rl/requirements.txt`
3. Modify the config in `rl/main.py` to suit your requirements.
4. Run `rl/main.py`
5. Run `rl/pytorch_to_onnx.py` to convert the model to the ONNX format for faster inference.
6. Set the RL model path [line 35] in `boolean_negamax_tt.py` to this new model.

### For detailed information, please read the project report pdf.
