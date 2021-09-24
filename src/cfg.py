import sys
from pathlib import Path

import torch

from utils.checkpoints import Checkpoints

model_name = sys.argv[1] if len(sys.argv) >= 2 else "default"
model_dir = Path("models") / model_name
log_dir = model_dir / "logs"
checkpoints = Checkpoints(model_dir / "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 47

# seq2seq
EncoderLSTM = {
    "embedding_dim": 64,
    "hidden_size": 128,
    "dropout_p": 0.0,
}
DecoderLSTM = {
    "embedding_dim": 64,
    "hidden_size": 256,
}

# training
learning_rate = 1e-2
max_recipe_len = 100
batch_size = 16
teacher_forcing_p = 1
