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
    "nlayers": 1,
    "dropout_p": 0.3,
    "bidirectional": True,
}
DecoderLSTM = {
    "embedding_dim": 64,
    "hidden_size": 256,
    "nlayers": 1,
    "dropout_p": 0.3,
    "context_size": 256, # 2 * encoder.hidden_size
}

# training
learning_rate = 1e-3
batch_size = 32
max_recipe_len = 100