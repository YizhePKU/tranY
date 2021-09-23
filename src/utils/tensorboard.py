from pathlib import Path
from datetime import datetime
import torch
import torch.utils.tensorboard

log_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d %H:%M:%S")

writer = torch.utils.tensorboard.SummaryWriter(
    log_dir=log_dir,
    max_queue=1,
)

profiler = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    with_stack=True,
)
