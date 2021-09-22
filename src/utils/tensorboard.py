import torch
import torch.utils.tensorboard

writer = torch.utils.tensorboard.SummaryWriter()

profiler = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("runs"),
    with_stack=True,
    record_shapes=True,
)
