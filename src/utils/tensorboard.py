import cfg
import torch
import torch.utils.tensorboard

writer = torch.utils.tensorboard.SummaryWriter(
    log_dir=cfg.log_dir,
    max_queue=1,
)

profiler = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.log_dir),
    with_stack=True,
)
