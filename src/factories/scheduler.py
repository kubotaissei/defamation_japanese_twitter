from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def get_scheduler(config, optimizer, num_train_steps):
    if config.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    elif config.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=num_train_steps,
            num_cycles=config.num_cycles,
        )
    return scheduler
