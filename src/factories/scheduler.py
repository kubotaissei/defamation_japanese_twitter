from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def get_scheduler(config, optimizer):
    if config.class_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=config.num_train_steps,
        )
    elif config.class_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=config.num_train_steps,
            num_cycles=config.num_cycles,
        )
    return scheduler
