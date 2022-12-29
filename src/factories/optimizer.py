from torch.optim import Adam, SGD, AdamW


def get_adamw(params, lr, eps, betas, weight_decay):
    optimizer = AdamW(
        params=params, lr=lr, eps=eps, betas=betas, weight_decay=weight_decay
    )
    return optimizer


def get_optimizer(opt_class, **params):
    print("optimizer class:", opt_class)
    f = globals().get(opt_class)
    return f(**params)
