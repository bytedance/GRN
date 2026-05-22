
def get_lambda(args):
    if args.scheduler == "linear":
        def lr_lambda(step):
            warmup_steps = args.warmup_steps
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.
        return lr_lambda
    else:
        raise NotImplementedError
