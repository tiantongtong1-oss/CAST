"""Fixed-update scheduling windows, independent of target dataset size."""


def next_batch(loader, iterator):
    """Restart DataLoader rather than caching batches: augmentations are resampled."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        try:
            return next(iterator), iterator
        except StopIteration as error:
            raise ValueError('Cannot cycle an empty target loader') from error


def batches_for_window(loader, steps, iterator=None):
    iterator = iter(loader) if iterator is None else iterator
    for _ in range(steps):
        batch, iterator = next_batch(loader, iterator)
        yield batch, iterator


def schedule_metadata(args, native_batches):
    steps = args.target_steps_per_round or native_batches
    return dict(mode='fixed_updates' if args.target_steps_per_round else 'native_epochs',
                updates_per_round=steps, total_updates=args.epochs * steps,
                js_start_update=args.rescue_js_start_epoch * steps,
                radius_start_update=args.radius_rescue_warmup_epochs * steps,
                prototype_start_update=args.proto_warmup_epochs * steps,
                prototype_ramp_updates=args.proto_ramp_epochs * steps,
                source_refresh_updates=args.source_refresh_interval * steps)
