import logging
import os
import json

def log_training_and_model_info(logger, args, lm_config, model, world_size):
    """Log a structured summary of training args and model configuration."""
    # Model config - try to get a dict representation
    try:
        model_cfg = lm_config.to_dict() if hasattr(lm_config, 'to_dict') else lm_config.__dict__
    except Exception:
        model_cfg = str(lm_config)

    param_count_m = sum(p.numel() for p in model.parameters() if not p.stop_gradient) / 1e6

    info = {
        'model': {
            'config': model_cfg,
            'trainable_params_million': param_count_m.tolist(),
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'accumulation_steps': args.accumulation_steps,
            'dtype': args.dtype,
            'device': args.device,
            'max_seq_len': args.max_seq_len,
            'use_moe': args.use_moe,
        },
        'distributed': {
            'ddp': bool(args.ddp),
            'world_size': int(world_size),
        }
    }
    logger.info(f"Model and training configuration:\n{json.dumps(info, indent=4)}")

