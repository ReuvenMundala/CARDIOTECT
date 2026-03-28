"""
Cardiotect - CLI Training Interface (V2)

Simple command-line launcher for training.
"""

import argparse
import os
import sys

from .train_engine import Trainer  # type: ignore
from .dataset import CardiotectDataset  # type: ignore
from .config import DEFAULT_BATCH_SIZE  # type: ignore
from .utils import setup_logging  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Cardiotect Training CLI")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help="Enable AMP (default: True)")
    parser.add_argument('--no_amp', action='store_true', help="Disable AMP")
    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--resume', default=None, help="Path to resume checkpoint")
    parser.add_argument('--lr', type=float, default=None, help="Override learning rate")
    args = parser.parse_args()
    
    if args.no_amp:
        args.use_amp = False
    
    setup_logging(os.path.join(args.output_dir, 'logs'))
    
    # Dataset
    print("Loading datasets...")
    train_ds = CardiotectDataset(subset='train')
    val_ds = CardiotectDataset(subset='val')
    
    # Config
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'use_amp': args.use_amp,
        'output_dir': args.output_dir,
    }
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Resume
    resume_path = args.resume
    if resume_path is None:
        default_resume = os.path.join(args.output_dir, 'checkpoints', 'resume.ckpt')
        if os.path.exists(default_resume):
            resume_path = default_resume
    
    # Train
    def log_fn(msg):
        print(msg)
    
    trainer = Trainer(
        config, train_ds, val_ds,
        log_callback=log_fn,
        resume_path=resume_path
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nCTRL+C — saving emergency checkpoint...")
        trainer._save_checkpoint(trainer.current_epoch, is_resume=True)
        sys.exit(0)


if __name__ == '__main__':
    main()
