import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from evaluate import evaluate
from unet import EncoderDoubleCoordAttUNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
val_img_dir = Path('./data/val_imgs/')
val_mask_dir = Path('./data/val_masks/')
dir_checkpoint = Path('./checkpoints4/')#为了多gpu任务


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        load_checkpoint: str = None,  # 新增：加载检查点路径
):
    # 1. Create dataset
    train_set = BasicDataset(dir_img, dir_mask)
    val_set = BasicDataset(val_img_dir, val_mask_dir)
    n_val = len(val_set)
    n_train = len(train_set)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 3. Initialize logging
    experiment = wandb.init(
        project='U-Net',
        resume='allow',
        anonymous='must'
    )
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    # 4. Initialize training components
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    # 5. Load checkpoint if provided
    start_epoch = 1
    best_dice = 0.0
    best_epoch = 0
    global_step = 0

    if load_checkpoint and os.path.isfile(load_checkpoint):
        checkpoint = torch.load(load_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        best_epoch = checkpoint['best_epoch']
        global_step = checkpoint.get('global_step', 0)

        if 'mask_values' in checkpoint:
            train_set.mask_values = checkpoint['mask_values']

        logging.info(f'Resuming training from epoch {start_epoch}')
        logging.info(f'Previous best Dice: {best_dice:.4f} at epoch {best_epoch}')

    logging.info(f'''Starting training:
        Epochs:          {epochs} (resuming from {start_epoch})
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 6. Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Validation
            val_score = evaluate(model, val_loader, device, amp)
            scheduler.step(val_score)
            logging.info(f'[Epoch {epoch}] Validation Dice: {val_score:.4f}')

            # Save checkpoint
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'grad_scaler_state_dict': grad_scaler.state_dict(),
                    'best_dice': best_dice,
                    'best_epoch': best_epoch,
                    'global_step': global_step,
                    'mask_values': train_set.mask_values,
                }

                # Save regular checkpoint
                torch.save(checkpoint, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
                logging.info(f'Checkpoint {epoch} saved!')

                # Save best checkpoint
                if val_score > best_dice:
                    best_dice = val_score
                    best_epoch = epoch
                    torch.save(checkpoint, str(dir_checkpoint / 'best_checkpoint.pth'))
                    logging.info(f'New best model saved (Dice: {best_dice:.4f})')

            # Log to wandb
            try:
                experiment.log({
                    'epoch': epoch,
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'best validation Dice': best_dice,
                    'train_loss': epoch_loss / len(train_loader)
                })
            except:
                pass


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default="checkpoints4/checkpoint_epoch90.pth",
                        help='Load model and training state from a checkpoint file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Initialize model
    model = EncoderDoubleCoordAttUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # Load checkpoint if specified
    if args.load:
        if not os.path.isfile(args.load):
            raise FileNotFoundError(f'Checkpoint file not found: {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            load_checkpoint=args.load  # 传递检查点路径
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            load_checkpoint=args.load
        )