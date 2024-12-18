from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
from src.custom_callbacks import ImageCaptionCallback


def setup_training(max_epochs=30,
                   dirpath_log='lightning_logs_v1',
                   dirpath_checkpoints='checkpoints_v1'):
    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=dirpath_log,
        name='image_captioning',
        default_hp_metric=False
    )

    # Model checkpoint callback to save the best model during training
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath_checkpoints,
        filename='caption-model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    # Early stopping callback to stop training early if the model is not improving
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )

    # Custom callback to generate captions for validation images
    caption_callback = ImageCaptionCallback(num_samples=4)

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='mps',
        devices=1,
        precision='32-true',
        callbacks=[
            checkpoint_callback,
            early_stopping,
            caption_callback
        ],
        logger=logger,
        log_every_n_steps=10,
    )

    return trainer, checkpoint_callback


def resume_training(
        checkpoint_path,
        model,
        train_loader,
        val_loader,
        max_epochs=30,
        dir_path_logs='lightning_logs_resume_training',
        dir_path_checkpoints='checkpoints_resume_training',
        number=1):

    loaded_model = model.load_from_checkpoint(
        checkpoint_path,
        vocab=model.vocab,
        strict=True
    )

    trainer, checkpoint_callback = setup_training(
        max_epochs=max_epochs,
        dirpath_log=dir_path_logs + f'_{number}',
        dirpath_checkpoints=dir_path_checkpoints + f'_{number}'
    )

    # Manually reset the epoch counter in trainer's state
    trainer.fit_loop.epoch_progress.current.completed = 0

    trainer.fit(
        loaded_model,
        train_loader,
        val_loader
    )

    return trainer, loaded_model, checkpoint_callback
