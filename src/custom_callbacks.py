import lightning as L
import torch
import matplotlib.pyplot as plt


class ImageCaptionCallback(L.Callback):
    """
    Callback to generate captions for a few images in the validation set and log them to TensorBoard.
    """
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        val_batch = next(iter(trainer.val_dataloaders))
        images, true_captions, _ = val_batch

        # Move images to the same device as model
        images = images.to(pl_module.device)

        # Randomly select num_samples
        # indices = torch.randperm(len(images))[:self.num_samples]
        # images = images[indices]

        # fixed images
        images = images[:self.num_samples]

        generated_captions = []
        for image in images:
            caption = pl_module.generate_caption(image)
            generated_captions.append(caption)

        images = images.cpu()

        fig = plt.figure(figsize=(12, 3 * self.num_samples))
        for idx, (image, caption) in enumerate(zip(images, generated_captions)):
            img = torch.clamp(image * 0.229 + 0.485, 0, 1)

            plt.subplot(self.num_samples, 1, idx + 1)
            plt.imshow(img.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f'Generated: {caption}', pad=20)

        trainer.logger.experiment.add_figure(
            'Generated Captions',
            fig,
            global_step=trainer.global_step
        )
        plt.close(fig)
