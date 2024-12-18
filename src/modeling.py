import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
import random


def get_device():
    """Get the appropriate device (MPS if available, else CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_image_encoder(embed_size=512, fine_tune_encoder=False):
    """
    Create the image encoder using ResNet50.
    Args:
        embed_size (int): Size of the embedding
        fine_tune_encoder (bool): If True, allows encoder parameters to be updated.
                                  If False, freezes them.
    """
    # Load pre-trained ResNet
    resnet = models.resnet50(pretrained=True)
    # Remove the last classification layer
    modules = list(resnet.children())[:-1]
    encoder = nn.Sequential(*modules)

    if not fine_tune_encoder:
        # Freeze ResNet parameters if not fine-tuning
        for param in encoder.parameters():
            param.requires_grad = False

    projection = nn.Linear(2048, embed_size)
    return encoder, projection


class CaptionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.caption_embedding = None  # Will be set by the main module

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, vocab_size)

    def set_embedding_layer(self, embedding_layer):
        """Set the embedding layer (called by main module)"""
        self.caption_embedding = embedding_layer

    def forward(self, features, captions, caption_embeddings, teacher_forcing_ratio=1.0):
        """
        Forward pass with proper teacher forcing logic.
        """
        assert self.caption_embedding is not None, "Embedding layer not set!"

        batch_size = features.size(0)
        max_length = captions.size(1) - 1  # -1 because we don't predict after <end>

        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, max_length, self.linear.out_features).to(features.device)

        input_seq = features.unsqueeze(1)
        hidden = None

        for t in range(max_length):
            # Forward through LSTM
            lstm_out, hidden = self.lstm(input_seq, hidden)
            output = self.linear(lstm_out)
            outputs[:, t, :] = output.squeeze(1)

            if t < max_length - 1:
                use_teacher_forcing = (random.random() < teacher_forcing_ratio)
                if use_teacher_forcing:
                    # Use ground truth embedding as next input
                    input_seq = caption_embeddings[:, t + 1].unsqueeze(1)
                else:
                    # Use predicted token embedding as next input
                    top1_pred = output.argmax(2)
                    input_seq = self.caption_embedding(top1_pred.squeeze(1)).unsqueeze(1)

        return outputs


class ImageCaptioningModule(L.LightningModule):
    def __init__(self,
                 vocab,
                 embed_size=512,
                 hidden_size=512,
                 num_layers=1,
                 learning_rate=1e-4,
                 teacher_forcing_ratio=1.0,
                 finetune_encoder_after=-1):  # -1 means never finetune
        super().__init__()
        self.save_hyperparameters(ignore=['vocab'])
        self.vocab = vocab
        self.learning_rate = learning_rate
        self._teacher_forcing_ratio = teacher_forcing_ratio
        self.finetune_encoder_after = finetune_encoder_after

        # Image encoder - initially frozen
        self.image_encoder, self.image_projection = get_image_encoder(
            embed_size,
            fine_tune_encoder=False
        )

        # Caption embedding
        self.caption_embedding = nn.Embedding(
            len(vocab),
            embed_size,
            padding_idx=vocab['<pad>']
        )

        # Create decoder and set its embedding layer
        self.decoder = CaptionDecoder(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=len(vocab),
            num_layers=num_layers
        )
        self.decoder.set_embedding_layer(self.caption_embedding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    def on_train_epoch_start(self):
        """Check if we should start fine-tuning the encoder"""
        if self.finetune_encoder_after != -1 and self.current_epoch >= self.finetune_encoder_after:
            # Unfreeze encoder
            for param in self.image_encoder.parameters():
                param.requires_grad = True
            # Log that we're starting fine-tuning
            print(f"\nEpoch {self.current_epoch}: Starting encoder fine-tuning")

    @property
    def teacher_forcing_ratio(self):
        return self._teacher_forcing_ratio

    def forward(self, images, captions):
        # Get image features
        with torch.no_grad():
            features = self.image_encoder(images)
        features = features.view(features.size(0), -1)
        features = self.image_projection(features)

        # Get caption embeddings
        caption_embeddings = self.caption_embedding(captions)

        # Generate outputs with teacher forcing
        outputs = self.decoder(
            features,
            captions,
            caption_embeddings,
            self._teacher_forcing_ratio
        )

        return outputs

    def training_step(self, batch, batch_idx):
        """Training step with detailed logging"""
        images, captions, _ = batch
        outputs = self(images, captions)

        targets = captions[:, 1:]  # Remove <start>
        loss = self.criterion(
            outputs.reshape(-1, len(self.vocab)),
            targets.reshape(-1)
        )

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log learning rate
        opt = self.optimizers()
        if opt is not None:
            self.log('learning_rate', opt.param_groups[0]['lr'], on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with detailed logging"""
        images, captions, _ = batch
        outputs = self(images, captions)

        targets = captions[:, 1:]
        loss = self.criterion(
            outputs.reshape(-1, len(self.vocab)),
            targets.reshape(-1)
        )

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def generate_caption(self, image, max_length=50):
        self.eval()
        with torch.no_grad():
            # Get image features
            features = self.image_encoder(image.unsqueeze(0))
            features = features.view(features.size(0), -1)
            features = self.image_projection(features)

            # Initialize generation
            caption = [self.vocab['<start>']]
            hidden = None

            for _ in range(max_length):
                current_word = torch.LongTensor([caption[-1]]).to(self.device)
                current_embed = self.caption_embedding(current_word).unsqueeze(1)

                if len(caption) == 1:
                    lstm_input = features.unsqueeze(1)
                else:
                    lstm_input = current_embed

                output, hidden = self.decoder.lstm(lstm_input, hidden)
                scores = self.decoder.linear(output.squeeze(1))
                predicted = scores.argmax(1)

                predicted_word = predicted.item()
                caption.append(predicted_word)

                if predicted_word == self.vocab['<end>']:
                    break

            # Convert indices to words
            idx_to_word = {idx: word for word, idx in self.vocab.items()}
            caption_words = [idx_to_word[idx] for idx in caption[1:-1]]

        return ' '.join(caption_words)


def save_model_state(model, trainer, checkpoint_callback, save_path='model_checkpoint.pt'):
    """
    Save the complete model state including training information.
    """
    # Get optimizer (trainer.optimizers returns a list)
    optimizers = trainer.optimizers
    optimizer = optimizers[0] if optimizers else None

    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': trainer.current_epoch,
        'vocab': model.vocab,
        'best_model_path': checkpoint_callback.best_model_path if checkpoint_callback else None,
        'model_config': {
            'embed_size': model.hparams.embed_size,
            'hidden_size': model.hparams.hidden_size,
            'learning_rate': model.hparams.learning_rate,
            'teacher_forcing_ratio': model.teacher_forcing_ratio,
            'finetune_encoder_after': model.finetune_encoder_after
        }
    }

    torch.save(state, save_path)
    print(f"Model state saved to {save_path}")


def load_model_state(save_path, model=None):
    state = torch.load(save_path)

    if model is None:
        # Create new model with saved config
        model = ImageCaptioningModule(
            vocab=state['vocab'],
            embed_size=state['model_config']['embed_size'],
            hidden_size=state['model_config']['hidden_size'],
            learning_rate=state['model_config']['learning_rate'],
            teacher_forcing_ratio=state['model_config']['teacher_forcing_ratio'],
            finetune_encoder_after=state['model_config']['finetune_encoder_after']
        )

    model.load_state_dict(state['model_state_dict'])
    print(f"Model state loaded from {save_path}")

    return model, state
