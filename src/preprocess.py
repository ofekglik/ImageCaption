import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from typing import Dict

from src.utils import load_image


def create_flickr_dataframe(captions_file: str, images_dir: str) -> pd.DataFrame:
    """Create a DataFrame from Flickr30k dataset."""
    df = pd.read_csv(captions_file, header=0, names=['filename', 'caption'], encoding='utf-8')
    df['caption'] = df['caption'].str.strip().str.strip('"')
    df['image_path'] = df['filename'].apply(lambda x: os.path.join(images_dir, x))
    return df


def build_vocabulary(df: pd.DataFrame, min_freq: int = 5) -> dict:
    """Build vocabulary from captions."""
    word_freq = {}
    df['caption'] = df['caption'].astype(str)
    for caption in df['caption']:
        for word in caption.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1

    vocab = {
        '<pad>': 0,
        '<start>': 1,
        '<end>': 2,
        '<unk>': 3,
    }

    idx = len(vocab)
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


class FlickrDataset(Dataset):
    """Custom Dataset for Flickr30k data."""

    def __init__(self,
                 df: pd.DataFrame,
                 vocab: Dict[str, int],
                 transform=None,
                 max_length: int = 50):
        """
        Initialize the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing image paths and captions
            vocab (Dict[str, int]): Vocabulary dictionary mapping words to indices
            transform: Optional image transforms
            max_length (int): Maximum caption length (including <start> and <end> tokens)
        """
        self.df = df
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            tuple: (image_tensor, caption_tensor, caption_length)
        """
        row = self.df.iloc[idx]
        image_path = row['image_path']
        caption = row['caption']

        image = load_image(image_path, self.transform)

        tokens = caption.lower().split()
        tokens = tokens[:(self.max_length - 2)]

        tokens = ['<start>'] + tokens + ['<end>']

        # Convert tokens to indices
        caption_indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        caption_length = len(caption_indices)

        # Pad caption
        caption_indices = caption_indices + [self.vocab['<pad>']] * (self.max_length - len(caption_indices))

        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)
        length_tensor = torch.tensor(caption_length, dtype=torch.long)

        return image, caption_tensor, length_tensor


def create_data_loader(df: pd.DataFrame,
                       vocab: Dict[str, int],
                       transform,
                       batch_size: int = 32,
                       shuffle: bool = True,
                       num_workers: int = 4):
    dataset = FlickrDataset(df, vocab, transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )