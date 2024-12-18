from typing import Tuple
import torch
from torchvision import transforms
import random
from PIL import Image
from IPython.display import display, HTML
from io import BytesIO
import base64


def get_transforms(image_size: Tuple[int, int] = (224, 224)) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get image transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def load_image(image_path: str, transform=None) -> torch.Tensor:
    """Load and transform image."""
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    return image


def show_image_with_captions(df):
    """
    Display a random image from the dataset along with all its captions.
    """
    # Get a random image filename
    image_filename = random.choice(df['filename'].unique())

    # Get captions and image path
    captions = df[df['filename'] == image_filename]['caption'].tolist()
    image_path = df[df['filename'] == image_filename]['image_path'].iloc[0]

    # Display image with captions
    return display_image_with_captions(image_path, captions)


def display_image_with_captions(image_path, captions) -> str:
    """
    Parameters:
    image_path (str): Path to the image file
    captions (list): List of strings containing captions
    """
    # Read image
    img = Image.open(image_path)

    # Convert PIL image to base64 string
    buffered = BytesIO()
    img.save(buffered, format=img.format or 'PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Create HTML for side-by-side display
    captions_html = "<br><br>".join(f"{i + 1}. {caption}" for i, caption in enumerate(captions))

    html = f"""
    <div style="display: flex; align-items: start; gap: 20px; max-width: 1200px;">
        <div style="flex: 1;">
            <img src="data:image/{img.format or 'png'};base64,{img_str}" 
                 style="max-width: 100%; height: auto;">
        </div>
        <div style="flex: 1; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3 style="margin-top: 0; color: black;">Captions:</h3>
            <div style="font-size: 14px; line-height: 1.6; color: black;">
                {captions_html}
            </div>
        </div>
    </div>
    """

    display(HTML(html))
    return image_path
