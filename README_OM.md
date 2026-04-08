# OM AI Model

**OM (Omni Modal)** - An AI model for image editing and text generation.

## Features

- **Image Description**: Generate text descriptions from images
- **Image Editing**: Apply various edits to images (enhance, filter, segmentation)
- **Flexible Architecture**: Works with or without torchvision
  - Uses pre-trained ResNet when torchvision is available
  - Falls back to simple CNN backbone otherwise

## Installation

### Basic Requirements
```bash
pip install torch pillow numpy
```

### For Better Performance (Optional)
```bash
pip install torchvision
```

## Usage

### Basic Example

```python
from PIL import Image
from om_model import OMModel, Vocabulary

# Initialize the model
model = OMModel(vocab_size=10000)
print(f"Model Name: {model.name}")
print(f"Version: {model.version}")

# Create a vocabulary
vocab = Vocabulary(max_size=1000)
sample_texts = [
    "a dog playing in the park",
    "a cat sitting on a couch",
    "beautiful sunset over the ocean"
]
vocab.build_vocab(sample_texts)

# Load an image
image = Image.open("your_image.jpg")

# Generate description
description = model.describe_image(image, idx_to_word=vocab.idx2word)
print(f"Description: {description}")

# Edit the image
edited_image = model.edit_image(image, edit_type='enhance')
edited_image.save("edited_image.jpg")

# Save the model
model.save_model("om_model.pth")

# Load the model later
model.load_model("om_model.pth")
```

### Advanced Usage

#### Image Editing Options

```python
# Enhancement editing
edited = model.edit_image(image, edit_type='enhance')

# Filter application
edited = model.edit_image(image, edit_type='filter')

# Manual parameters
params = {
    'brightness': 0.5,
    'contrast': 0.3,
    'saturation': 0.2
}
edited = model.edit_image(image, edit_type='enhance', params=params)
```

#### Custom Training

```python
import torch
from torch.utils.data import DataLoader

# Prepare your dataset
# (Implement custom Dataset class for your data)

# Initialize model
model = OMModel(vocab_size=10000)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for images, captions in dataloader:
        optimizer.zero_grad()
        
        outputs = model(images, captions)
        loss = criterion(outputs['text_logits'], captions)
        
        loss.backward()
        optimizer.step()
```

## Model Architecture

The OM model consists of:

1. **Image Encoder**: CNN-based encoder (ResNet or simple CNN)
2. **Text Decoder**: LSTM-based sequence generator
3. **Image Editing Heads**: 
   - Filter prediction head
   - Enhancement parameter head
   - Segmentation head (optional)

## API Reference

### OMModel

Main model class with the following methods:

- `__init__(vocab_size, embed_size, hidden_size, encoder_model, max_seq_length)`: Initialize model
- `forward(images, captions)`: Forward pass
- `describe_image(image, tokenizer, idx_to_word)`: Generate image description
- `edit_image(image, edit_type, params)`: Edit image
- `save_model(path)`: Save model weights
- `load_model(path)`: Load model weights

### Vocabulary

Vocabulary management class:

- `build_vocab(texts)`: Build vocabulary from texts
- `encode(text)`: Convert text to token indices
- `decode(indices)`: Convert indices to text

## File Structure

```
/workspace/
├── om_model.py          # Main model implementation
├── om_model.pth         # Saved model weights (after training)
└── README.md            # This file
```

## Notes

- The model uses a simple CNN backbone by default if torchvision is not installed
- For production use, install torchvision to use pre-trained ResNet models
- Training on large datasets is recommended for better performance
- The model supports both CPU and GPU inference

## License

MIT License

## Author

OM AI Team
