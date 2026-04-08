# Sarath AI - Open Media Intelligence

## Overview

**Sarath AI** is an advanced multi-modal AI model developed by **Open Media Intelligence**. The core model, named **OM (Open Media)**, provides powerful capabilities for image editing and text generation with integrated Agent Mode for connecting to external AI APIs.

## Features

### 1. Image Description (Text Generation)
- Generate detailed text descriptions from images
- LSTM-based decoder for sequence generation
- Custom vocabulary management with tokenization
- Supports beam search and configurable max length

### 2. Image Editing
- **Enhancement Mode**: Automatically adjusts brightness, contrast, saturation, and sharpness
- **Filter Mode**: Apply various filters (blur, sharpen, edge enhance, emboss, smooth)
- **Segmentation Mode**: Pixel-wise segmentation (extensible)
- Neural network-based parameter prediction

### 3. Agent Mode - API Integration
Sarath AI Agent seamlessly connects the local OM model with external AI APIs:

#### Supported Providers:
- **OpenAI**: GPT-4 Vision (image analysis), DALL-E (image editing)
- **Anthropic**: Claude Vision for detailed understanding
- **Stability AI**: Advanced image generation/editing
- **Custom Endpoints**: Extensible to any REST API

#### Key Agent Features:
- Smart fallback to local model when API unavailable
- Unified `process()` method for all tasks
- Natural language instruction mapping
- Multi-provider support with easy switching
- Batch processing capabilities

## Installation

```bash
# Required dependencies
pip install torch numpy pillow requests

# Optional (for better performance)
pip install torchvision
```

## Quick Start

### Basic Usage

```python
from sarath_ai import OMModel, OMAgent, Vocabulary
from PIL import Image

# Initialize the OM Model
model = OMModel(vocab_size=10000)

# Create vocabulary
vocab = Vocabulary()
vocab.add_sentence("a beautiful landscape with mountains")
vocab.add_sentence("a city street with people")

# Load an image
image = Image.open("photo.jpg")

# Generate description
description = model.describe_image(image, vocab)
print(f"Description: {description}")

# Edit image
edited = model.edit_image(image, edit_type='enhance')
edited.save("edited_photo.jpg")
```

### Agent Mode with API

```python
from sarath_ai import OMModel, OMAgent
from PIL import Image

# Initialize model and agent
model = OMModel()
agent = OMAgent(
    local_model=model,
    api_provider="openai",  # or 'anthropic', 'stability'
    api_key="your-api-key"
)

# Load image
image = Image.open("photo.jpg")

# Describe using API (with local fallback)
description = agent.process(image=image, task="describe")
print(f"API Description: {description}")

# Edit with natural language
edited = agent.process(
    image=image,
    task="edit",
    text_input="make it brighter and more vibrant"
)
edited.save("enhanced.jpg")

# Comprehensive analysis
analysis = agent.process(image=image, task="analyze")
print(f"Analysis: {analysis}")
```

### Batch Processing

```python
from PIL import Image
from sarath_ai import OMModel, OMAgent

model = OMModel()
agent = OMAgent(model, api_provider="openai", api_key="your-key")

# Load multiple images
images = [Image.open(f"img{i}.jpg") for i in range(5)]

# Process all images
results = agent.batch_process(images, task="describe")

for i, result in enumerate(results):
    print(f"Image {i+1}: {result}")
```

## Model Architecture

### Components

1. **ImageEncoder**
   - Uses pre-trained ResNet-50 when torchvision available
   - Falls back to custom CNN otherwise
   - Outputs 512-dimensional feature vectors

2. **TextDecoder**
   - LSTM-based sequence generator
   - Embedding layer + LSTM + Linear projection
   - Configurable hidden size and layers

3. **ImageEditHead**
   - Multiple heads for different editing tasks
   - Enhancement parameters prediction
   - Filter classification
   - Segmentation mask generation

### Specifications

- **Model Name**: OM (Open Media)
- **Version**: 1.0.0
- **Developer**: Open Media Intelligence
- **Lead AI**: Sarath
- **Embed Size**: 512
- **Hidden Size**: 512
- **Vocabulary Size**: 10,000 (configurable)

## Training

```python
from sarath_ai import OMModel, Vocabulary, ImageCaptionDataset, train_model
from torch.utils.data import DataLoader
from torchvision import transforms

# Prepare data
image_paths = ["img1.jpg", "img2.jpg", ...]
captions = ["a dog running", "a cat sleeping", ...]

vocab = Vocabulary()
dataset = ImageCaptionDataset(image_paths, captions, vocab)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize and train model
model = OMModel(vocab_size=vocab.n_words)
model = train_model(
    model=model,
    dataloader=dataloader,
    vocab=vocab,
    num_epochs=10,
    learning_rate=1e-4,
    device='cuda'  # or 'cpu'
)

# Save trained model
model.save_model("sarath_ai.pth", vocab)
```

## API Reference

### OMModel

| Method | Description |
|--------|-------------|
| `describe_image(image, vocab, max_length)` | Generate text description for an image |
| `edit_image(image, edit_type, params)` | Edit image with specified type and parameters |
| `save_model(path, vocab)` | Save model weights and vocabulary |
| `load_model(path)` | Load model weights |

### OMAgent

| Method | Description |
|--------|-------------|
| `process(image, task, text_input, vocab, use_api)` | Process image with API or local fallback |
| `batch_process(images, task, text_input, vocab)` | Process multiple images |

### Vocabulary

| Method | Description |
|--------|-------------|
| `add_word(word)` | Add a word to vocabulary |
| `add_sentence(sentence)` | Add all words from a sentence |
| `get_index(word)` | Get index for a word |
| `get_word(idx)` | Get word for an index |
| `save(path)` / `load(path)` | Save/load vocabulary |

## Environment Variables

Set API keys via environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export STABILITY_API_KEY="your-stability-key"
```

## Examples

### Image Enhancement

```python
from sarath_ai import OMModel
from PIL import Image

model = OMModel()
image = Image.open("dark_photo.jpg")

# Auto-enhance
enhanced = model.edit_image(image, edit_type='enhance')
enhanced.save("bright_photo.jpg")

# Manual parameters
enhanced = model.edit_image(
    image,
    edit_type='enhance',
    params={
        'brightness': 1.5,
        'contrast': 1.3,
        'saturation': 1.2,
        'sharpness': 1.1
    }
)
```

### Filter Application

```python
filters = ['blur', 'sharpen', 'edge_enhance', 'emboss', 'smooth']

for filter_name in filters:
    filtered = model.edit_image(
        image,
        edit_type='filter',
        params={'filter': filter_name}
    )
    filtered.save(f"photo_{filter_name}.jpg")
```

### Natural Language Editing

```python
agent = OMAgent(model, api_provider="openai", api_key="your-key")

# Natural language instructions
instructions = [
    "make it brighter",
    "increase contrast and saturation",
    "apply a blur effect",
    "enhance the colors"
]

for instruction in instructions:
    edited = agent.process(
        image=image,
        task="edit",
        text_input=instruction
    )
```

## License

Developed by **Open Media Intelligence**

Lead AI: **Sarath**

## Support

For issues, questions, or contributions, please contact Open Media Intelligence.

---

*Sarath AI - Empowering intelligent media processing*
