# OM (Open Media) AI Model

**Developed by: Open Media Intelligence**  
**Version: 2.0.0 (Agent Mode Enabled)**

## Overview

OM (Open Media) is a versatile AI model designed for image editing and text generation tasks. It features a dual-mode architecture that combines:

1. **Local Processing**: Lightweight on-device image editing and description generation
2. **Agent Mode**: Intelligent orchestration with external AI APIs for advanced capabilities

## Features

### Core Capabilities
- **Image Description**: Generate text descriptions from input images
- **Image Editing**: Apply enhancements, filters, and transformations
- **Agent Mode**: Connect with external AI APIs (OpenAI, Anthropic, Stability AI)

### Supported API Providers
- **OpenAI**: GPT-4 Vision for image analysis, DALL-E for image editing
- **Anthropic**: Claude Vision for detailed image understanding
- **Stability AI**: Advanced image generation and editing
- **Custom Endpoints**: Extendable to any REST API

## Installation

### Requirements
- Python 3.8+
- PyTorch
- PIL/Pillow
- NumPy
- requests (for Agent Mode)

### Optional (Recommended)
- torchvision (for pre-trained ResNet backbone)

```bash
pip install torch pillow numpy requests
pip install torchvision  # Optional, for better performance
```

## Usage

### Basic Usage - Local Model

```python
from om_model import OMModel, Vocabulary
from PIL import Image

# Initialize the model
model = OMModel(vocab_size=10000)

# Load an image
image = Image.open("your_image.jpg")

# Generate image description
description = model.describe_image(image, idx_to_word=vocab.idx2word)
print(f"Description: {description}")

# Edit the image
edited_image = model.edit_image(image, edit_type='enhance')
edited_image.save("edited_image.jpg")

# Save/Load model weights
model.save_model("om_model.pth")
model.load_model("om_model.pth")
```

### Agent Mode - With API Integration

```python
from om_model import OMModel, OMAgent
from PIL import Image

# Initialize local model
local_model = OMModel(vocab_size=10000)

# Initialize Agent with API key
agent = OMAgent(
    local_model=local_model,
    api_provider="openai",  # or "anthropic", "stability"
    api_key="your-api-key"  # Or set OPENAI_API_KEY environment variable
)

# Load an image
image = Image.open("your_image.jpg")

# Describe image using API (falls back to local if no API key)
description = agent.process(image=image, task="describe")
print(f"Description: {description}")

# Edit image with text instruction
edited_image = agent.process(
    image=image, 
    text_input="make it brighter and more vibrant",
    task="edit"
)
edited_image.save("edited_image.jpg")

# Comprehensive analysis (combines local + API)
analysis = agent.process(image=image, task="analyze")
print(f"Local Analysis: {analysis['local_analysis']}")
print(f"API Analysis: {analysis['api_analysis']}")
```

### Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export STABILITY_API_KEY="your-stability-key"
```

## Architecture

### OMModel Components

1. **ImageEncoder**: Encodes images into feature vectors
   - Uses ResNet-50 (with torchvision) or SimpleCNN (fallback)
   
2. **TextDecoder**: LSTM-based sequence generator for captions
   
3. **ImageEditHead**: Predicts editing parameters
   - Enhancement head (brightness, contrast, saturation, etc.)
   - Filter head (convolution kernels)

### OMAgent Features

- **Smart Fallback**: Automatically uses local model when API is unavailable
- **Unified Interface**: Single `process()` method for all tasks
- **Multi-Provider Support**: Switch between API providers seamlessly
- **Instruction Mapping**: Converts natural language to local operations

## API Reference

### OMModel

```python
class OMModel:
    def __init__(self, vocab_size=10000, embed_size=512, hidden_size=512):
        """Initialize OM model."""
        
    def describe_image(self, image, tokenizer=None, idx_to_word=None) -> str:
        """Generate text description from image."""
        
    def edit_image(self, image, edit_type='enhance', params=None) -> Image.Image:
        """Edit image based on type and parameters."""
        
    def save_model(self, path: str):
        """Save model weights."""
        
    def load_model(self, path: str):
        """Load model weights."""
```

### OMAgent

```python
class OMAgent:
    def __init__(self, local_model=None, api_provider="openai", 
                 api_key=None, use_local_first=True):
        """Initialize OM Agent with optional API integration."""
        
    def process(self, image=None, text_input=None, task="describe"):
        """
        Unified processing method.
        
        Args:
            image: Input PIL Image
            text_input: Text prompt/instruction
            task: One of "describe", "edit", "analyze"
            
        Returns:
            str, Image.Image, or Dict based on task
        """
        
    def describe_image_api(self, image, prompt) -> str:
        """Call external API for image description."""
        
    def edit_image_api(self, image, instruction) -> Image.Image:
        """Call external API for image editing."""
```

## Examples

### Example 1: Basic Image Description

```python
from om_model import OMModel, Vocabulary
from PIL import Image

model = OMModel()
image = Image.open("photo.jpg")

# Create vocabulary
vocab = Vocabulary()
vocab.build_vocab(["sample text for vocabulary"])

description = model.describe_image(image, idx_to_word=vocab.idx2word)
print(description)
```

### Example 2: Image Enhancement

```python
from om_model import OMModel
from PIL import Image

model = OMModel()
image = Image.open("dark_photo.jpg")

# Enhance with custom parameters
edited = model.edit_image(
    image, 
    edit_type='enhance',
    params={
        'brightness': 0.3,
        'contrast': 0.2,
        'saturation': 0.1
    }
)
edited.save("enhanced_photo.jpg")
```

### Example 3: Agent with Multiple Providers

```python
from om_model import OMModel, OMAgent

model = OMModel()

# OpenAI Agent
openai_agent = OMAgent(local_model=model, api_provider="openai")

# Anthropic Agent  
anthropic_agent = OMAgent(local_model=model, api_provider="anthropic")

# Compare results
image = Image.open("test.jpg")
result1 = openai_agent.process(image=image, task="describe")
result2 = anthropic_agent.process(image=image, task="describe")
```

## Model Specifications

- **Name**: OM (Open Media)
- **Developer**: Open Media Intelligence
- **Total Parameters**: ~17 million
- **Input Size**: 224x224 RGB images
- **Output**: Text tokens (descriptions) or edited images
- **License**: MIT

## Performance Notes

- **Without torchvision**: Uses SimpleCNN backbone (faster initialization, lower accuracy)
- **With torchvision**: Uses ResNet-50 backbone (better accuracy, requires download)
- **Agent Mode**: Latency depends on API response time
- **Local Mode**: Real-time processing on CPU, faster on GPU

## Troubleshooting

### "torchvision not available" Warning
Install torchvision for better performance:
```bash
pip install torchvision
```

### "No API Key found" Warning
Set your API key as an environment variable or pass it directly:
```python
agent = OMAgent(api_key="your-key")
# or
export OPENAI_API_KEY="your-key"
```

### Memory Issues
Reduce model size:
```python
model = OMModel(vocab_size=5000, embed_size=256, hidden_size=256)
```

## Contributing

Contributions are welcome! Please submit issues and pull requests to improve OM.

## License

MIT License - See LICENSE file for details.

---

**OM (Open Media)** - *Intelligent Image Processing by Open Media Intelligence*
