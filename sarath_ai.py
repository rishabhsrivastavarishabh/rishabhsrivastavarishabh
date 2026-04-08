"""
Sarath AI - Open Media Intelligence
====================================
An advanced AI model for image editing and text generation.

Model Name: OM (Open Media)
Developer: Open Media Intelligence
Lead AI: Sarath

Features:
- Image Description (Text Generation)
- Image Editing (Enhancement, Filters, Segmentation)
- Agent Mode with External API Integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import json
import requests
from typing import Optional, Dict, List, Tuple, Any
import base64
from io import BytesIO


# ============================================================================
# VOCABULARY MANAGEMENT
# ============================================================================

class Vocabulary:
    """Manages word-to-index mappings for text generation."""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.word_count = {}
        self.n_words = 4
    
    def add_word(self, word: str):
        """Add a word to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
    
    def add_sentence(self, sentence: str):
        """Add all words from a sentence."""
        for word in sentence.lower().split():
            self.add_word(word)
    
    def get_index(self, word: str) -> int:
        """Get index for a word, return UNK index if not found."""
        return self.word2idx.get(word.lower(), self.word2idx['<UNK>'])
    
    def get_word(self, idx: int) -> str:
        """Get word for an index."""
        return self.idx2word.get(idx, '<UNK>')
    
    def save(self, path: str):
        """Save vocabulary to file."""
        data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},
            'n_words': self.n_words
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        self.n_words = data['n_words']


# ============================================================================
# IMAGE ENCODER
# ============================================================================

class ImageEncoder(nn.Module):
    """Encodes images into feature vectors."""
    
    def __init__(self, embed_size: int = 512, use_pretrained: bool = True):
        super(ImageEncoder, self).__init__()
        self.use_pretrained = use_pretrained
        
        # Try to use ResNet if torchvision is available
        try:
            import torchvision.models as models
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
            modules = list(resnet.children())[:-1]  # Remove last layer
            self.resnet = nn.Sequential(*modules)
            self.embed = nn.Linear(2048, embed_size)
        except ImportError:
            # Fallback to simple CNN if torchvision not available
            print("Torchvision not available, using simple CNN encoder")
            self.resnet = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.embed = nn.Linear(256, embed_size)
        
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features


# ============================================================================
# TEXT DECODER
# ============================================================================

class TextDecoder(nn.Module):
    """Generates text descriptions from image features."""
    
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super(TextDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        embeddings = self.dropout(self.embedding(captions))
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs
    
    def generate(self, features: torch.Tensor, vocab: Vocabulary, max_length: int = 30) -> str:
        """Generate caption from image features."""
        self.eval()
        with torch.no_grad():
            batch_size = features.size(0)
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device),
                     torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device))
            
            input_token = torch.full((batch_size, 1), vocab.get_index('<START>'), dtype=torch.long).to(features.device)
            generated_words = []
            
            for _ in range(max_length):
                embedding = self.dropout(self.embedding(input_token))  # (batch, 1, embed)
                output, hidden = self.lstm(embedding, hidden)  # (batch, 1, hidden)
                output = self.linear(output.squeeze(1))  # (batch, vocab)
                
                _, predicted = output.max(dim=1)
                word = vocab.get_word(predicted[0].item())
                
                if word == '<END>':
                    break
                
                if word not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                    generated_words.append(word)
                
                input_token = predicted.unsqueeze(1)  # (batch, 1)
            
            return ' '.join(generated_words)


# ============================================================================
# IMAGE EDITING HEAD
# ============================================================================

class ImageEditHead(nn.Module):
    """Neural network head for image editing parameters prediction."""
    
    def __init__(self, input_size: int = 512):
        super(ImageEditHead, self).__init__()
        
        # Enhancement parameters: brightness, contrast, saturation, sharpness
        self.enhancement_fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 enhancement parameters
        )
        
        # Filter type classification
        self.filter_fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 filter types
        )
        
        # Segmentation mask (simplified)
        self.segmentation_fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Binary segmentation
        )
    
    def forward(self, features: torch.Tensor, task: str = 'enhance') -> torch.Tensor:
        """Forward pass based on task type."""
        if task == 'enhance':
            return self.enhancement_fc(features)
        elif task == 'filter':
            return self.filter_fc(features)
        elif task == 'segment':
            return self.segmentation_fc(features)
        else:
            raise ValueError(f"Unknown task: {task}")


# ============================================================================
# OM MODEL (Open Media Model)
# ============================================================================

class OMModel(nn.Module):
    """
    OM (Open Media) Model by Open Media Intelligence
    Lead AI: Sarath
    
    Capabilities:
    - Image description generation
    - Image editing (enhancement, filters, segmentation)
    - Multi-modal processing
    """
    
    def __init__(self, vocab_size: int = 10000, embed_size: int = 512, 
                 hidden_size: int = 512, num_layers: int = 1):
        super(OMModel, self).__init__()
        
        self.encoder = ImageEncoder(embed_size=embed_size)
        self.decoder = TextDecoder(embed_size, hidden_size, vocab_size, num_layers)
        self.edit_head = ImageEditHead(input_size=embed_size)
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Forward pass for training (image captioning)."""
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def describe_image(self, image: Image.Image, vocab: Vocabulary, 
                      max_length: int = 30) -> str:
        """Generate text description for an image."""
        self.eval()
        
        # Preprocess image
        transform = self._get_transform()
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.encoder(image_tensor)
            caption = self.decoder.generate(features, vocab, max_length)
        
        return caption
    
    def edit_image(self, image: Image.Image, edit_type: str = 'enhance',
                   params: Optional[Dict[str, float]] = None) -> Image.Image:
        """
        Edit an image based on specified type and parameters.
        
        Args:
            image: Input PIL Image
            edit_type: Type of edit ('enhance', 'filter', 'segment')
            params: Dictionary of parameters (optional, auto-predicted if None)
        
        Returns:
            Edited PIL Image
        """
        self.eval()
        
        # Preprocess image
        transform = self._get_transform()
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.encoder(image_tensor)
            
            if params is None:
                # Predict parameters using neural network
                if edit_type == 'enhance':
                    pred_params = self.edit_head(features, task='enhance').squeeze()
                    params = {
                        'brightness': 1.0 + (pred_params[0].item() * 0.5),
                        'contrast': 1.0 + (pred_params[1].item() * 0.5),
                        'saturation': 1.0 + (pred_params[2].item() * 0.5),
                        'sharpness': 1.0 + (pred_params[3].item() * 0.5)
                    }
                elif edit_type == 'filter':
                    pred_filter = self.edit_head(features, task='filter').squeeze()
                    filter_idx = pred_filter.argmax().item()
                    filters = ['none', 'blur', 'sharpen', 'edge_enhance', 'emboss', 'smooth']
                    params = {'filter': filters[filter_idx]}
        
        # Apply edits
        edited_image = self._apply_edits(image, edit_type, params)
        return edited_image
    
    def _apply_edits(self, image: Image.Image, edit_type: str, 
                     params: Dict[str, Any]) -> Image.Image:
        """Apply actual edits to the image."""
        edited = image.copy()
        
        if edit_type == 'enhance':
            if 'brightness' in params:
                enhancer = ImageEnhance.Brightness(edited)
                edited = enhancer.enhance(max(0.1, min(2.0, params['brightness'])))
            
            if 'contrast' in params:
                enhancer = ImageEnhance.Contrast(edited)
                edited = enhancer.enhance(max(0.1, min(2.0, params['contrast'])))
            
            if 'saturation' in params:
                enhancer = ImageEnhance.Color(edited)
                edited = enhancer.enhance(max(0.1, min(2.0, params['saturation'])))
            
            if 'sharpness' in params:
                enhancer = ImageEnhance.Sharpness(edited)
                edited = enhancer.enhance(max(0.1, min(2.0, params['sharpness'])))
        
        elif edit_type == 'filter':
            filter_name = params.get('filter', 'none')
            if filter_name == 'blur':
                edited = edited.filter(ImageFilter.BLUR)
            elif filter_name == 'sharpen':
                edited = edited.filter(ImageFilter.SHARPEN)
            elif filter_name == 'edge_enhance':
                edited = edited.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_name == 'emboss':
                edited = edited.filter(ImageFilter.EMBOSS)
            elif filter_name == 'smooth':
                edited = edited.filter(ImageFilter.SMOOTH)
        
        return edited
    
    def _get_transform(self):
        """Get image transformation pipeline."""
        try:
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        except ImportError:
            # Fallback to simple PIL-based transform
            class SimpleTransform:
                def __call__(self, img):
                    img = img.resize((224, 224))
                    arr = np.array(img).astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    arr = (arr - mean) / std
                    return torch.from_numpy(arr.transpose(2, 0, 1)).float()
            return SimpleTransform()
    
    def save_model(self, path: str, vocab: Optional[Vocabulary] = None):
        """Save model weights and vocabulary."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_size': self.embed_size,
            'hidden_size': self.hidden_size
        }, path)
        
        if vocab:
            vocab_path = path.replace('.pth', '_vocab.json')
            vocab.save(vocab_path)
    
    def load_model(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        self.vocab_size = checkpoint['vocab_size']
        self.embed_size = checkpoint['embed_size']
        self.hidden_size = checkpoint['hidden_size']
        
        # Reinitialize decoder with correct vocab size
        self.decoder = TextDecoder(
            self.embed_size, 
            self.hidden_size, 
            self.vocab_size
        )
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()


# ============================================================================
# AGENT MODE - API INTEGRATION
# ============================================================================

class OMAgent:
    """
    Sarath AI Agent Mode
    Connects local OM model with external AI APIs for enhanced capabilities.
    
    Supported Providers:
    - OpenAI (GPT-4 Vision, DALL-E)
    - Anthropic (Claude Vision)
    - Stability AI (Image Generation/Editing)
    - Custom REST endpoints
    """
    
    def __init__(self, local_model: OMModel, api_provider: str = 'openai',
                 api_key: Optional[str] = None, custom_endpoint: Optional[str] = None):
        """
        Initialize Sarath AI Agent.
        
        Args:
            local_model: Local OM model instance
            api_provider: API provider name ('openai', 'anthropic', 'stability', 'custom')
            api_key: API key for the provider
            custom_endpoint: Custom API endpoint URL
        """
        self.local_model = local_model
        self.api_provider = api_provider
        self.api_key = api_key or os.getenv(f'{api_provider.upper()}_API_KEY')
        self.custom_endpoint = custom_endpoint
        
        # API endpoints
        self.endpoints = {
            'openai': {
                'vision': 'https://api.openai.com/v1/chat/completions',
                'edit': 'https://api.openai.com/v1/images/edits'
            },
            'anthropic': {
                'vision': 'https://api.anthropic.com/v1/messages'
            },
            'stability': {
                'edit': 'https://api.stability.ai/v1/generation/image-to-image'
            }
        }
        
        # Task mapping for natural language understanding
        self.task_mappings = {
            'describe': ['describe', 'explain', 'tell about', 'what is'],
            'edit': ['edit', 'modify', 'change', 'adjust', 'enhance'],
            'analyze': ['analyze', 'inspect', 'examine', 'study'],
            'caption': ['caption', 'title', 'name']
        }
    
    def _detect_task(self, text_input: str) -> str:
        """Detect task type from natural language input."""
        text_lower = text_input.lower()
        
        for task, keywords in self.task_mappings.items():
            if any(keyword in text_lower for keyword in keywords):
                return task
        
        return 'describe'  # Default task
    
    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _call_openai_vision(self, image: Image.Image, prompt: str) -> str:
        """Call OpenAI GPT-4 Vision API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        base64_image = self._encode_image_base64(image)
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        payload = {
            'model': 'gpt-4-vision-preview',
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': prompt
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/png;base64,{base64_image}'
                            }
                        }
                    ]
                }
            ],
            'max_tokens': 500
        }
        
        response = requests.post(self.endpoints['openai']['vision'], 
                               headers=headers, json=payload)
        response.raise_for_status()
        
        return response.json()['choices'][0]['message']['content']
    
    def _call_anthropic_vision(self, image: Image.Image, prompt: str) -> str:
        """Call Anthropic Claude Vision API."""
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        base64_image = self._encode_image_base64(image)
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': 'claude-3-opus-20240229',
            'max_tokens': 500,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': 'image/png',
                                'data': base64_image
                            }
                        },
                        {
                            'type': 'text',
                            'text': prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(self.endpoints['anthropic']['vision'],
                               headers=headers, json=payload)
        response.raise_for_status()
        
        return response.json()['content'][0]['text']
    
    def _call_local_model(self, image: Image.Image, task: str, 
                         vocab: Optional[Vocabulary] = None) -> Any:
        """Fallback to local OM model."""
        if task in ['describe', 'analyze', 'caption']:
            if vocab is None:
                # Create dummy vocabulary for testing
                vocab = Vocabulary()
                for word in ['image', 'photo', 'picture', 'scene', 'object']:
                    vocab.add_word(word)
            
            return self.local_model.describe_image(image, vocab)
        
        elif task == 'edit':
            return self.local_model.edit_image(image, edit_type='enhance')
        
        else:
            return "Task not recognized"
    
    def process(self, image: Image.Image, task: str = 'describe',
                text_input: str = '', vocab: Optional[Vocabulary] = None,
                use_api: bool = True) -> Any:
        """
        Process image using Agent Mode (API + Local fallback).
        
        Args:
            image: Input PIL Image
            task: Task type ('describe', 'edit', 'analyze', 'caption')
            text_input: Natural language instruction
            vocab: Vocabulary for local model
            use_api: Whether to try API first (falls back to local if unavailable)
        
        Returns:
            Result (text description, edited image, or analysis)
        """
        # Detect task from text input if not explicitly provided
        if text_input and task == 'describe':
            task = self._detect_task(text_input)
        
        # Construct prompt based on task
        prompts = {
            'describe': 'Describe this image in detail.',
            'analyze': 'Analyze this image thoroughly, including objects, colors, composition, and context.',
            'caption': 'Provide a concise, descriptive caption for this image.',
            'edit': f'Edit instructions: {text_input}' if text_input else 'Enhance this image.'
        }
        
        prompt = prompts.get(task, 'Describe this image.')
        if text_input and task != 'edit':
            prompt = f"{prompt} Additional context: {text_input}"
        
        # Try API first if requested
        if use_api and self.api_key:
            try:
                if self.api_provider == 'openai':
                    result = self._call_openai_vision(image, prompt)
                    return result
                
                elif self.api_provider == 'anthropic':
                    result = self._call_anthropic_vision(image, prompt)
                    return result
                
                # Add more providers as needed
                
            except Exception as e:
                print(f"API call failed ({self.api_provider}): {e}")
                print("Falling back to local OM model...")
        
        # Fallback to local model
        return self._call_local_model(image, task, vocab)
    
    def batch_process(self, images: List[Image.Image], task: str = 'describe',
                     text_input: str = '', vocab: Optional[Vocabulary] = None) -> List[Any]:
        """Process multiple images."""
        results = []
        for i, image in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}...")
            result = self.process(image, task, text_input, vocab)
            results.append(result)
        return results


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class ImageCaptionDataset(Dataset):
    """Dataset for image captioning training."""
    
    def __init__(self, image_paths: List[str], captions: List[str], 
                 vocab: Vocabulary, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
        
        # Build vocabulary from captions
        for caption in captions:
            vocab.add_sentence(caption)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Convert caption to indices
        words = caption.lower().split()
        indices = [self.vocab.get_index('<START>')]
        indices.extend([self.vocab.get_index(w) for w in words])
        indices.append(self.vocab.get_index('<END>'))
        
        target = torch.tensor(indices[1:], dtype=torch.long)
        input_caption = torch.tensor(indices[:-1], dtype=torch.long)
        
        return image, input_caption, target


def train_model(model: OMModel, dataloader: DataLoader, vocab: Vocabulary,
                num_epochs: int = 10, learning_rate: float = 1e-4,
                device: str = 'cpu'):
    """Train the OM model."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get_index('<PAD>'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for images, captions, targets in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, captions)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return model


# ============================================================================
# DEMO AND TESTING
# ============================================================================

def demo_sarath_ai():
    """Demonstrate Sarath AI capabilities."""
    print("=" * 60)
    print("SARATH AI - Open Media Intelligence")
    print("OM Model (Open Media) Demo")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Initializing OM Model...")
    model = OMModel(vocab_size=10000)
    print("   ✓ Model initialized successfully")
    
    # Create vocabulary
    print("\n2. Creating vocabulary...")
    vocab = Vocabulary()
    sample_words = ['a', 'photo', 'of', 'beautiful', 'landscape', 'mountains', 
                   'river', 'sunset', 'city', 'building', 'people', 'street']
    for word in sample_words:
        vocab.add_word(word)
    print(f"   ✓ Vocabulary created with {vocab.n_words} words")
    
    # Create a test image
    print("\n3. Creating test image...")
    test_image = Image.new('RGB', (224, 224), color='blue')
    print("   ✓ Test image created (224x224 blue image)")
    
    # Test image description
    print("\n4. Testing image description...")
    try:
        description = model.describe_image(test_image, vocab)
        print(f"   Generated description: '{description}'")
    except Exception as e:
        print(f"   Note: Description generation requires trained model. Error: {e}")
    
    # Test image editing
    print("\n5. Testing image editing...")
    edited_image = model.edit_image(test_image, edit_type='enhance')
    print("   ✓ Image enhancement applied")
    
    edited_filter = model.edit_image(test_image, edit_type='filter', 
                                    params={'filter': 'sharpen'})
    print("   ✓ Filter applied (sharpen)")
    
    # Initialize agent
    print("\n6. Initializing Sarath AI Agent...")
    agent = OMAgent(local_model=model, api_provider='openai')
    print("   ✓ Agent initialized (API mode ready when key provided)")
    
    # Test agent processing
    print("\n7. Testing agent processing...")
    result = agent.process(test_image, task='describe', vocab=vocab, use_api=False)
    print(f"   Agent result: {result}")
    
    # Save model
    print("\n8. Saving model...")
    model.save_model('om_model.pth', vocab)
    print("   ✓ Model saved to 'om_model.pth'")
    print("   ✓ Vocabulary saved to 'om_model_vocab.json'")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nSarath AI is ready to use!")
    print("\nUsage examples:")
    print("  from om_model import OMModel, OMAgent, Vocabulary")
    print("  ")
    print("  # Initialize")
    print("  model = OMModel()")
    print("  agent = OMAgent(model, api_provider='openai', api_key='your-key')")
    print("  ")
    print("  # Process images")
    print("  image = Image.open('photo.jpg')")
    print("  description = agent.process(image, task='describe')")
    print("  edited = agent.process(image, task='edit', text_input='make it brighter')")
    print("  ")
    print("Developed by Open Media Intelligence")
    print("Lead AI: Sarath")


if __name__ == '__main__':
    demo_sarath_ai()
