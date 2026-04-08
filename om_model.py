"""
OM (Open Media) - AI Model for Image Editing and Text Generation
=================================================================
Developed by: Open Media Intelligence
Version: 2.0.0 (Agent Mode Enabled)

This module provides the OM (Open Media) AI model that can:
1. Edit images based on text instructions (Local)
2. Generate descriptive text from images (Local)
3. Agent Mode: Connect with external AI Model APIs for advanced tasks
4. Perform various image manipulation tasks

Note: This implementation uses pure PyTorch and PIL for maximum compatibility.
For production use with pre-trained models, install torchvision separately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import requests
import json
import os
from typing import Optional, Tuple, Dict, Any, List, Union

print("--- OM (Open Media) by Open Media Intelligence ---")
print("Initializing Core Engine...")


class SimpleCNNBackbone(nn.Module):
    """
    Simple CNN backbone for image encoding (used when torchvision is not available).
    Can be replaced with ResNet/ViT when torchvision is installed.
    """
    
    def __init__(self, out_channels: int = 512):
        super(SimpleCNNBackbone, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.out_channels = out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return features.squeeze(-1).squeeze(-1)


try:
    from torchvision import transforms, models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("Note: torchvision not available. Using simple CNN backbone instead of pre-trained ResNet.")
    print("For better performance, install torchvision: pip install torchvision")


class ImageEncoder(nn.Module):
    """
    Encodes images into latent representations using a CNN backbone.
    Uses pre-trained ResNet if torchvision is available, otherwise uses simple CNN.
    """
    
    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True):
        super(ImageEncoder, self).__init__()
        
        if HAS_TORCHVISION and model_name in ['resnet50', 'resnet34']:
            # Load pre-trained ResNet
            if model_name == 'resnet50':
                backbone = models.resnet50(pretrained=pretrained)
                self.features = nn.Sequential(*list(backbone.children())[:-2])
                self.out_channels = 2048
            elif model_name == 'resnet34':
                backbone = models.resnet34(pretrained=pretrained)
                self.features = nn.Sequential(*list(backbone.children())[:-2])
                self.out_channels = 512
            
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.use_simple = False
        else:
            # Use simple CNN backbone
            self.features = SimpleCNNBackbone(out_channels=512).features
            self.out_channels = 512
            self.use_simple = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Encoded features of shape (batch_size, out_channels)
        """
        features = self.features(x)
        if not self.use_simple:
            features = self.pool(features)
        return features.squeeze(-1).squeeze(-1)


class TextDecoder(nn.Module):
    """
    Decoder module that generates text from image features.
    Uses LSTM-based architecture for sequence generation.
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int = 512, 
        hidden_size: int = 512, 
        num_layers: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.5
    ):
        super(TextDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        features: torch.Tensor, 
        captions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        Args:
            features: Image features of shape (batch_size, feature_dim)
            captions: Ground truth captions for training (batch_size, seq_length)
        
        Returns:
            outputs: Logits of shape (batch_size, seq_length, vocab_size)
            predictions: Generated token indices
        """
        batch_size = features.size(0)
        
        # Project image features to hidden size
        h0 = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        if captions is not None:
            # Training mode: use teacher forcing
            embeddings = self.dropout(self.embedding(captions))
            outputs, _ = self.lstm(embeddings, (h0, c0))
            outputs = self.fc(outputs)
            return outputs, None
        else:
            # Inference mode: generate tokens autoregressively
            predictions = []
            inputs = torch.ones(batch_size, 1).long().to(features.device)  # Start token
            
            for _ in range(self.max_seq_length):
                embeddings = self.dropout(self.embedding(inputs))
                output, (h0, c0) = self.lstm(embeddings, (h0, c0))
                logits = self.fc(output.squeeze(1))
                
                # Get next token
                probs = F.softmax(logits, dim=1)
                _, predicted = torch.max(probs, 1)
                
                predictions.append(predicted.cpu().numpy())
                inputs = predicted.unsqueeze(1)
                
                # Stop if all sequences generated EOS
                if (predicted == 0).all():
                    break
            
            # Convert to list format
            if predictions:
                # Stack predictions and convert to list
                pred_array = np.stack(predictions, axis=1)  # (batch, seq_len)
                pred_list = pred_array.tolist()
            else:
                pred_list = []
            return None, pred_list


class ImageEditHead(nn.Module):
    """
    Neural network head for image editing operations.
    Predicts transformation parameters or pixel-wise modifications.
    """
    
    def __init__(self, input_channels: int = 2048, edit_type: str = 'filter'):
        super(ImageEditHead, self).__init__()
        
        self.edit_type = edit_type
        
        if edit_type == 'filter':
            # Predict filter parameters
            self.fc = nn.Sequential(
                nn.Linear(input_channels, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 9)  # 3x3 convolution kernel
            )
        elif edit_type == 'enhance':
            # Predict enhancement parameters (brightness, contrast, saturation, etc.)
            self.fc = nn.Sequential(
                nn.Linear(input_channels, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 6)  # brightness, contrast, saturation, sharpness, hue, gamma
            )
        elif edit_type == 'segmentation':
            # Pixel-wise segmentation
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 2, 1)  # Binary segmentation
            )
        else:
            raise ValueError(f"Unknown edit type: {edit_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class OMModel(nn.Module):
    """
    OM (Omni Modal) - Main AI Model for Image Editing and Text Generation
    
    This model combines:
    1. Image understanding (encoding)
    2. Text generation (captioning/description)
    3. Image editing capabilities
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_size: int = 512,
        hidden_size: int = 512,
        encoder_model: str = 'resnet50',
        max_seq_length: int = 50
    ):
        super(OMModel, self).__init__()
        
        self.name = "OM"
        self.version = "1.0.0"
        
        # Image Encoder
        self.encoder = ImageEncoder(model_name=encoder_model)
        
        # Text Decoder
        self.decoder = TextDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            max_seq_length=max_seq_length
        )
        
        # Image Editing Heads
        self.edit_filter = ImageEditHead(input_channels=self.encoder.out_channels, edit_type='filter')
        self.edit_enhance = ImageEditHead(input_channels=self.encoder.out_channels, edit_type='enhance')
        
        # Projection layer to match encoder output to decoder input
        self.projection = nn.Linear(self.encoder.out_channels, hidden_size)
        
    def forward(
        self, 
        images: torch.Tensor, 
        captions: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass of the OM model.
        
        Args:
            images: Input images of shape (batch_size, 3, H, W)
            captions: Optional ground truth captions for training
        
        Returns:
            Dictionary containing:
            - text_logits: Text generation logits (if captions provided)
            - predictions: Generated text tokens (if no captions)
            - image_features: Encoded image features
            - edit_params: Predicted editing parameters
        """
        # Encode image
        image_features = self.encoder(images)
        
        # Project features for decoder
        projected_features = self.projection(image_features)
        
        # Generate text
        if captions is not None:
            text_logits = self.decoder(projected_features, captions)
            predictions = None
        else:
            text_logits, predictions = self.decoder(projected_features, None)
        
        # Get editing parameters
        filter_params = self.edit_filter(image_features)
        enhance_params = self.edit_enhance(image_features)
        
        return {
            'text_logits': text_logits,
            'predictions': predictions,
            'image_features': image_features,
            'filter_params': filter_params,
            'enhance_params': enhance_params
        }
    
    def describe_image(self, image: Image.Image, tokenizer=None, idx_to_word=None) -> str:
        """
        Generate a text description of the input image.
        
        Args:
            image: PIL Image object
            tokenizer: Function to tokenize text (optional)
            idx_to_word: Mapping from token indices to words
        
        Returns:
            Generated text description
        """
        self.eval()
        
        # Transform image (works with or without torchvision)
        if HAS_TORCHVISION:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Manual transform implementation without torchvision
            def transform(img):
                img = img.resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                # Normalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = (img_array - mean) / std
                # Convert to CHW format
                img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
                return img_tensor
        
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.forward(image_tensor)
            predictions = outputs['predictions']
        
        # Convert tokens to text
        if idx_to_word and predictions:
            words = [idx_to_word.get(idx, '<UNK>') for idx in predictions[0]]
            description = ' '.join(words)
        else:
            description = f"Generated description with {len(predictions[0]) if predictions else 0} tokens"
        
        return description
    
    def edit_image(
        self, 
        image: Image.Image, 
        edit_type: str = 'enhance',
        params: Optional[Dict[str, float]] = None
    ) -> Image.Image:
        """
        Edit the input image based on specified parameters.
        
        Args:
            image: PIL Image object
            edit_type: Type of edit ('enhance', 'filter', 'segmentation')
            params: Manual editing parameters (optional)
        
        Returns:
            Edited PIL Image
        """
        self.eval()
        
        # Transform image (works with or without torchvision)
        if HAS_TORCHVISION:
            transform_module = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Manual transform implementation without torchvision
            def transform_module(img):
                img = img.resize((224, 224))
                img_array = np.array(img).astype(np.float32) / 255.0
                # Normalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_array = (img_array - mean) / std
                # Convert to CHW format
                img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
                return img_tensor
        
        image_tensor = transform_module(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.forward(image_tensor)
            
            if edit_type == 'enhance':
                if params is None:
                    enhance_params = outputs['enhance_params'].cpu().numpy()[0]
                    params = {
                        'brightness': enhance_params[0],
                        'contrast': enhance_params[1],
                        'saturation': enhance_params[2],
                        'sharpness': enhance_params[3],
                        'hue': enhance_params[4],
                        'gamma': enhance_params[5]
                    }
                
                # Apply enhancements
                edited_image = self._apply_enhancements(image, params)
                
            elif edit_type == 'filter':
                if params is None:
                    filter_params = outputs['filter_params'].cpu().numpy()[0]
                    kernel = filter_params.reshape(3, 3)
                else:
                    kernel = np.array(params.get('kernel', np.zeros(9))).reshape(3, 3)
                
                edited_image = self._apply_filter(image, kernel)
                
            else:
                edited_image = image
        
        return edited_image
    
    def _apply_enhancements(self, image: Image.Image, params: Dict[str, float]) -> Image.Image:
        """Apply enhancement parameters to image."""
        from PIL import ImageEnhance
        
        enhanced = image.copy()
        
        # Brightness
        if 'brightness' in params:
            enhancer = ImageEnhance.Brightness(enhanced)
            factor = 1.0 + params['brightness'] * 0.5
            enhanced = enhancer.enhance(factor)
        
        # Contrast
        if 'contrast' in params:
            enhancer = ImageEnhance.Contrast(enhanced)
            factor = 1.0 + params['contrast'] * 0.5
            enhanced = enhancer.enhance(factor)
        
        # Saturation
        if 'saturation' in params:
            enhancer = ImageEnhance.Color(enhanced)
            factor = 1.0 + params['saturation'] * 0.5
            enhanced = enhancer.enhance(factor)
        
        # Sharpness
        if 'sharpness' in params:
            enhancer = ImageEnhance.Sharpness(enhanced)
            factor = 1.0 + params['sharpness'] * 0.5
            enhanced = enhancer.enhance(factor)
        
        return enhanced
    
    def _apply_filter(self, image: Image.Image, kernel: np.ndarray) -> Image.Image:
        """Apply convolution filter to image."""
        from PIL import ImageFilter
        
        # Normalize kernel
        kernel_sum = np.sum(np.abs(kernel))
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        
        # Convert to list format for PIL
        kernel_list = kernel.flatten().tolist()
        
        # Apply filter
        filtered_image = image.filter(ImageFilter.Kernel(
            size=(3, 3),
            kernel=kernel_list,
            scale=1.0,
            offset=0
        ))
        
        return filtered_image
    
    def save_model(self, path: str):
        """Save model weights to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'name': self.name,
            'version': self.version
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights from file."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


class OMAgent:
    """
    OM Agent Mode - Connects OM Model with External AI APIs
    
    This class orchestrates between:
    1. Local OM Model (lightweight operations)
    2. External AI Model APIs (advanced processing)
    
    Supports multiple API providers:
    - OpenAI (GPT-4V, DALL-E)
    - Anthropic (Claude)
    - Stability AI (Image generation/editing)
    - Custom endpoints
    """
    
    def __init__(
        self, 
        local_model: Optional[OMModel] = None,
        api_provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_local_first: bool = True
    ):
        self.local_model = local_model
        self.api_provider = api_provider
        self.api_key = api_key or os.getenv(f"{api_provider.upper()}_API_KEY")
        self.base_url = base_url or self._get_default_base_url(api_provider)
        self.use_local_first = use_local_first
        
        # Session for API requests
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update(self._get_auth_headers())
            
        print(f"OM Agent initialized with provider: {api_provider}")
        if self.api_key:
            print("✓ API Key configured")
        else:
            print("⚠ No API Key found - using local mode only")
    
    def _get_default_base_url(self, provider: str) -> str:
        """Get default base URL for API provider."""
        urls = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "stability": "https://api.stability.ai/v1",
            "custom": ""
        }
        return urls.get(provider, "")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API provider."""
        if self.api_provider == "openai":
            return {"Authorization": f"Bearer {self.api_key}"}
        elif self.api_provider == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
        elif self.api_provider == "stability":
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}
    
    def describe_image_api(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Use external AI API to describe an image."""
        if not self.api_key:
            if self.local_model:
                print("Falling back to local model...")
                return self.local_model.describe_image(image)
            raise ValueError("No API key configured and no local model available")
        
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        if self.api_provider == "openai":
            return self._call_openai_vision(img_base64, prompt)
        elif self.api_provider == "anthropic":
            return self._call_anthropic_vision(img_base64, prompt)
        else:
            raise ValueError(f"Unsupported provider for vision: {self.api_provider}")
    
    def _call_openai_vision(self, img_base64: str, prompt: str) -> str:
        """Call OpenAI GPT-4 Vision API."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }],
            "max_tokens": 500
        }
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_anthropic_vision(self, img_base64: str, prompt: str) -> str:
        """Call Anthropic Claude Vision API."""
        url = f"{self.base_url}/messages"
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 500,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_base64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        }
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    
    def edit_image_api(self, image: Image.Image, instruction: str, output_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Use external AI API to edit an image based on text instruction."""
        if not self.api_key:
            if self.local_model:
                print("Falling back to local model...")
                return self._local_edit_from_instruction(image, instruction)
            raise ValueError("No API key configured and no local model available")
        
        if self.api_provider == "stability":
            return self._call_stability_edit(image, instruction, output_size)
        elif self.api_provider == "openai":
            return self._call_openai_edit(image, instruction, output_size)
        else:
            raise ValueError(f"Unsupported provider for image editing: {self.api_provider}")
    
    def _local_edit_from_instruction(self, image: Image.Image, instruction: str) -> Image.Image:
        """Map text instructions to local edit operations."""
        instruction_lower = instruction.lower()
        if self.local_model:
            if any(word in instruction_lower for word in ["bright", "light", "exposure"]):
                return self.local_model.edit_image(image, edit_type='enhance', params={'brightness': 0.3, 'contrast': 0.1})
            elif any(word in instruction_lower for word in ["dark", "dim"]):
                return self.local_model.edit_image(image, edit_type='enhance', params={'brightness': -0.3})
            elif any(word in instruction_lower for word in ["sharp", "clear", "focus"]):
                return self.local_model.edit_image(image, edit_type='enhance', params={'sharpness': 0.5})
            elif any(word in instruction_lower for word in ["blur", "soft"]):
                return image.filter(ImageFilter.GaussianBlur(radius=2))
            elif any(word in instruction_lower for word in ["gray", "grey", "black", "white", "monochrome"]):
                return image.convert('L').convert('RGB')
        return image
    
    def _call_stability_edit(self, image: Image.Image, instruction: str, output_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Call Stability AI for image editing."""
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        url = f"{self.base_url}/generation/{output_size[0] if output_size else 1024}-edit"
        payload = {"prompt": instruction, "image": img_base64, "mode": "image-editing"}
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return Image.open(BytesIO(base64.b64decode(result["images"][0])))
    
    def _call_openai_edit(self, image: Image.Image, instruction: str, output_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """Call OpenAI DALL-E for image editing."""
        from io import BytesIO
        url = f"{self.base_url}/images/edits"
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        files = {
            "image": buffered.getvalue(),
            "prompt": (None, instruction),
            "n": (None, "1"),
            "size": (None, f"{output_size[0]}x{output_size[1]}" if output_size else "1024x1024")
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        result = response.json()
        img_response = requests.get(result["data"][0]["url"])
        return Image.open(BytesIO(img_response.content))
    
    def process(self, image: Optional[Image.Image] = None, text_input: Optional[str] = None, task: str = "describe") -> Union[str, Image.Image, Dict[str, Any]]:
        """Unified processing method that decides between local and API processing."""
        if task == "describe":
            if not image:
                raise ValueError("Image required for description task")
            if self.use_local_first and self.local_model:
                try:
                    return self.local_model.describe_image(image)
                except Exception as e:
                    print(f"Local model failed: {e}")
                    if self.api_key:
                        return self.describe_image_api(image, text_input or "Describe this image.")
                    raise
            if self.api_key:
                return self.describe_image_api(image, text_input or "Describe this image.")
            raise ValueError("No processing method available")
        elif task == "edit":
            if not image:
                raise ValueError("Image required for edit task")
            if not text_input:
                raise ValueError("Instruction required for edit task")
            if self.use_local_first and self.local_model:
                try:
                    return self.local_model.edit_image(image, edit_type='enhance')
                except Exception as e:
                    print(f"Local model failed: {e}")
                    if self.api_key:
                        return self.edit_image_api(image, text_input)
                    raise
            if self.api_key:
                return self.edit_image_api(image, text_input)
            raise ValueError("No processing method available")
        elif task == "analyze":
            if not image:
                raise ValueError("Image required for analysis")
            results = {"local_analysis": None, "api_analysis": None}
            if self.local_model:
                try:
                    results["local_analysis"] = {"description": self.local_model.describe_image(image), "features_extracted": True}
                except Exception as e:
                    results["local_analysis"] = {"error": str(e)}
            if self.api_key:
                try:
                    results["api_analysis"] = self.describe_image_api(image, "Provide a detailed analysis of this image including objects, colors, mood, and context.")
                except Exception as e:
                    results["api_analysis"] = {"error": str(e)}
            return results
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def __repr__(self):
        return f"OMAgent(provider={self.api_provider}, local_model={self.local_model is not None}, api_configured={self.api_key is not None})"


class Vocabulary:
    """
    Simple vocabulary class for text tokenization.
    """
    
    def __init__(self, max_size: int = 10000):
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.max_size = max_size
        self.word_count = {}
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of texts."""
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_count[word] = self.word_count.get(word, 0) + 1
        
        # Sort by frequency and add to vocabulary
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words[:self.max_size - len(self.word2idx)]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token indices."""
        words = text.lower().split()
        return [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices to text."""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("OM (Open Media) AI Model - by Open Media Intelligence")
    print("=" * 60)
    
    # Initialize model
    model = OMModel(vocab_size=10000)
    print(f"\nModel Name: {model.name}")
    print(f"Model Version: {model.version}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample vocabulary
    sample_texts = [
        "a dog playing in the park",
        "a cat sitting on a couch",
        "beautiful sunset over the ocean",
        "mountain landscape with snow",
        "city street at night"
    ]
    
    vocab = Vocabulary(max_size=1000)
    vocab.build_vocab(sample_texts)
    print(f"\nVocabulary Size: {len(vocab)}")
    
    # Create dummy image for testing
    dummy_image = Image.new('RGB', (224, 224), color='blue')
    
    # Test image description
    print("\n" + "-" * 60)
    print("Testing Image Description:")
    print("-" * 60)
    description = model.describe_image(dummy_image, idx_to_word=vocab.idx2word)
    print(f"Description: {description}")
    
    # Test image editing
    print("\n" + "-" * 60)
    print("Testing Local Image Editing:")
    print("-" * 60)
    edited_image = model.edit_image(dummy_image, edit_type='enhance')
    print(f"Original Image Size: {dummy_image.size}")
    print(f"Edited Image Size: {edited_image.size}")
    
    # Save model
    print("\n" + "-" * 60)
    print("Saving Model:")
    print("-" * 60)
    model.save_model("om_model.pth")
    
    print("\n" + "=" * 60)
    print("OM Model initialized successfully!")
    print("=" * 60)

    # Test Agent Mode
    print("\n" + "=" * 60)
    print("Testing OM Agent Mode:")
    print("=" * 60)
    
    # Initialize agent with local model only (no API key)
    agent = OMAgent(local_model=model, api_provider="openai")
    print(f"\nAgent: {agent}")
    
    # Test local fallback
    print("\nTesting Agent with Local Fallback:")
    result = agent.process(image=dummy_image, task="describe")
    if isinstance(result, str) and len(result) > 100:
        print(f"Result: {result[:100]}...")
    else:
        print(f"Result: {result}")
    
    print("\n" + "=" * 60)
    print("OM (Open Media) Model initialized successfully!")
    print("Developed by: Open Media Intelligence")
    print("=" * 60)
