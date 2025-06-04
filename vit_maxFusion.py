import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Optional
import requests
from io import BytesIO
import os


class ViTAttentionVisualizer:
    def __init__(
        self, model_name: str = "google/vit-base-patch16-224", cache_dir: str = None
    ):
        """
        Initialize the ViT attention visualizer.

        Args:
            model_name: HuggingFace model name for ViT
            cache_dir: Directory to cache the model (defaults to ./models)
        """
        # Set cache directory to current folder if not specified
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "models")

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Downloading/loading model to: {cache_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = ViTForImageClassification.from_pretrained(
            model_name, output_attentions=True, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

        # Get patch size and image size from config
        self.patch_size = self.model.config.patch_size
        self.image_size = self.processor.size["height"]
        self.num_patches = (self.image_size // self.patch_size) ** 2

    def load_image(self, image_path: str) -> Image.Image:
        """Load image from file path or URL."""
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        return image.convert("RGB")

    def apply_discard_ratio(self, attention: torch.Tensor, discard_ratio: float = 0.9) -> torch.Tensor:
        """
        Apply discard ratio to attention weights, keeping only the top (1-discard_ratio) weights.
        
        Args:
            attention: Attention tensor of shape [seq_len, seq_len]
            discard_ratio: Ratio of weights to discard (0.9 means keep top 10%)
            
        Returns:
            Filtered attention tensor
        """
        if discard_ratio <= 0:
            return attention
            
        # Get threshold for each row (each token's attention to others)
        sorted_attention, _ = torch.sort(attention, dim=-1, descending=True)
        threshold_idx = max(1, int(attention.size(-1) * (1 - discard_ratio)))
        
        # Create threshold tensor
        thresholds = sorted_attention[:, threshold_idx-1:threshold_idx]
        
        # Create mask and apply
        mask = attention >= thresholds
        filtered_attention = attention * mask.float()
        
        # Renormalize to ensure rows sum to 1
        row_sums = filtered_attention.sum(dim=-1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-8)
        filtered_attention = filtered_attention / row_sums
        
        return filtered_attention

    def attention_rollout(self, attentions: tuple, start_layer: int = 0, 
                         discard_ratio: float = 0.0, use_max_fusion: bool = False) -> torch.Tensor:
        """
        Compute attention rollout with optional discard ratio and max fusion.
        
        Args:
            attentions: Tuple of attention tensors from all layers
            start_layer: Starting layer for rollout computation
            discard_ratio: Ratio of attention weights to discard
            use_max_fusion: Whether to use max fusion instead of multiplication
            
        Returns:
            Rollout attention tensor
        """
        if use_max_fusion:
            return self._max_fusion_rollout(attentions, start_layer, discard_ratio)
        else:
            return self._vanilla_rollout(attentions, start_layer, discard_ratio)
    
    def _vanilla_rollout(self, attentions: tuple, start_layer: int = 0, 
                        discard_ratio: float = 0.0) -> torch.Tensor:
        """Vanilla attention rollout with optional discard ratio."""
        # Start with identity matrix
        result = torch.eye(attentions[0].size(-1), device=self.device)
        
        for i in range(start_layer, len(attentions)):
            # Get attention for current layer [batch, heads, seq_len, seq_len]
            attention = attentions[i].squeeze(0).mean(dim=0)  # Average over heads
            
            # Apply discard ratio if specified
            if discard_ratio > 0:
                attention = self.apply_discard_ratio(attention, discard_ratio)
            
            # Add residual connection (identity matrix)
            attention = attention + torch.eye(attention.size(0), device=self.device)
            
            # Normalize
            attention = attention / attention.sum(dim=-1, keepdim=True)
            
            # Multiply with previous result
            result = torch.matmul(attention, result)
        
        return result
    
    def _max_fusion_rollout(self, attentions: tuple, start_layer: int = 0, 
                           discard_ratio: float = 0.9) -> torch.Tensor:
        """Max fusion rollout: take max attention across layers."""
        processed_attentions = []
        
        for i in range(start_layer, len(attentions)):
            # Get attention for current layer
            attention = attentions[i].squeeze(0).mean(dim=0)  # Average over heads
            
            # Apply discard ratio
            if discard_ratio > 0:
                attention = self.apply_discard_ratio(attention, discard_ratio)
            
            processed_attentions.append(attention)
        
        if not processed_attentions:
            return torch.eye(attentions[0].size(-1), device=self.device)
        
        # Stack and take maximum across layers
        stacked = torch.stack(processed_attentions, dim=0)
        max_attention, _ = torch.max(stacked, dim=0)
        
        # Add residual connection and normalize
        max_attention = max_attention + torch.eye(max_attention.size(0), device=self.device)
        max_attention = max_attention / max_attention.sum(dim=-1, keepdim=True)
        
        return max_attention

    def get_attention_maps(self, image: Image.Image, layer_idx: int = -1, 
                          discard_ratio: float = 0.0, use_max_fusion: bool = False,
                          use_cls_token: bool = True) -> np.ndarray:
        """
        Extract attention maps from ViT model with advanced options.

        Args:
            image: PIL Image
            layer_idx: Which layer's attention to visualize (-1 for rollout across all layers)
            discard_ratio: Ratio of attention weights to discard (0.0 means no discard)
            use_max_fusion: Whether to use max fusion for multi-layer attention
            use_cls_token: Whether to use CLS token attention or average patch attention

        Returns:
            Attention map as numpy array
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = outputs.attentions  # Tuple of attention weights for each layer

        if layer_idx == -1 or discard_ratio > 0 or use_max_fusion:
            # Use rollout method for advanced attention computation
            rollout_attention = self.attention_rollout(
                attentions, start_layer=0, discard_ratio=discard_ratio, 
                use_max_fusion=use_max_fusion
            )
            
            if use_cls_token:
                # Use CLS token (index 0) attention to all patches
                attention_map = rollout_attention[0, 1:]  # Skip CLS->CLS attention
            else:
                # Average attention from all patches to all patches
                patch_attention = rollout_attention[1:, 1:]  # Remove CLS token
                attention_map = patch_attention.mean(dim=0)
        else:
            # Single layer attention (original method)
            attention = attentions[layer_idx].squeeze(0).mean(dim=0)  # Average over heads
            
            if use_cls_token:
                attention_map = attention[0, 1:]  # CLS token attention to patches
            else:
                # Remove CLS token and average
                attention = attention[1:, 1:]
                attention_map = attention.mean(dim=0)

        return attention_map.cpu().numpy()

    def visualize_attention_comparison(
        self,
        image_path: str,
        layer_idx: int = -1,
        discard_ratio: float = 0.9,
        alpha: float = 0.6,
        colormap: str = "jet",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a 4-column comparison: Original + Vanilla + Overlay + Advanced (discard+max fusion).

        Args:
            image_path: Path to input image
            layer_idx: Which layer's attention to visualize
            discard_ratio: Ratio of attention weights to discard for advanced method
            alpha: Transparency of attention overlay
            colormap: Matplotlib colormap for attention
            save_path: Path to save the visualization
        """
        # Load and process image
        image = self.load_image(image_path)
        img_array = np.array(image)

        # Get different attention maps
        vanilla_attention = self.get_attention_maps(image, layer_idx, discard_ratio=0.0, use_max_fusion=False)
        advanced_attention = self.get_attention_maps(image, layer_idx, discard_ratio=discard_ratio, use_max_fusion=True)

        # Reshape to 2D grids
        grid_size = int(np.sqrt(self.num_patches))
        vanilla_2d = vanilla_attention.reshape(grid_size, grid_size)
        advanced_2d = advanced_attention.reshape(grid_size, grid_size)

        # Resize to original image size
        original_size = image.size
        vanilla_resized = cv2.resize(vanilla_2d, original_size, interpolation=cv2.INTER_CUBIC)
        advanced_resized = cv2.resize(advanced_2d, original_size, interpolation=cv2.INTER_CUBIC)

        # Normalize attention maps
        vanilla_resized = (vanilla_resized - vanilla_resized.min()) / (vanilla_resized.max() - vanilla_resized.min())
        advanced_resized = (advanced_resized - advanced_resized.min()) / (advanced_resized.max() - advanced_resized.min())

        # Create 4-column visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Column 1: Original Image
        axes[0].imshow(img_array)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis("off")

        # Column 2: Vanilla Attention Map
        im1 = axes[1].imshow(vanilla_resized, cmap=colormap)
        axes[1].set_title("Vanilla Attention Rollout", fontsize=14, fontweight='bold')
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Column 3: Vanilla Overlay
        axes[2].imshow(img_array)
        axes[2].imshow(vanilla_resized, cmap=colormap, alpha=alpha)
        axes[2].set_title("Vanilla Overlay", fontsize=14, fontweight='bold')
        axes[2].axis("off")

        # Column 4: Advanced Method (Discard + Max Fusion)
        axes[3].imshow(img_array)
        axes[3].imshow(advanced_resized, cmap=colormap, alpha=alpha)
        axes[3].set_title(f"Discard Ratio {discard_ratio} + Max Fusion", fontsize=14, fontweight='bold')
        axes[3].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def visualize_attention(
        self,
        image_path: str,
        layer_idx: int = -1,
        alpha: float = 0.6,
        colormap: str = "jet",
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Original visualization method (kept for backward compatibility).
        """
        # Load and process image
        image = self.load_image(image_path)
        original_size = image.size

        # Get attention map
        attention_map = self.get_attention_maps(image, layer_idx)

        # Reshape attention map to 2D grid
        grid_size = int(np.sqrt(self.num_patches))
        attention_2d = attention_map.reshape(grid_size, grid_size)

        # Resize attention map to original image size
        attention_resized = cv2.resize(
            attention_2d, original_size, interpolation=cv2.INTER_CUBIC
        )

        # Normalize attention map
        attention_resized = (attention_resized - attention_resized.min()) / (
            attention_resized.max() - attention_resized.min()
        )

        # Convert image to numpy array
        img_array = np.array(image)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Attention map
        im = axes[1].imshow(attention_resized, cmap=colormap)
        axes[1].set_title(f"Attention Map (Layer {layer_idx})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(img_array)
        axes[2].imshow(attention_resized, cmap=colormap, alpha=alpha)
        axes[2].set_title("Attention Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

        return img_array, attention_resized

    def compare_layers(
        self,
        image_path: str,
        layers: list = [-4, -3, -2, -1],
        save_path: Optional[str] = None,
    ):
        """
        Compare attention maps across different layers.
        """
        image = self.load_image(image_path)

        fig, axes = plt.subplots(2, len(layers), figsize=(4 * len(layers), 8))
        if len(layers) == 1:
            axes = axes.reshape(2, 1)

        for i, layer_idx in enumerate(layers):
            attention_map = self.get_attention_maps(image, layer_idx)
            grid_size = int(np.sqrt(self.num_patches))
            attention_2d = attention_map.reshape(grid_size, grid_size)

            # Raw attention map
            im1 = axes[0, i].imshow(attention_2d, cmap="jet")
            axes[0, i].set_title(f"Layer {layer_idx}")
            axes[0, i].axis("off")

            # Overlay on original image
            img_resized = image.resize((grid_size * 10, grid_size * 10))
            attention_resized = cv2.resize(
                attention_2d,
                (grid_size * 10, grid_size * 10),
                interpolation=cv2.INTER_CUBIC,
            )
            attention_resized = (attention_resized - attention_resized.min()) / (
                attention_resized.max() - attention_resized.min()
            )

            axes[1, i].imshow(np.array(img_resized))
            axes[1, i].imshow(attention_resized, cmap="jet", alpha=0.5)
            axes[1, i].set_title(f"Layer {layer_idx} Overlay")
            axes[1, i].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def compare_discard_ratios(
        self,
        image_path: str,
        discard_ratios: list = [0.0, 0.7, 0.85, 0.9],
        save_path: Optional[str] = None,
    ):
        """
        Compare different discard ratios to show the effect.
        
        Args:
            image_path: Path to input image
            discard_ratios: List of discard ratios to compare
            save_path: Path to save the visualization
        """
        image = self.load_image(image_path)
        img_array = np.array(image)
        
        fig, axes = plt.subplots(1, len(discard_ratios), figsize=(5 * len(discard_ratios), 5))
        if len(discard_ratios) == 1:
            axes = [axes]
        
        for i, discard_ratio in enumerate(discard_ratios):
            # Get attention with current discard ratio
            attention_map = self.get_attention_maps(
                image, layer_idx=-1, discard_ratio=discard_ratio, use_max_fusion=True
            )
            
            # Reshape and resize
            grid_size = int(np.sqrt(self.num_patches))
            attention_2d = attention_map.reshape(grid_size, grid_size)
            attention_resized = cv2.resize(attention_2d, image.size, interpolation=cv2.INTER_CUBIC)
            attention_resized = (attention_resized - attention_resized.min()) / (
                attention_resized.max() - attention_resized.min()
            )
            
            # Plot overlay
            axes[i].imshow(img_array)
            axes[i].imshow(attention_resized, cmap="jet", alpha=0.6)
            
            title = f"Discard Ratio: {discard_ratio}" if discard_ratio > 0 else "Vanilla (No Discard)"
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            axes[i].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()


# Example usage
def main():
    # Initialize the visualizer with cache in current folder
    print("Initializing ViT Attention Visualizer...")
    visualizer = ViTAttentionVisualizer(cache_dir="./models")

    # Example with a map image (replace with your map image path)
    image_path = "./src/sample2.png"  # Replace with actual path

    try:
        # Original 3-column visualization
        print("Generating original attention visualization...")
        visualizer.visualize_attention(
            image_path=image_path,
            layer_idx=-1,
            alpha=0.6,
            save_path="./src/S2_post_ori.png",
        )

        # NEW: 4-column comparison with advanced method
        print("Generating 4-column comparison...")
        visualizer.visualize_attention_comparison(
            image_path=image_path,
            layer_idx=-1,
            discard_ratio=0.9,  # Keep only top 10% attention weights
            alpha=0.6,
            save_path="./src/S2_post_comparison.png",
        )

        # NEW: Compare different discard ratios
        print("Comparing different discard ratios...")
        visualizer.compare_discard_ratios(
            image_path=image_path,
            discard_ratios=[0.0, 0.7, 0.85, 0.9],
            save_path="./src/S2_discard_comparison.png",
        )

        # Compare multiple layers
        print("Comparing attention across layers...")
        visualizer.compare_layers(
            image_path=image_path,
            layers=[-4, -3, -2, -1],
            save_path="./src/S2_post_layers.png",
        )

    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to provide a valid image path.")


if __name__ == "__main__":
    main()