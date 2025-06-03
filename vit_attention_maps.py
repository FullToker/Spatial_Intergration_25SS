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

    def get_attention_maps(self, image: Image.Image, layer_idx: int = -1) -> np.ndarray:
        """
        Extract attention maps from ViT model.

        Args:
            image: PIL Image
            layer_idx: Which layer's attention to visualize (-1 for last layer)

        Returns:
            Attention map as numpy array
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = outputs.attentions  # Tuple of attention weights for each layer

        # Get attention from specified layer
        attention = attentions[
            layer_idx
        ]  # Shape: [batch_size, num_heads, seq_len, seq_len]

        # Average across heads and remove batch dimension
        attention = attention.squeeze(0).mean(dim=0)  # Shape: [seq_len, seq_len]

        # Remove CLS token (first token) and focus on patch tokens
        attention = attention[1:, 1:]  # Shape: [num_patches, num_patches]

        # Average attention across all patches (how much each patch attends to others)
        attention_map = attention.mean(dim=0)  # Shape: [num_patches]

        return attention_map.cpu().numpy()

    def visualize_attention(
        self,
        image_path: str,
        layer_idx: int = -1,
        alpha: float = 0.6,
        colormap: str = "jet",
        save_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize attention on the input image.

        Args:
            image_path: Path to input image
            layer_idx: Which layer's attention to visualize
            alpha: Transparency of attention overlay
            colormap: Matplotlib colormap for attention
            save_path: Path to save the visualization

        Returns:
            Tuple of (original_image, attention_visualization)
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

        Args:
            image_path: Path to input image
            layers: List of layer indices to compare
            save_path: Path to save the comparison
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


# Example usage
def main():
    # Initialize the visualizer with cache in current folder
    print("Initializing ViT Attention Visualizer...")
    visualizer = ViTAttentionVisualizer(cache_dir="./models")

    # Example with a map image (replace with your map image path)
    image_path = "./src/sample4.png"  # Replace with actual path

    try:
        # Visualize attention for the last layer
        print("Generating attention visualization...")
        visualizer.visualize_attention(
            image_path=image_path,
            layer_idx=-1,
            alpha=0.6,
            save_path="./src/" + "S4_att_ori.png",
        )

        # Compare multiple layers
        print("Comparing attention across layers...")
        visualizer.compare_layers(
            image_path=image_path,
            layers=[-4, -3, -2, -1],
            save_path="./src/" + "S4_layers_ori.png",
        )

    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to provide a valid image path.")


if __name__ == "__main__":
    main()

# Additional utility functions for map-specific analysis


class MapAttentionAnalyzer(ViTAttentionVisualizer):
    """Extended class specifically for analyzing map images."""

    def get_high_attention_regions(
        self, image_path: str, threshold: float = 0.7, min_region_size: int = 100
    ) -> list:
        """
        Identify regions with high attention values.

        Args:
            image_path: Path to map image
            threshold: Attention threshold (0-1)
            min_region_size: Minimum size of regions to consider

        Returns:
            List of high attention regions with coordinates
        """
        image = self.load_image(image_path)
        attention_map = self.get_attention_maps(image)

        grid_size = int(np.sqrt(self.num_patches))
        attention_2d = attention_map.reshape(grid_size, grid_size)

        # Normalize
        attention_norm = (attention_2d - attention_2d.min()) / (
            attention_2d.max() - attention_2d.min()
        )

        # Find high attention regions
        high_attention_mask = attention_norm > threshold

        # Find connected components
        num_labels, labels = cv2.connectedComponents(
            high_attention_mask.astype(np.uint8)
        )

        regions = []
        for label in range(1, num_labels):
            region_mask = labels == label
            if np.sum(region_mask) >= min_region_size:
                y_coords, x_coords = np.where(region_mask)
                regions.append(
                    {
                        "bbox": (
                            x_coords.min(),
                            y_coords.min(),
                            x_coords.max(),
                            y_coords.max(),
                        ),
                        "center": (x_coords.mean(), y_coords.mean()),
                        "attention_score": attention_norm[region_mask].mean(),
                        "size": len(x_coords),
                    }
                )

        return sorted(regions, key=lambda x: x["attention_score"], reverse=True)

    def analyze_geographic_features(self, image_path: str):
        """
        Analyze what geographic features the model focuses on.
        This is a basic implementation - you might want to extend this
        based on your specific map types and requirements.
        """
        regions = self.get_high_attention_regions(image_path)

        print("High Attention Regions Analysis:")
        print("=" * 50)

        for i, region in enumerate(regions[:5]):  # Top 5 regions
            print(f"Region {i + 1}:")
            print(f"  - Attention Score: {region['attention_score']:.3f}")
            print(f"  - Center: ({region['center'][0]:.1f}, {region['center'][1]:.1f})")
            print(f"  - Bounding Box: {region['bbox']}")
            print(f"  - Size: {region['size']} patches")
            print()

        return regions


# Usage example for maps
"""
# Initialize map analyzer with custom cache directory
map_analyzer = MapAttentionAnalyzer(cache_dir="./models")

# Or use default cache in current folder
map_analyzer = MapAttentionAnalyzer()

# Analyze a map image
map_path = "your_map.jpg"
map_analyzer.visualize_attention(map_path, save_path="map_attention.png")
regions = map_analyzer.analyze_geographic_features(map_path)
"""
