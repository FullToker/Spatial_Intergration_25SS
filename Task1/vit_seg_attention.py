import torch
import torch.nn.functional as F
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    DPTImageProcessor,
    DPTForSemanticSegmentation,
)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Optional, Union
import requests
from io import BytesIO
import os


class EnhancedViTAttentionVisualizer:
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        model_type: str = "classification",
        cache_dir: str = None,
    ):
        """
        Initialize the enhanced ViT attention visualizer supporting both classification and segmentation models.

        Args:
            model_name: HuggingFace model name for ViT
            model_type: Type of model - "classification", "segmentation", or "segformer"
            cache_dir: Directory to cache the model (defaults to ./models)
        """
        # Set cache directory to current folder if not specified
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "models")

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Downloading/loading model to: {cache_dir}")
        print(f"Model type: {model_type}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_name = model_name

        # Initialize different model types
        if model_type == "classification":
            self._init_classification_model(model_name, cache_dir)
        elif model_type == "segmentation":
            self._init_segmentation_model(model_name, cache_dir)
        elif model_type == "segformer":
            self._init_segformer_model(model_name, cache_dir)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"Model loaded successfully on {self.device}")

    def _init_classification_model(self, model_name: str, cache_dir: str):
        """Initialize ViT classification model."""
        self.processor = ViTImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = ViTForImageClassification.from_pretrained(
            model_name, output_attentions=True, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

        # Get patch size and image size from config
        self.patch_size = self.model.config.patch_size
        self.image_size = self.processor.size["height"]
        self.num_patches = (self.image_size // self.patch_size) ** 2

    def _init_segmentation_model(self, model_name: str, cache_dir: str):
        """Initialize ViT-based segmentation model (DPT)."""
        try:
            self.processor = DPTImageProcessor.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            self.model = DPTForSemanticSegmentation.from_pretrained(
                model_name, output_attentions=True, cache_dir=cache_dir
            ).to(self.device)
        except:
            # Fallback to generic ViT processor if DPT processor fails
            self.processor = ViTImageProcessor.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            # Create a custom segmentation model wrapper
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                model_name, output_attentions=True, cache_dir=cache_dir
            ).to(self.device)

        self.model.eval()

        # For segmentation models, we need to infer patch information
        if hasattr(self.model.config, "patch_size"):
            self.patch_size = self.model.config.patch_size
        else:
            self.patch_size = 16  # Default patch size

        if hasattr(self.processor, "size"):
            if isinstance(self.processor.size, dict):
                self.image_size = self.processor.size.get("height", 384)
            else:
                self.image_size = self.processor.size
        else:
            self.image_size = 384  # Default for many segmentation models

        self.num_patches = (self.image_size // self.patch_size) ** 2

    def _init_segformer_model(self, model_name: str, cache_dir: str):
        """Initialize Segformer model."""
        self.processor = SegformerImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, output_attentions=True, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

        # Segformer uses different patch sizes for different stages
        self.patch_size = 4  # Smallest patch size in Segformer
        self.image_size = 512  # Standard Segformer input size
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
        Extract attention maps from ViT model (classification or segmentation).

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
            if self.model_type == "classification":
                outputs = self.model(**inputs)
                attentions = outputs.attentions
            elif self.model_type == "segmentation":
                try:
                    outputs = self.model(**inputs)
                    if (
                        hasattr(outputs, "attentions")
                        and outputs.attentions is not None
                    ):
                        attentions = outputs.attentions
                    else:
                        # For some segmentation models, we need to access the backbone
                        if hasattr(self.model, "backbone"):
                            backbone_outputs = self.model.backbone(
                                inputs["pixel_values"]
                            )
                            attentions = backbone_outputs.attentions
                        else:
                            # Generic approach for models without explicit attention output
                            raise ValueError("Model does not provide attention outputs")
                except Exception as e:
                    print(f"Error extracting attention from segmentation model: {e}")
                    return np.zeros((self.num_patches,))
            elif self.model_type == "segformer":
                outputs = self.model(**inputs)
                # Segformer has multiple encoder stages, get the last one
                attentions = outputs.attentions[-1] if outputs.attentions else None

        if attentions is None:
            print("No attention weights available for this model")
            return np.zeros((self.num_patches,))

        # Get attention from specified layer
        if isinstance(attentions, (list, tuple)):
            attention = attentions[layer_idx]
        else:
            attention = attentions

        # Handle different attention tensor shapes
        if len(attention.shape) == 4:
            # Standard format: [batch_size, num_heads, seq_len, seq_len]
            attention = attention.squeeze(0).mean(dim=0)
        elif len(attention.shape) == 3:
            # Already squeezed: [num_heads, seq_len, seq_len]
            attention = attention.mean(dim=0)

        # For segmentation models, we might not have CLS token
        if self.model_type == "classification":
            # Remove CLS token (first token) and focus on patch tokens
            if attention.shape[0] > self.num_patches:
                attention = attention[1:, 1:]
        else:
            # For segmentation models, adjust based on actual sequence length
            seq_len = attention.shape[0]
            target_patches = min(seq_len, self.num_patches)
            attention = attention[:target_patches, :target_patches]

        # Average attention across all patches
        attention_map = attention.mean(dim=0)

        # Ensure we have the right number of patches
        if len(attention_map) != self.num_patches:
            # Resize if necessary
            grid_size = int(np.sqrt(len(attention_map)))
            if grid_size * grid_size == len(attention_map):
                attention_2d = attention_map.reshape(grid_size, grid_size)
                target_grid = int(np.sqrt(self.num_patches))
                attention_2d_resized = cv2.resize(
                    attention_2d.cpu().numpy(),
                    (target_grid, target_grid),
                    interpolation=cv2.INTER_CUBIC,
                )
                attention_map = torch.from_numpy(attention_2d_resized.flatten())

        return attention_map.cpu().numpy()

    def get_segmentation_prediction(self, image: Image.Image) -> np.ndarray:
        """
        Get segmentation prediction for segmentation models.

        Args:
            image: PIL Image

        Returns:
            Segmentation mask as numpy array
        """
        if self.model_type not in ["segmentation", "segformer"]:
            print("Segmentation prediction only available for segmentation models")
            return None

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            if self.model_type == "segformer":
                logits = outputs.logits
            else:
                logits = (
                    outputs.logits if hasattr(outputs, "logits") else outputs.prediction
                )

            # Resize logits to original image size
            logits = F.interpolate(
                logits,
                size=image.size[
                    ::-1
                ],  # PIL size is (width, height), we need (height, width)
                mode="bilinear",
                align_corners=False,
            )

            # Get predicted class for each pixel
            predicted_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        return predicted_mask

    def visualize_attention(
        self,
        image_path: str,
        layer_idx: int = -1,
        alpha: float = 0.6,
        colormap: str = "jet",
        save_path: Optional[str] = None,
        show_segmentation: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize attention on the input image with optional segmentation overlay.

        Args:
            image_path: Path to input image
            layer_idx: Which layer's attention to visualize
            alpha: Transparency of attention overlay
            colormap: Matplotlib colormap for attention
            save_path: Path to save the visualization
            show_segmentation: Whether to show segmentation results (for seg models)

        Returns:
            Tuple of (original_image, attention_visualization)
        """
        # Load and process image
        image = self.load_image(image_path)
        original_size = image.size

        # Get attention map
        attention_map = self.get_attention_maps(image, layer_idx)

        # Reshape attention map to 2D grid
        grid_size = int(np.sqrt(len(attention_map)))
        attention_2d = attention_map.reshape(grid_size, grid_size)

        # Resize attention map to original image size
        attention_resized = cv2.resize(
            attention_2d, original_size, interpolation=cv2.INTER_CUBIC
        )

        # Normalize attention map
        if attention_resized.max() > attention_resized.min():
            attention_resized = (attention_resized - attention_resized.min()) / (
                attention_resized.max() - attention_resized.min()
            )

        # Convert image to numpy array
        img_array = np.array(image)

        # Create visualization
        num_plots = (
            4
            if show_segmentation and self.model_type in ["segmentation", "segformer"]
            else 3
        )
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        if num_plots == 1:
            axes = [axes]

        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Attention map
        im = axes[1].imshow(attention_2d, cmap=colormap)
        axes[1].set_title(f"Attention Map (Layer {layer_idx})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(img_array)
        axes[2].imshow(attention_resized, cmap=colormap, alpha=alpha)
        axes[2].set_title("Attention Overlay")
        axes[2].axis("off")

        # Segmentation prediction (if applicable)
        if show_segmentation and self.model_type in ["segmentation", "segformer"]:
            seg_mask = self.get_segmentation_prediction(image)
            if seg_mask is not None:
                axes[3].imshow(seg_mask, cmap="tab20")
                axes[3].set_title("Segmentation Prediction")
                axes[3].axis("off")

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
            grid_size = int(np.sqrt(len(attention_map)))
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

            if attention_resized.max() > attention_resized.min():
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
    # Initialize visualizers for different model types
    print("=== Testing Classification Model ===")
    visualizer_cls = EnhancedViTAttentionVisualizer(
        model_name="google/vit-base-patch16-224",
        model_type="classification",
        cache_dir="./models",
    )

    print("\n=== Testing Segmentation Model ===")
    # Example segmentation models you can try:
    # "Intel/dpt-large-ade" - DPT model for ADE20K
    # "nvidia/segformer-b0-finetuned-ade-512-512" - Segformer model
    try:
        visualizer_seg = EnhancedViTAttentionVisualizer(
            model_name="nvidia/segformer-b0-finetuned-ade-512-512",
            model_type="segformer",
            cache_dir="./models",
        )
    except Exception as e:
        print(f"Could not load segmentation model: {e}")
        visualizer_seg = None

    # Example image path
    image_path = "./src/sample2.png"  # Replace with actual path

    try:
        # Test classification model
        print("\nGenerating attention visualization for classification model...")
        visualizer_cls.visualize_attention(
            image_path=image_path,
            layer_idx=-1,
            alpha=0.6,
            save_path="./src/S2_att_clf.png",
        )

        # Test segmentation model if available
        if visualizer_seg:
            print("\nGenerating attention visualization for segmentation model...")
            visualizer_seg.visualize_attention(
                image_path=image_path,
                layer_idx=-1,
                alpha=0.6,
                save_path="./src/S2_att_seg.png",
                show_segmentation=True,
            )

            # Compare layers for segmentation model
            print("\nComparing attention across layers for segmentation model...")
            visualizer_seg.compare_layers(
                image_path=image_path,
                layers=[-4, -3, -2, -1],
                save_path="./src/S2_layers_seg.png",
            )

    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to provide a valid image path.")


if __name__ == "__main__":
    main()
