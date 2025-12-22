"""
VLM model wrappers for robot state classification.

Available models (used in experiments):
- Qwen2.5-VL: Best performance, supports finetuning
- LLaVA / LLaVA-NeXT: Alternative VLM
- Gemma3: Google's multimodal model
- Phi-4: Microsoft's VLM

All models support:
- predict(): Single image prediction
- predict_v2(): Few-shot with sample images in one message
- predict_v3(): Few-shot with multi-turn conversation

Usage:
    from models.qwen_vlm_original import Qwen2_5Model
    model = Qwen2_5Model()
    
    # Zero-shot
    result = model.predict(image_path, prompt)
    
    # Few-shot with examples
    examples = [{"image_path": "img1.png", "text": "Label: initialize"}, ...]
    result = model.predict_v2(examples, target_image, prompt)
"""

# Original model files used in experiments
# Import these for backward compatibility
try:
    from .qwen_vlm_original import Qwen2_5Model
except ImportError:
    pass

try:
    from .llava_vlm import LlavaModel, LlavaNextModel
except ImportError:
    pass

try:
    from .gemma3_vlm import Gemma3Model
except ImportError:
    pass

try:
    from .phi4_vlm import PHI_4_Model
except ImportError:
    pass
