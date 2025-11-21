from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
import torch
import torch.nn.functional as F
import json
import base64
import io
from .utils import tensor_from_base64, _preprocess_canvas_png
from .transliterator import get_transliterator

def home(request):
    return render(request, 'landing/index.html')

MODEL_PATH = Path(__file__).resolve().parent / "model" / "emnist_letters_traced.pt"
IDX_TO_CHAR_PATH = Path(__file__).resolve().parent / "model" / "idx_to_char.json"

# Load model and character mapping
_model = None
_idx_to_char = {}

try:
    _model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
    _model.eval()
    
    with open(IDX_TO_CHAR_PATH, 'r') as f:
        _idx_to_char = json.load(f)
except FileNotFoundError as e:
    print(f"Warning: Could not load model or mapping: {e}")
except Exception as e:
    print(f"Error loading model: {e}")

@csrf_exempt
def predict(request):
    """Handle handwriting canvas prediction requests"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    if _model is None:
        return JsonResponse({'error': 'Model not loaded'}, status=500)
    
    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        
        if not image_data:
            return JsonResponse({'error': 'No image data provided'}, status=400)
        
        # Convert base64 image to tensor and get preprocessed image for debugging
        input_tensor, preprocessed_img, debug_info = _preprocess_canvas_png(image_data, debug=True)
        
        # Debug: Save preprocessed image to check visually
        debug_mode = data.get('debug', False)
        preprocessed_base64 = None
        if debug_mode:
            import io
            import base64
            buf = io.BytesIO()
            preprocessed_img.save(buf, format='PNG')
            preprocessed_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        # Add tensor statistics to debug
        if debug_mode:
            debug_info['tensor_shape'] = list(input_tensor.shape)
            debug_info['tensor_min'] = float(input_tensor.min())
            debug_info['tensor_max'] = float(input_tensor.max())
            debug_info['tensor_mean'] = float(input_tensor.mean())
            debug_info['tensor_std'] = float(input_tensor.std())
        
        # Run inference
        with torch.no_grad():
            logits = _model(input_tensor)  # shape: [1, 26]
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, k=3, dim=1)
        
        # Add logits to debug
        if debug_mode:
            debug_info['logits'] = logits[0].tolist()[:5]  # First 5 logits
        
        # Convert to predictions
        # EMNIST Letters: model outputs 0-25 but training used labels 1-26
        # So we need to map: model_idx -> training_label -> letter
        # training_label = model_idx + 1, letter = ascii[training_label - 1] = ascii[model_idx]
        # BUT the model might already account for this, so let's use the mapping as-is for now
        predictions = []
        import string
        for i in range(3):
            idx = top_indices[0, i].item()
            prob = top_probs[0, i].item()
            # Try direct ascii mapping (0='a', 1='b', ..., 25='z')
            char = string.ascii_lowercase[int(idx)] if 0 <= idx < 26 else '?'
            predictions.append({
                'char': char,
                'confidence': float(prob),
                'model_index': idx  # For debugging
            })
        
        response = {
            'predictions': predictions,
            'success': True
        }
        
        if debug_mode:
            if preprocessed_base64:
                response['preprocessed_image'] = f"data:image/png;base64,{preprocessed_base64}"
            response['debug_info'] = debug_info
        
        return JsonResponse(response)
        
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def transliterate(request):
    """Handle SingKhmer to Khmer transliteration requests"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        singkhmer = data.get('text', '').strip()
        
        if not singkhmer:
            return JsonResponse({'error': 'No text provided'}, status=400)
        
        # Get transliterator instance
        transliterator = get_transliterator()
        
        # Get top 3 candidates
        candidates = transliterator.translate(singkhmer, top_k=3)
        
        # Ensure we always have 3 candidates (pad with empty if needed)
        while len(candidates) < 3:
            candidates.append('')
        
        # Return with best candidate in the middle (iOS style)
        # Order: [2nd best, 1st best, 3rd best]
        response = {
            'candidates': [
                candidates[1] if len(candidates) > 1 else '',  # Left (2nd best)
                candidates[0] if len(candidates) > 0 else '',  # Middle (best)
                candidates[2] if len(candidates) > 2 else ''   # Right (3rd best)
            ],
            'success': True
        }
        
        return JsonResponse(response)
        
    except Exception as e:
        import traceback
        print(f"Transliteration error: {e}")
        print(traceback.format_exc())
        return JsonResponse({'error': str(e), 'success': False}, status=500)

# Grammar checker view
from .grammar_checker import analyze_sentence, get_predictor
import numpy as np

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def grammar_checker(request):
    """Render grammar checker page allowing Khmer sentence validation."""
    # Handle AJAX requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        if request.method == 'POST':
            sentence = request.POST.get('sentence', '').strip()
            if not sentence:
                return JsonResponse({'error': 'Please enter a sentence.'}, status=400)
            try:
                result = analyze_sentence(sentence)
                # Convert numpy types to native Python types
                result = convert_to_serializable(result)
                return JsonResponse({'result': result})
            except Exception as e:
                import traceback
                print(f"Grammar check error: {e}")
                print(traceback.format_exc())
                return JsonResponse({'error': str(e)}, status=500)
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    # Handle regular page requests
    context = {
        'result': None,
        'error': None,
        'sentence': '',
        'loaded': get_predictor() is not None,
    }
    if request.method == 'POST':
        sentence = request.POST.get('sentence', '').strip()
        context['sentence'] = sentence
        if sentence:
            try:
                result = analyze_sentence(sentence)
                context['result'] = result
            except Exception as e:
                context['error'] = str(e)
        else:
            context['error'] = 'Please enter a sentence.'
    return render(request, 'landing/grammar_checker.html', context)
