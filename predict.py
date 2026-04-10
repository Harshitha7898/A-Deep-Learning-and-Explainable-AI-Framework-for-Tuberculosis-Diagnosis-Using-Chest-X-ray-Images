import torch
import torch.nn.functional as F
import numpy as np
from model_loader import model_loader

class Predictor:
    def __init__(self):
        self.class_names = ["Normal", "Tuberculosis"]
    
    def predict_single(self, model, input_tensor):
        """Get prediction from a single model"""
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            prob_values = probs.cpu().numpy()[0]
            predicted_class = np.argmax(prob_values)
            confidence = prob_values[predicted_class] * 100
            
            return {
                "label": self.class_names[predicted_class],
                "confidence": round(confidence, 1),
                "probs": [round(float(p) * 100, 1) for p in prob_values]
            }
    
    def predict_all(self, input_tensor):
        """Run predictions on all loaded models"""
        results = {}
        
        # Predict with each model
        for model_name, model in model_loader.models.items():
            if model is not None:
                results[model_name] = self.predict_single(model, input_tensor)
        
        # Calculate ensemble (average softmax of ConvNeXt + ViT)
        if 'convnext' in results and 'vit' in results:
            convnext_probs = results['convnext']['probs']
            vit_probs = results['vit']['probs']
            ensemble_probs = [(convnext_probs[0] + vit_probs[0]) / 2, 
                              (convnext_probs[1] + vit_probs[1]) / 2]
            ensemble_class = np.argmax(ensemble_probs)
            ensemble_confidence = ensemble_probs[ensemble_class]
            
            results['ensemble'] = {
                "label": self.class_names[ensemble_class],
                "confidence": round(ensemble_confidence, 1),
                "probs": [round(p, 1) for p in ensemble_probs]
            }
        
        # Calculate final agreement
        if len(results) >= 2:
            labels = [r['label'] for r in results.values()]
            final_label = max(set(labels), key=labels.count)
            agreement_count = labels.count(final_label)
            
            # Get confidence from ensemble if available, otherwise from majority
            if 'ensemble' in results:
                final_confidence = results['ensemble']['confidence']
            else:
                final_confidence = max([r['confidence'] for r in results.values()])
            
            results['final'] = {
                "label": final_label,
                "confidence": final_confidence,
                "agreement_count": agreement_count
            }
        
        return results

predictor = Predictor()