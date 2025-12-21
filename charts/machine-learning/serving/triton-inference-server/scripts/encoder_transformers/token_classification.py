import numpy as np
import torch
from transformers import AutoModelForTokenClassification
import triton_python_backend_utils as pb_utils
from encoder_base import EncoderBaseModel

class TritonPythonModel(EncoderBaseModel):
    """Token Classification implementation (Fixed for EncoderBaseModel)"""
    
    def _init_output_config(self):
        """Initialize output tensor configuration"""
        # Check for 'predictions' output
        try:
            pred_config = pb_utils.get_output_config_by_name(self.model_config, "predictions")
            self.predictions_dtype = pb_utils.triton_string_to_numpy(pred_config["data_type"])
            self.has_predictions = True
        except:
            self.has_predictions = False

        # Check for optional 'logits' output
        try:
            logits_config = pb_utils.get_output_config_by_name(self.model_config, "logits")
            self.logits_dtype = pb_utils.triton_string_to_numpy(logits_config["data_type"])
            self.has_logits = True
        except:
            self.has_logits = False
            
        if not self.has_predictions and not self.has_logits:
            raise ValueError("Config must define at least 'predictions' or 'logits' output")
    
    def _load_model(self, model_location):
        return AutoModelForTokenClassification.from_pretrained(model_location)
    
    def _process_outputs(self, model_output, attention_mask):
        """Process per-token predictions and slice to original lengths"""
        # 1. Keep logits on CPU for slicing
        logits = model_output.logits.detach().cpu()
        
        # 2. Get original lengths (dummies will have length 0)
        original_lengths = attention_mask.sum(dim=1).tolist()
        
        results = []
        # Loop through the FULL batch (including bucket padding)
        for i in range(len(original_lengths)):
            seq_len = int(original_lengths[i])
            
            # SKIP if it's an XLA dummy sequence (optional but cleaner)
            if seq_len == 0:
                results.append({}) # Or skip entirely if slicing in base class
                continue

            item_data = {}
            if self.has_logits:
                # Slicing [seq_len, num_labels]
                item_data['logits'] = logits[i, :seq_len].numpy().astype(self.logits_dtype)
            
            if self.has_predictions:
                # argmax across label dimension
                preds = torch.argmax(logits[i, :seq_len], dim=-1)
                item_data['predictions'] = preds.numpy().astype(self.predictions_dtype)
                
            results.append(item_data)
        
        # The slice happens in encoder_base.py: results[:input_batch_size]
        return results
    
    def _create_response(self, request_results):
        """
        Creates a response for a single request. 
        request_results: List of dicts containing 'logits' and/or 'predictions'
        """
        output_tensors = []
        batch_size = len(request_results)
        max_seq_len = max(
            r['predictions'].shape[0] if self.has_predictions else r['logits'].shape[0] 
            for r in request_results
        )

        if self.has_predictions:
            # Pad sequences to max length in this specific request batch
            padded_preds = np.zeros((batch_size, max_seq_len), dtype=self.predictions_dtype)
            for i, res in enumerate(request_results):
                seq_len = res['predictions'].shape[0]
                padded_preds[i, :seq_len] = res['predictions']
            output_tensors.append(pb_utils.Tensor("predictions", padded_preds))

        if self.has_logits:
            num_labels = request_results[0]['logits'].shape[1]
            padded_logits = np.zeros((batch_size, max_seq_len, num_labels), dtype=self.logits_dtype)
            for i, res in enumerate(request_results):
                seq_len = res['logits'].shape[0]
                padded_logits[i, :seq_len, :] = res['logits']
            output_tensors.append(pb_utils.Tensor("logits", padded_logits))

        return pb_utils.InferenceResponse(output_tensors=output_tensors)
    
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Auto-complete Triton config for Token Classification"""
        inputs = [{"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]}]
        outputs = [
            {"name": "predictions", "data_type": "TYPE_INT32", "dims": [-1]},
            {"name": "logits", "data_type": "TYPE_FP32", "dims": [-1, -1]}
        ]

        config = auto_complete_model_config.as_dict()
        input_names = [i['name'] for i in config.get('input', [])]
        output_names = [o['name'] for o in config.get('output', [])]

        for inp in inputs:
            if inp['name'] not in input_names:
                auto_complete_model_config.add_input(inp)
        for out in outputs:
            if out['name'] not in output_names:
                auto_complete_model_config.add_output(out)

        auto_complete_model_config.set_model_transaction_policy(dict(decoupled=False))
        return auto_complete_model_config