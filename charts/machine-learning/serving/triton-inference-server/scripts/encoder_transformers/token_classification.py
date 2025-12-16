import numpy as np
from transformers import AutoModelForTokenClassification
import triton_python_backend_utils as pb_utils
from encoder_base import EncoderBaseModel

class TritonPythonModel(EncoderBaseModel):
    """Token Classification implementation (NER, POS tagging, etc.) using EncoderBaseModel"""
    
    def _init_output_config(self):
        """Initialize output tensor configuration for token classification"""
        predictions_config = pb_utils.get_output_config_by_name(self.model_config, "predictions")
        self.predictions_dtype = pb_utils.triton_string_to_numpy(predictions_config["data_type"])
        
        # Optional: Also support returning logits instead of predictions
        try:
            logits_config = pb_utils.get_output_config_by_name(self.model_config, "logits")
            self.logits_dtype = pb_utils.triton_string_to_numpy(logits_config["data_type"])
            self.return_logits = True
        except:
            self.return_logits = False
    
    def _load_model(self, model_location):
        """Load token classification model"""
        self.logger.log_info(f"Loading AutoModelForTokenClassification from {model_location}")
        return AutoModelForTokenClassification.from_pretrained(model_location)
    
    def _process_outputs(self, model_output, original_lengths):
        """Process token classification predictions - return per-token label predictions
        
        Args:
            model_output: Model output with .logits attribute
            original_lengths: List of original sequence lengths (before padding)
            
        Returns:
            List of prediction arrays, one per input, sliced to original length
        """
        # Get logits and move to CPU
        # Shape: [batch_size, seq_len, num_labels]
        logits = model_output.logits.detach().cpu()
        
        if self.return_logits:
            # Return raw logits for each token
            results = []
            for i in range(len(original_lengths)):
                seq_len = original_lengths[i]
                # Shape: [seq_len, num_labels]
                results.append({
                    'logits': logits[i, :seq_len].numpy().astype(self.logits_dtype).tolist()
                })
        else:
            # Return predicted label IDs
            predictions = logits.argmax(dim=-1)  # [batch_size, seq_len]
            results = []
            for i in range(len(original_lengths)):
                seq_len = original_lengths[i]
                # Shape: [seq_len]
                results.append({
                    'predictions': predictions[i, :seq_len].numpy().astype(self.predictions_dtype).tolist()
                })
        
        return results
    
    def _create_response(self, result):
        """Create Triton response from processed predictions
        
        Args:
            result: Dict with predictions or logits for single input
            
        Returns:
            pb_utils.InferenceResponse with prediction/logit tensors
        """
        if self.return_logits:
            logits_tensor = pb_utils.Tensor("logits", np.array(result['logits'], dtype=self.logits_dtype))
            return pb_utils.InferenceResponse(output_tensors=[logits_tensor])
        else:
            pred_tensor = pb_utils.Tensor("predictions", np.array(result['predictions'], dtype=self.predictions_dtype))
            return pb_utils.InferenceResponse(output_tensors=[pred_tensor])
    
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Auto-complete Triton config for token classification"""
        inputs = [{"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]}]
        outputs = [{"name": "predictions", "data_type": "TYPE_INT32", "dims": [-1]}]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_model_transaction_policy(dict(decoupled=False))
        auto_complete_model_config.set_max_batch_size(0)

        return auto_complete_model_config