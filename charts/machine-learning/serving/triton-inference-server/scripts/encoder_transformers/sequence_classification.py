import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
import triton_python_backend_utils as pb_utils
from encoder_base import EncoderBaseModel

class TritonPythonModel(EncoderBaseModel):
    """Sequence Classification implementation (Fixed for EncoderBaseModel)"""
    
    def _init_output_config(self):
        """Initialize output tensor configuration for sequence classification"""
        logits_config = pb_utils.get_output_config_by_name(self.model_config, "logits")
        self.logits_dtype = pb_utils.triton_string_to_numpy(logits_config["data_type"])
    
    def _load_model(self, model_location):
        """Load sequence classification model"""
        self.logger.log_info(f"Loading AutoModelForSequenceClassification from {model_location}")
        return AutoModelForSequenceClassification.from_pretrained(model_location)
    
    def _process_outputs(self, model_output, attention_mask):
        """Process sequence classification logits - one prediction per sequence"""
        
        # 1. Get the logits and move to CPU
        # Shape: [Batch_Padded, Num_Labels]
        logits = model_output.logits.detach().cpu().numpy().astype(self.logits_dtype)
        
        # 2. Return the full numpy array or a list of rows
        # We do NOT slice by input_count here because encoder_base.py 
        # will handle the final [:input_batch_size] slice.
        return [logits[i] for i in range(len(logits))]
    
    def _create_response(self, request_results):
        """Create Triton response from a slice of results
        
        Args:
            request_results: List of numpy arrays [num_labels] for a single request
        """
        # Stack the list of arrays into a single [Batch, Num_Labels] tensor
        output_array = np.stack(request_results)
        
        output_tensor = pb_utils.Tensor("logits", output_array)
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])
    
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Auto-complete Triton config for sequence classification"""
        inputs = [{"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]}]
        # Dimensions are [Num_Labels]
        outputs = [{"name": "logits", "data_type": "TYPE_FP32", "dims": [-1]}]

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