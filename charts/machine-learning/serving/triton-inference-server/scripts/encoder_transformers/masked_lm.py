import numpy as np
from transformers import AutoModelForMaskedLM
import triton_python_backend_utils as pb_utils
from encoder_base import EncoderBaseModel


class TritonPythonModel(EncoderBaseModel):
    """Masked Language Model implementation using EncoderBaseModel"""
    
    def _init_output_config(self):
        """Initialize output tensor configuration for masked LM"""
        logits_config = pb_utils.get_output_config_by_name(self.model_config, "logits")
        self.logits_dtype = pb_utils.triton_string_to_numpy(logits_config["data_type"])
    
    def _load_model(self, model_location):
        """Load masked language model"""
        self.logger.log_info(f"Loading AutoModelForMaskedLM from {model_location}")
        return AutoModelForMaskedLM.from_pretrained(model_location)
    
    def _process_outputs(self, model_output, original_lengths):
        """Process masked LM logits - return per-token predictions
        
        Args:
            model_output: Model output with .logits attribute
            original_lengths: List of original sequence lengths (before padding)
            
        Returns:
            List of logits arrays, one per input, sliced to original length
        """
        # Get logits and move to CPU
        logits = model_output.logits.detach().cpu()
        
        # Extract only the original batch and sequence lengths
        results = []
        for i in range(len(original_lengths)):
            seq_len = original_lengths[i]
            # Convert to numpy then to list for Triton
            # Shape: [seq_len, vocab_size]
            results.append(logits[i, :seq_len].numpy().astype(self.logits_dtype).tolist())
        
        return results
    
    def _create_response(self, result):
        """Create Triton response from processed logits
        
        Args:
            result: List of logits for single input [seq_len, vocab_size]
            
        Returns:
            pb_utils.InferenceResponse with logits tensor
        """
        output_tensor = pb_utils.Tensor("logits", np.array(result, dtype=self.logits_dtype))
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])
    
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Auto-complete Triton config for masked LM"""
        inputs = [{"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]}]
        outputs = [{"name": "logits", "data_type": "TYPE_FP32", "dims": [-1]}]

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