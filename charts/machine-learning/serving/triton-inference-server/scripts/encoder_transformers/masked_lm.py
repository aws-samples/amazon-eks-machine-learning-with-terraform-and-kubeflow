import numpy as np
import torch
from transformers import AutoModelForMaskedLM
import triton_python_backend_utils as pb_utils
from encoder_base import EncoderBaseModel

class TritonPythonModel(EncoderBaseModel):
    """Masked Language Model implementation (Fixed for EncoderBaseModel)"""
    
    def _init_output_config(self):
        """Initialize output tensor configuration for masked LM"""
        logits_config = pb_utils.get_output_config_by_name(self.model_config, "logits")
        self.logits_dtype = pb_utils.triton_string_to_numpy(logits_config["data_type"])
    
    def _load_model(self, model_location):
        """Load masked language model"""
        self.logger.log_info(f"Loading AutoModelForMaskedLM from {model_location}")
        return AutoModelForMaskedLM.from_pretrained(model_location)
    
    def _process_outputs(self, model_output, attention_mask):
        """Process masked LM logits - return per-token predictions"""
        # Get logits and move to CPU
        logits = model_output.logits.detach().cpu()
        
        # Calculate original lengths from attention mask [Batch, SeqLen]
        original_lengths = attention_mask.sum(dim=1).tolist()
        
        results = []
        for i in range(len(original_lengths)):
            seq_len = int(original_lengths[i])
            # Slice to remove padding tokens: [seq_len, vocab_size]
            # We use .copy() to ensure the memory is contiguous for numpy
            results.append(logits[i, :seq_len].numpy().astype(self.logits_dtype))
        
        return results
    
    def _create_response(self, request_results):
        """
        Create Triton response from a slice of results.
        request_results: List of numpy arrays, each [seq_len, vocab_size]
        """
        # Because sequence lengths vary, we cannot pack them into a 
        # single 3D numpy array [Batch, Seq, Vocab] easily unless they are 
        # the same length. For MLMs, we usually return them as a single 
        # response, but since sequence lengths differ, we must handle 
        # them as a list of tensors or pad them here.
        
        # Standard approach for Triton with variable sequences in one request:
        # If the client sent shape [2, 1], they expect a batch output.
        # We must pad these results to the max length in this specific request.
        max_seq_len = max(r.shape[0] for r in request_results)
        vocab_size = request_results[0].shape[1]
        
        batch_size = len(request_results)
        padded_logits = np.zeros((batch_size, max_seq_len, vocab_size), dtype=self.logits_dtype)
        
        for i, r in enumerate(request_results):
            padded_logits[i, :r.shape[0], :] = r
            
        output_tensor = pb_utils.Tensor("logits", padded_logits)
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])
    
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """Auto-complete Triton config for masked LM"""
        inputs = [{"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]}]
        # Dimensions are [SequenceLength, VocabSize]
        outputs = [{"name": "logits", "data_type": "TYPE_FP32", "dims": [-1, -1]}]

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
        # Keep max_batch_size at 0 for auto-complete, 
        # but your actual config.pbtxt should use > 0
        return auto_complete_model_config