from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InsightsGenerator:
    def __init__(self, model_name="distilgpt2"):
        """
        Initialize the InsightsGenerator with a pre-trained model and tokenizer.
        Default model is distilgpt2.
        """
        # Load pre-trained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set the padding token to eos_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token as pad_token

    def generate_insights(self, input_text, max_length=150, num_return_sequences=1):
        """
        Generate insights based on the provided input text.

        Args:
            input_text (str): The input text to generate insights from.
            max_length (int): Maximum length of the generated text.
            num_return_sequences (int): Number of insights to generate.

        Returns:
            List of str: The generated insights as a list of strings.
        """
        # Tokenize the input text and encode it to tensor format
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        
        # Check if 'attention_mask' is set; if not, set it to all ones with the same shape as input_ids
        inputs['attention_mask'] = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
        inputs['pad_token_id'] = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        # Generate output using the model
        outputs = self.model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],  # Provide the attention mask explicitly
            pad_token_id=inputs['pad_token_id'],     # Set pad_token_id to eos_token_id
            max_length=max_length, 
            num_return_sequences=num_return_sequences,
            do_sample=True,          # Enable sampling
            no_repeat_ngram_size=2,  
            top_k=50,                
            top_p=0.95,              
            temperature=0.7          
        )
        
        # Decode the output for each generated sequence
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        return generated_texts
