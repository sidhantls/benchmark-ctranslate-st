import transformers
from tqdm import tqdm 
import torch 
import time
import numpy as np 

def run_speed_test(model, tokenizer, num_docs=100, max_length=256, is_ctranslate=False):
    sample_text = "This is a sample document for speed testing." * 100
    
    # Tokenize the input document once

    if not is_ctranslate:
        inputs = tokenizer(sample_text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    else:
        inputs = tokenizer(sample_text, padding=True, truncation=True, max_length=max_length)
    
    times = []
    
    for _ in tqdm(range(num_docs)):
        start_time = time.time()
        
        if not is_ctranslate:
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
        else:
            outputs = model.forward_batch([inputs['input_ids']])
            outputs = np.array(outputs.last_hidden_state)
        
        # Post-processing
        # sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        # normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        times.append(time.time() - start_time)
    
    return np.mean(times), np.median(times), np.std(times)
