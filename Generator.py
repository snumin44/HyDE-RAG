class Generator():
    """
    Class for Generation Model, which is one of the HyDEQ Pipe Line. 
    """
    def __init__(self, model,
                 tokenizer,
                 max_new_tokens=128,
                 temperature=1,
                 top_k=50,
                 top_p=1,
                 do_sample=True,
                 wait_till_success=True
                ):
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.wait_till_success = wait_till_success

    def generate(self, prompt, n):
        texts = []
        for _ in range(n):
            self.model.eval()
            result = self.model.generate(
                **tokenizer(
                    prompt,
                    return_tensors='pt',
                    return_token_type_ids=False
                           ),
                pad_token_id= self.tokenizer.eos_token_id,
                max_length=self.max_new_tokens,
                temperature=self.temperature,
                top_k = self.top_k,
                top_p = self.top_p,
                do_sample=self.do_sample)
            text = tokenizer.decode(result[0])
            texts.append(text)
        return texts