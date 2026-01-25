# IMplementing the new tokenizer with extended vocabulary

class Tokenizer:
    def __init__(self, model_config):
        self.model_config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_base)

    def _add_tokens(self):
        special_tokens = self.model_config.get_special_tokens()

        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_tokenizer(self):
        self._add_tokens()
        return self.tokenizer
