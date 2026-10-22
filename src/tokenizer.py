import tiktoken

class TikTokenTokenizer:
    """Wrapper for a fixed tiktoken encoding. It does not train a tokenizer."""
    def __init__(self, encoding_name="cl100k_base"):
        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.eot_id = self.enc.eot_token
        self.vocab_size = self.enc.max_token_value + 1

    def encode(self, text):
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids])
