from nemo_run.cli.cli_parser import ParseError

class RealType:
    def __init__(self, value=42):
        self.value = value

class TokenizerSpec:
    def __init__(self, hidden=1000):
        self.hidden = hidden

# Register parser for TokenizerSpec
from nemo_run.cli.cli_parser import type_parser
from typing import Type

@type_parser.register_parser(TokenizerSpec)
def parse_tokenizer_spec(value: str, annotation: Type) -> TokenizerSpec:
    # Parse the value in the format TokenizerSpec(hidden=100) or factory function calls
    import re
    
    # Handle factory function calls
    if value.startswith("null_tokenizer"):
        match = re.match(r"null_tokenizer\(vocab_size=(\d+)\)", value)
        if match:
            vocab_size = int(match.group(1))
            return TokenizerSpec(hidden=vocab_size)
        return TokenizerSpec()  # default values
        
    # Handle direct TokenizerSpec instantiation
    match = re.match(r"TokenizerSpec\(hidden=(\d+)\)", value)
    if match:
        hidden = int(match.group(1))
        return TokenizerSpec(hidden=hidden)
    
    raise ParseError(value, TokenizerSpec, "Invalid TokenizerSpec format")
