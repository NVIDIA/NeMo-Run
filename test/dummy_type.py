from nemo_run.cli.cli_parser import ParseError, type_parser
from typing import Type, Optional, Any
import nemo_run
from nemo_run.config import Config

# Mock MegatronTokenizer base class
class MegatronTokenizer:
    def __init__(self, *args, **kwargs):
        pass

# Mock NullTokenizer that matches the original definition
class NullTokenizer(MegatronTokenizer):
    def __init__(self, vocab_size: int):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

    def tokenize(self, text: str) -> list[int]:
        return [int(x) for x in text.split(' ')]

    def detokenize(self, ids: list[int]) -> str:
        text = [str(x) for x in ids]
        return ' '.join(text)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    @property
    def vocab_size(self) -> int:
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self) -> dict:
        raise NotImplementedError

    @property
    def inv_vocab(self) -> dict:
        raise NotImplementedError

    @property
    def cls(self) -> int:
        return -1

    @property
    def sep(self) -> int:
        return -1

    @property
    def mask(self) -> int:
        return -1

    @property
    def eod(self) -> int:
        return self._eod_id

    @property
    def additional_special_tokens_ids(self) -> Optional[list[int]]:
        return None

class RealType:
    def __init__(self, value: int = 42):
        self.value = value

# Register parser for NullTokenizer
def parse_null_tokenizer(value: str, annotation: Type) -> NullTokenizer:
    """Parse a string into a NullTokenizer instance.
    
    Args:
        value: String in format 'null_tokenizer(vocab_size=100)' or 'null_tokenizer'
        annotation: Expected type (should be NullTokenizer)
    
    Returns:
        NullTokenizer instance with specified vocab_size
        
    Raises:
        ParseError if the format is invalid
    """
    import re
    
    # Handle factory function calls
    if value.startswith("null_tokenizer"):
        # Handle with parameters
        match = re.match(r"null_tokenizer\(vocab_size=(\d+)\)", value)
        if match:
            vocab_size = int(match.group(1))
            return NullTokenizer(vocab_size=vocab_size)
        
        # Handle without parameters
        if value == "null_tokenizer" or value == "null_tokenizer()":
            return NullTokenizer(vocab_size=256000)
    
    raise ParseError(value, NullTokenizer, f"Invalid NullTokenizer format: {value}")

# Register the parser
type_parser.register_parser(NullTokenizer)(parse_null_tokenizer)

@nemo_run.cli.factory(target=NullTokenizer)
@nemo_run.autoconvert
def null_tokenizer(vocab_size: int = 256000) -> Config[NullTokenizer]:
    """Factory function to create a NullTokenizer configuration.
    
    Args:
        vocab_size: Size of the vocabulary (default: 256000)
        
    Returns:
        Config[NullTokenizer] with the specified vocabulary size
    """
    return get_nmt_tokenizer(library='null', vocab_size=vocab_size)

def get_nmt_tokenizer(
    library: str = 'sentencepiece',
    model_name: Optional[str] = None,
    vocab_size: Optional[int] = None,
    **kwargs: Any
) -> Config[NullTokenizer]:
    """Mock version of get_nmt_tokenizer that only handles the null tokenizer case.
    
    Args:
        library: Must be 'null' for this mock implementation
        model_name: Not used in mock implementation
        vocab_size: Size of the vocabulary (required for null tokenizer)
        **kwargs: Additional arguments (not used in mock implementation)
        
    Returns:
        Config[NullTokenizer] with the specified vocabulary size
        
    Raises:
        ValueError if library is not 'null' or vocab_size is not specified
    """
    if library != 'null':
        raise ValueError(f"Mock get_nmt_tokenizer only supports library='null', got {library}")
    if vocab_size is None:
        raise ValueError("vocab_size must be specified for null tokenizer")
    return Config(NullTokenizer, vocab_size=vocab_size)
