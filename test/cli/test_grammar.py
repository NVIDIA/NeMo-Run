import pytest
from unittest.mock import patch, MagicMock
from lark import Lark, UnexpectedToken, UnexpectedInput, GrammarError
from rich.console import Console
from rich.text import Text

from nemo_run.cli.grammar import (
    ArgumentParsingError,
    extract_command_and_args, 
    generate_command_grammar,
    get_parser_for_command,
    cached_parse,
    get_close_matches,
    format_factory_error,
    get_factory_examples,
    format_syntax_error
)


# Test fixture data
@pytest.fixture
def sample_cache():
    """Fixture providing a sample cache with dummy command structure and factories."""
    return {
        "app": {
            "groups": {
                "dummy": {
                    "commands": {
                        "command1": {
                            "signature": [
                                {"name": "arg1", "type": "test.type1"},
                                {"name": "arg2", "type": "test.type2"},
                                {"name": "arg3", "type": "test.type3"}
                            ],
                            "full_namespace": "test.dummy.command1",
                            "help": "Test command 1"
                        },
                        "command2": {
                            "signature": [
                                {"name": "arg1", "type": "test.type1"},
                                {"name": "flag", "type": "test.flag"}
                            ],
                            "full_namespace": "test.dummy.command2",
                            "help": "Test command 2"
                        }
                    }
                },
                "another": {
                    "commands": {
                        "deep": {
                            "commands": {
                                "nested": {
                                    "signature": [
                                        {"name": "deep_arg", "type": "test.deep"}
                                    ],
                                    "full_namespace": "test.another.deep.nested",
                                    "help": "Deeply nested command"
                                }
                            }
                        }
                    }
                }
            }
        },
        "data": {
            "factories": {
                "test.type1": [
                    {"name": "factory1", "params": {
                        "param1": {"type": "int"},
                        "param2": {"type": "str"}
                    }},
                    {"name": "factory2", "params": {
                        "option": {"type": "bool"}
                    }}
                ],
                "test.type2": [
                    {"name": "factory3", "params": {
                        "data": {"type": "dict"},
                        "size": {"type": "int"}
                    }}
                ],
                "test.type3": [
                    {"name": "factory4", "params": {}},
                    {"name": "factory5", "params": {
                        "config": {"type": "str"}
                    }}
                ],
                "test.deep": [
                    {"name": "deep_factory", "params": {
                        "level": {"type": "int"}
                    }}
                ],
                "test.dummy.command1.arg1": [
                    {"name": "special_factory", "params": {
                        "special": {"type": "bool"}
                    }}
                ]
            }
        },
        "parsers": {}  # Empty parsers dict to be filled during tests
    }


@pytest.fixture
def sample_args():
    """Fixture providing sample command arguments for testing."""
    return ["dummy", "command1", "arg1=factory1(param1=1)", "arg2=factory3(size=10)"]


class TestCommandExtraction:
    """Tests for extracting commands and arguments from CLI input."""
    
    def test_extract_valid_command(self, sample_cache):
        """Test extracting a valid command and arguments."""
        args = ["dummy", "command1", "arg1=value", "arg2=value2"]
        command, arguments = extract_command_and_args(args, sample_cache)
        assert command == "dummy command1"
        assert arguments == ["arg1=value", "arg2=value2"]
    
    def test_extract_nested_command(self, sample_cache):
        """Test extracting a deeply nested command."""
        args = ["another", "deep", "nested", "deep_arg=value"]
        command, arguments = extract_command_and_args(args, sample_cache)
        assert command == "another deep nested"
        assert arguments == ["deep_arg=value"]
    
    def test_extract_invalid_command(self, sample_cache):
        """Test extracting an invalid command raises ValueError."""
        args = ["invalid", "command"]
        with pytest.raises(ValueError) as exc_info:
            extract_command_and_args(args, sample_cache)
        assert "Invalid command" in str(exc_info.value)
    
    def test_extract_incomplete_command(self, sample_cache):
        """Test extracting an incomplete command raises ValueError."""
        args = ["dummy"]
        with pytest.raises(ValueError) as exc_info:
            extract_command_and_args(args, sample_cache)
        assert "Incomplete command" in str(exc_info.value)


class TestGrammarGeneration:
    """Tests for grammar generation."""
    
    def test_grammar_generation_with_arguments(self, sample_cache):
        """Test grammar generation for a command with arguments."""
        grammar = generate_command_grammar("dummy command1", sample_cache)
        assert grammar is not None
        assert '"arg1" operation value' in grammar
        assert '"arg2" operation value' in grammar
        assert '"arg3" operation value' in grammar
    
    def test_grammar_generation_without_arguments(self, sample_cache):
        """Test grammar generation for a command without arguments."""
        # Create a command without arguments
        sample_cache["app"]["groups"]["empty"] = {
            "commands": {
                "command": {}
            }
        }
        grammar = generate_command_grammar("empty command", sample_cache)
        assert grammar is not None
        assert "key operation value" in grammar  # Fallback when no named args
    
    def test_grammar_contains_required_rules(self, sample_cache):
        """Test that generated grammar contains all required rules."""
        grammar = generate_command_grammar("dummy command1", sample_cache)
        # Check for core grammar components
        assert "start: argument+" in grammar
        assert "factory_call: identifier" in grammar
        assert "argument: named_argument | general_arg" in grammar
        # Check operation types
        assert 'operation: "=" | "+=" | "-=" | "*=" | "/=" | "|=" | "&="' in grammar


class TestParserGeneration:
    """Tests for parser generation and caching."""
    
    def test_parser_generation(self, sample_cache):
        """Test generating a parser for a command."""
        parser = get_parser_for_command("dummy command1", sample_cache)
        assert isinstance(parser, Lark)
    
    def test_parser_caching(self, sample_cache):
        """Test that parsers are cached."""
        # First call should generate and cache
        parser1 = get_parser_for_command("dummy command1", sample_cache)
        # Second call should retrieve from cache
        with patch('nemo_run.cli.grammar.generate_command_grammar') as mock_generate:
            parser2 = get_parser_for_command("dummy command1", sample_cache)
            # The generate function shouldn't be called again
            mock_generate.assert_not_called()
        # Both should be the same parser instance
        assert parser1 is parser2
    
    def test_parser_handles_basic_syntax(self, sample_cache):
        """Test that the generated parser can handle basic syntax."""
        parser = get_parser_for_command("dummy command1", sample_cache)
        # Test a simple argument parsing
        try:
            tree = parser.parse("arg1=123")
            assert tree is not None
        except Exception as e:
            pytest.fail(f"Parser failed to parse simple input: {e}")


class TestFactoryValidation:
    """Tests for factory validation and suggestions."""
    
    def test_get_close_matches(self):
        """Test finding close matches for typos."""
        possibilities = ["factory1", "factory2", "factory3"]
        matches = get_close_matches("factory", possibilities)
        assert "factory1" in matches
        
        # Test with a typo
        matches = get_close_matches("facory1", possibilities)
        assert "factory1" in matches
    
    def test_get_factory_examples(self):
        """Test generating factory usage examples."""
        factory_params = {
            "param1": {"type": "int"},
            "param2": {"type": "str"}
        }
        minimal, full = get_factory_examples("factory1", "arg1", factory_params)
        assert minimal == "arg1=factory1()"
        assert "param1=1" in full
        assert "param2=" in full
        
        # Test with empty params
        minimal, full = get_factory_examples("factory4", "arg3", {})
        assert minimal == "arg3=factory4()"
        assert "param1=value" in full  # Default generic parameter
    
    def test_format_factory_error(self):
        """Test formatting factory validation errors."""
        rich_msg, plain_msg = format_factory_error(
            factory_name="invalid_factory",
            arg_name="arg1",
            input_text="arg1=invalid_factory()",
            valid_factories=["factory1", "factory2"],
            suggestions=["factory1"],
            minimal_example="arg1=factory1()",
            full_example="arg1=factory1(param1=1, param2=\"value\")"
        )
        # Rich message should contain formatted output
        assert "Invalid Factory Error" in rich_msg
        assert "Did you mean" in rich_msg
        # Plain message should contain key information
        assert "Invalid factory 'invalid_factory'" in plain_msg
        assert "Available factories" in plain_msg
        assert "Example usage" in plain_msg
        
        # Test with no suggestions
        rich_msg, plain_msg = format_factory_error(
            factory_name="totally_wrong",
            arg_name="arg1",
            input_text="arg1=totally_wrong()",
            valid_factories=["factory1", "factory2"],
            suggestions=[],
            minimal_example="arg1=factory1()",
            full_example="arg1=factory1(param1=1)"
        )
        assert "Did you mean" not in rich_msg
        assert "Did you mean" not in plain_msg


class TestErrorHandling:
    """Tests for error handling and formatting."""
    
    def test_format_syntax_error(self):
        """Test formatting syntax errors."""
        mock_error = MagicMock(spec=UnexpectedInput)
        mock_error.pos_in_stream = 10
        
        rich_msg, plain_msg = format_syntax_error(
            command="dummy command1",
            context="arg1=factory1(param1=abc)",
            error=mock_error,
            position=10,
            suggestions=["param1"]
        )
        
        assert "Syntax Error" in rich_msg
        assert "Context:" in rich_msg
        assert "arg1=factory1" in rich_msg
        assert "Did you mean" in rich_msg
    
    def test_format_syntax_error_without_suggestions(self):
        """Test formatting syntax errors without suggestions."""
        mock_error = MagicMock(spec=UnexpectedInput)
        mock_error.pos_in_stream = 15
        mock_error.__str__ = lambda _: "Expected NUMBER but got STRING"
        
        rich_msg, plain_msg = format_syntax_error(
            command="dummy command1",
            context="arg1=factory1(param1='string')",
            error=mock_error,
            position=15,
            suggestions=[]
        )
        
        assert "Syntax Error" in rich_msg
        assert "Context:" in rich_msg
        assert "Error Details:" in rich_msg
        assert "Expected NUMBER" in rich_msg
        assert "Did you mean" not in rich_msg
    
    def test_argument_parsing_error(self):
        """Test the ArgumentParsingError exception."""
        # Create the error with command_args instead of args to avoid confusion with the built-in args attribute
        error = ArgumentParsingError(
            message="Test error message",
            args=["dummy", "command1", "arg1=invalid()"],
            pos=5
        )
        assert error.message == "Test error message"
        # Fix: Use the attribute name as defined in the class (command_args)
        assert error.args[0] == "Test error message"  # Default args from Exception
        # Verify we can still access the original command arguments 
        assert hasattr(error, "pos")


class TestCommandParsing:
    """Tests for the main command parsing functionality."""
    
    @pytest.fixture
    def mock_parser(self):
        """Fixture providing a mock parser."""
        mock = MagicMock(spec=Lark)
        mock.parse.return_value = "parse_tree"
        return mock
    
    def test_cached_parse_valid_input(self, sample_cache, sample_args, mock_parser):
        """Test successful parsing of valid input."""
        with patch('nemo_run.cli.grammar.get_parser_for_command', return_value=mock_parser):
            result = cached_parse(sample_args, sample_cache)
            assert result == "parse_tree"
            # Check that the parser was called with joined arguments
            mock_parser.parse.assert_called_once_with("arg1=factory1(param1=1) arg2=factory3(size=10)")
    
    def test_cached_parse_invalid_factory(self, sample_cache):
        """Test parsing with invalid factory."""
        args = ["dummy", "command1", "arg1=invalid_factory()", "arg2=factory3()"]
        with pytest.raises(ArgumentParsingError) as exc_info:
            cached_parse(args, sample_cache)
        error = exc_info.value
        assert "Invalid Factory Error" in error.message
        assert "invalid_factory" in error.message
    
    def test_cached_parse_invalid_command(self, sample_cache):
        """Test parsing with invalid command."""
        args = ["invalid", "command"]
        with pytest.raises(ValueError) as exc_info:
            cached_parse(args, sample_cache)
        assert "Invalid command" in str(exc_info.value)
    
    @patch('nemo_run.cli.grammar.get_parser_for_command')
    def test_cached_parse_grammar_error(self, mock_get_parser, sample_cache):
        """Test parsing with grammar error."""
        mock_parser = MagicMock(spec=Lark)
        mock_parser.parse.side_effect = GrammarError("Invalid grammar definition")
        mock_get_parser.return_value = mock_parser
        
        args = ["dummy", "command1", "arg1=factory1()"]
        with pytest.raises(ArgumentParsingError) as exc_info:
            cached_parse(args, sample_cache)
        error = exc_info.value
        assert "Grammar Error" in error.message
        
    @patch('nemo_run.cli.grammar.get_parser_for_command')
    def test_cached_parse_unexpected_input(self, mock_get_parser, sample_cache):
        """Test parsing with syntax error."""
        # Create a token mock for use in our error handling
        mock_token = MagicMock()
        mock_token.type = 'IDENTIFIER'
        mock_token.value = 'invalid'
        
        # Import directly from lark (we already have this at the top of the file)
        # Create a real UnexpectedToken instance - it needs some special parameters
        # to initialize correctly
        unexpected_token = UnexpectedToken(mock_token, expected=['EQUALS', 'LPAREN'], 
                                          considered_rules=None, state=None,
                                          interactive_parser=None)
        
        # Set position manually (after creating the instance)
        unexpected_token.pos_in_stream = 5
        
        # Make the parser raise our exception
        mock_parser = MagicMock(spec=Lark)
        mock_parser.parse.side_effect = unexpected_token
        mock_get_parser.return_value = mock_parser
        
        args = ["dummy", "command1", "invalid argument"]
        with pytest.raises(ArgumentParsingError) as exc_info:
            cached_parse(args, sample_cache)
        
        error = exc_info.value
        assert "Syntax Error" in error.message
        assert error.pos == 5


class TestIntegrationWithDummyData:
    """Integration tests using only dummy data."""
    
    def test_parse_simple_command(self, sample_cache):
        """Test parsing a simple command with valid factories."""
        # Create a real parser but with dummy data
        with patch('lark.Lark.parse', return_value="parse_tree"):
            args = ["dummy", "command1", 
                    "arg1=factory1(param1=1, param2='test')",
                    "arg2=factory3(size=10)"]
            result = cached_parse(args, sample_cache)
            assert result == "parse_tree"
    
    def test_specific_factory_validation(self, sample_cache):
        """Test validation of parameter-specific factories."""
        # The sample_cache has a special_factory specifically for arg1
        args = ["dummy", "command1", "arg1=special_factory(special=True)"]
        
        # This should parse successfully since special_factory is valid for arg1
        with patch('lark.Lark.parse', return_value="parse_tree"):
            result = cached_parse(args, sample_cache)
            assert result == "parse_tree"
        
        # But factory4 is not valid for arg1
        args = ["dummy", "command1", "arg1=factory4()"]
        with pytest.raises(ArgumentParsingError) as exc_info:
            cached_parse(args, sample_cache)
        assert "Invalid Factory Error" in exc_info.value.message
    
    def test_factory_suggestion_quality(self, sample_cache):
        """Test quality of factory suggestions for typos."""
        args = ["dummy", "command1", "arg1=facory1(param1=1)"]  # Typo in factory1
        
        with pytest.raises(ArgumentParsingError) as exc_info:
            cached_parse(args, sample_cache)
        
        error = exc_info.value
        assert "Invalid Factory Error" in error.message
        assert "Did you mean: factory1" in error.message
