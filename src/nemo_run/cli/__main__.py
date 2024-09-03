"""Main entry point for the nemo_run CLI."""

from nemo_run.cli.api import create_cli


def main():
    """Create and run the CLI application."""
    app = create_cli()
    app()


if __name__ == "__main__":
    main()