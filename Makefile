# Makefile targets for Sphinx documentation (all targets prefixed with 'docs-')

.PHONY: docs-html docs-clean docs-live docs-env docs-publish \
        docs-html-internal docs-html-ga docs-html-ea docs-html-draft \
        docs-live-internal docs-live-ga docs-live-ea docs-live-draft \
        docs-publish-internal docs-publish-ga docs-publish-ea docs-publish-draft \
        docs-pinecone-test docs-pinecone-upload docs-pinecone-upload-dry docs-pinecone-update

# Usage:
#   make docs-html DOCS_ENV=internal   # Build docs for internal use
#   make docs-html DOCS_ENV=ga         # Build docs for GA
#   make docs-html                     # Build docs with no special tag
#   make docs-live DOCS_ENV=draft      # Live server with draft tag
#   make docs-publish DOCS_ENV=ga      # Production build (fails on warnings)
#   make docs-pinecone-update          # Build docs and update Pinecone index
#   make docs-pinecone-test            # Test Pinecone connection and setup

DOCS_ENV ?=

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    VENV_PYTHON = .venv-docs/Scripts/python.exe
    VENV_ACTIVATE = .venv-docs\Scripts\activate
    VENV_ACTIVATE_PS = .venv-docs\Scripts\Activate.ps1
    RM_CMD = if exist docs\_build rmdir /s /q docs\_build
    ECHO_BLANK = @echo.
else
    VENV_PYTHON = .venv-docs/bin/python
    VENV_ACTIVATE = source .venv-docs/bin/activate
    RM_CMD = cd docs && rm -rf _build
    ECHO_BLANK = @echo ""
endif

# Pass DOCS_ENV to sphinx-build if set

# Cross-platform uv run command
ifeq ($(OS),Windows_NT)
    UV_RUN = uv run --active
else
    UV_RUN = VIRTUAL_ENV=../.venv-docs uv run
endif

# Makefile targets for Sphinx documentation (all targets prefixed with 'docs-')

.PHONY: docs-html docs-clean docs-live docs-env


docs-html:
	@echo "Building HTML documentation..."
	cd docs && $(UV_RUN) python -m sphinx -b html $(if $(DOCS_ENV),-t $(DOCS_ENV)) . _build/html

docs-publish:
	@echo "Building HTML documentation for publication (fail on warnings)..."
	cd docs && $(UV_RUN) python -m sphinx --fail-on-warning --builder html $(if $(DOCS_ENV),-t $(DOCS_ENV)) . _build/html

docs-clean:
	@echo "Cleaning built documentation..."
	$(RM_CMD)

docs-live:
	@echo "Starting live-reload server (sphinx-autobuild)..."
	cd docs && $(UV_RUN) python -m sphinx_autobuild $(if $(DOCS_ENV),-t $(DOCS_ENV)) . _build/html

docs-env:
	@echo "Setting up docs virtual environment with uv..."
ifeq ($(OS),Windows_NT)
	@where uv >nul 2>&1 || ( \
		echo. && \
		echo ❌ uv is not installed or not in PATH && \
		echo. && \
		echo uv is a fast Python package installer and resolver. && \
		echo Please install it using one of the following methods: && \
		echo. && \
		echo 🪟 Windows PowerShell: && \
		echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex" && \
		echo. && \
		echo 📦 pip: && \
		echo   pip install uv && \
		echo. && \
		echo 🍺 Scoop: && \
		echo   scoop install uv && \
		echo. && \
		echo After installation, you may need to: && \
		echo 1. Restart your terminal && \
		echo 2. Or add uv to your PATH manually && \
		echo 3. Then run 'make docs-env' again && \
		echo. && \
		echo For more installation options, visit: https://docs.astral.sh/uv/getting-started/installation/ && \
		exit 1 \
	)
else
	@command -v uv >/dev/null 2>&1 || ( \
		echo ""; \
		echo "❌ uv is not installed or not in PATH"; \
		echo ""; \
		echo "uv is a fast Python package installer and resolver."; \
		echo "Please install it using one of the following methods:"; \
		echo ""; \
		echo "🐧 Linux/Unix:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo ""; \
		echo "🍺 Homebrew (macOS):"; \
		echo "  brew install uv"; \
		echo ""; \
		echo "📦 pip:"; \
		echo "  pip install uv"; \
		echo ""; \
		echo "After installation, you may need to:"; \
		echo "1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"; \
		echo "2. Or add uv to your PATH manually"; \
		echo "3. Then run 'make docs-env' again"; \
		echo ""; \
		echo "For more installation options, visit: https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	)
endif
	@echo "✅ uv found, creating virtual environment..."
	uv venv .venv-docs
	uv pip install -r requirements-docs.txt --python .venv-docs
	$(ECHO_BLANK)
	@echo "✅ Documentation environment setup complete!"
	$(ECHO_BLANK)
	@echo "📝 Note: The environment is NOT automatically activated."
	@echo "To manually activate the docs environment, run:"
ifeq ($(OS),Windows_NT)
	@echo "  For Command Prompt: $(VENV_ACTIVATE)"
	@echo "  For PowerShell: $(VENV_ACTIVATE_PS)"
else
	@echo "  $(VENV_ACTIVATE)"
endif
	$(ECHO_BLANK)
	@echo "Once activated, you can run other docs commands like:"
	@echo "  make docs-html    # Build HTML documentation"
	@echo "  make docs-live    # Start live-reload server"

# HTML build shortcuts

docs-html-internal:
	$(MAKE) docs-html DOCS_ENV=internal

docs-html-ga:
	$(MAKE) docs-html DOCS_ENV=ga

docs-html-ea:
	$(MAKE) docs-html DOCS_ENV=ea

docs-html-draft:
	$(MAKE) docs-html DOCS_ENV=draft

# Publish build shortcuts

docs-publish-internal:
	$(MAKE) docs-publish DOCS_ENV=internal

docs-publish-ga:
	$(MAKE) docs-publish DOCS_ENV=ga

docs-publish-ea:
	$(MAKE) docs-publish DOCS_ENV=ea

docs-publish-draft:
	$(MAKE) docs-publish DOCS_ENV=draft

# Live server shortcuts

docs-live-internal:
	$(MAKE) docs-live DOCS_ENV=internal

docs-live-ga:
	$(MAKE) docs-live DOCS_ENV=ga

docs-live-ea:
	$(MAKE) docs-live DOCS_ENV=ea

docs-live-draft:
	$(MAKE) docs-live DOCS_ENV=draft

# Pinecone index management targets

docs-pinecone-test:
	@echo "Testing Pinecone connection and setup..."
	$(VENV_PYTHON) scripts/test_pinecone_setup.py

docs-pinecone-upload-dry:
	@echo "Dry run: Testing Pinecone upload without sending data..."
	$(VENV_PYTHON) scripts/send_to_pinecone_llama.py --dry-run

docs-pinecone-upload:
	@echo "Uploading documentation to Pinecone..."
	$(VENV_PYTHON) scripts/send_to_pinecone_simple.py $(PINECONE_ARGS)

docs-pinecone-update: docs-html
	@echo "Building docs and updating Pinecone index..."
	$(VENV_PYTHON) scripts/send_to_pinecone_simple.py $(PINECONE_ARGS)
