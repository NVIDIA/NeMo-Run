"""Utility functions for JSON output."""

import fnmatch
from typing import Dict, Any
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

def get_setting(config, key: str, default=None):
    """Get a setting from json_output_settings with fallback to old config names."""
    settings = getattr(config, 'json_output_settings', {})
    
    # Try new settings format first
    if key in settings:
        return settings[key]
    
    # Fallback to old config names for backward compatibility
    old_config_map = {
        'enabled': 'json_output_enabled',
        'exclude_patterns': 'json_output_exclude_patterns', 
        'verbose': 'json_output_verbose',
        'parallel': 'json_output_parallel',
        'include_children': 'json_output_include_children',
        'include_child_content': 'json_output_include_child_content',
        'main_index_mode': 'json_output_main_index_mode',
        'max_main_index_docs': 'json_output_max_main_index_docs',
    }
    
    old_key = old_config_map.get(key)
    if old_key and hasattr(config, old_key):
        return getattr(config, old_key)
    
    return default

def is_content_gated(config: Any, docname: str) -> bool:
    """
    Check if a document is content gated by checking Sphinx's exclude_patterns.
    This works with the content gating extension that adds restricted documents
    to exclude_patterns during config-inited event.
    """
    sphinx_exclude_patterns = getattr(config, 'exclude_patterns', [])
    if not sphinx_exclude_patterns:
        return False
    
    # Convert docname to potential file paths that might be in exclude_patterns
    possible_paths = [docname + '.md', docname + '.rst', docname]
    
    for possible_path in possible_paths:
        # Check if this path matches any exclude pattern using fnmatch (supports glob patterns)
        for pattern in sphinx_exclude_patterns:
            if isinstance(pattern, str):
                if fnmatch.fnmatch(possible_path, pattern):
                    return True
    
    return False

def should_generate_json(config: Any, docname: str) -> bool:
    """Check if JSON should be generated for this document."""
    if not get_setting(config, 'enabled', True):
        return False
    
    if not docname or not isinstance(docname, str):
        logger.warning(f"Invalid docname for JSON generation: {docname}")
        return False
    
    # CRITICAL: Check content gating first - if document is content gated, don't generate JSON
    if is_content_gated(config, docname):
        logger.debug(f"Excluding {docname} from JSON generation due to content gating")
        return False
    
    # Check JSON output extension's own exclude patterns
    for pattern in get_setting(config, 'exclude_patterns', []):
        if isinstance(pattern, str) and docname.startswith(pattern):
            return False
    
    return True

def get_document_url(app: Sphinx, docname: str) -> str:
    """Get the URL for a document."""
    if not docname or not isinstance(docname, str):
        logger.warning(f"Invalid docname for URL generation: {docname}")
        return 'invalid.html'
        
    try:
        if hasattr(app.builder, 'get_target_uri'):
            return app.builder.get_target_uri(docname)
    except Exception as e:
        logger.warning(f"Failed to get target URI for {docname}: {e}")
    
    return docname + '.html' 