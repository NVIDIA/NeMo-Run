(admin-integrations-pinecone)=
# Pinecone Search Integration

Upload your Sphinx documentation to Pinecone for semantic search capabilities using hosted embeddings.

## Background

This integration uses **Pinecone's hosted embeddings** with the `llama-text-embed-v2` model to automatically generate high-quality 1024-dimensional embeddings from your documentation content. The system processes your documentation's `index.json` file (generated during the build process) and uploads structured content to Pinecone for semantic search.

## Prerequisites

Before setting up the Pinecone integration, ensure you have:

1. **Pinecone Account**: Active account at [pinecone.io](https://pinecone.io)
2. **Pinecone Index**: Configured with the following specifications:
   - **Dimensions**: 1024
   - **Metric**: cosine  
   - **Model**: llama-text-embed-v2 (hosted)
3. **Environment Variable**: `PINECONE_API_KEY` set in your environment
4. **Documentation Build**: Your Sphinx documentation built with the index extension

## Quick Start

### 1. Test Your Setup

First, validate your Pinecone connection and configuration:

```bash
python scripts/test_pinecone_setup.py
```

This script verifies:
- Pinecone API connection
- Index configuration (dimensions, hosted embeddings)
- Documentation index file availability
- Overall setup readiness

### 2. Preview Upload (Dry Run)

Before uploading, preview what will be sent to Pinecone:

```bash
python scripts/send_to_pinecone_simple.py --dry-run --namespace docs-content
```

### 3. Upload Documentation

Upload your documentation to Pinecone:

```bash
python scripts/send_to_pinecone_simple.py --namespace docs-content
```

### 4. Test Search Functionality

Once uploaded, test semantic search:

```bash
python scripts/query_pinecone_example.py
```

### 5. Using Make Commands

For streamlined workflows, use the provided Make targets:

```bash
# Test Pinecone connection
make docs-pinecone-test

# Build documentation and upload to Pinecone
make docs-pinecone-update PINECONE_ARGS="--namespace docs-content"

# Upload only (without rebuilding docs)
make docs-pinecone-upload PINECONE_ARGS="--namespace docs-content"

# Preview mode
make docs-pinecone-upload-dry PINECONE_ARGS="--namespace docs-content"
```

## How It Works

### Document Processing Pipeline

1. **Documentation Build**: Sphinx generates comprehensive `index.json` with content and metadata
2. **Content Extraction**: Script processes the JSON structure and extracts text content
3. **Pinecone Upload**: Documents sent to Pinecone with metadata using `upsert_records()`
4. **Hosted Embeddings**: Pinecone automatically generates 1024-dimensional embeddings using `llama-text-embed-v2`
5. **Storage**: Vectors stored in your specified namespace for semantic search

### Key Features

- **Hosted Embeddings**: No local model downloads or GPU requirements. Pinecone handles embedding generation automatically.
- **Optimal Dimensions**: Perfect 1024-dimensional vectors match your index configuration without compatibility issues.
- **Fast Processing**: Batch uploads and efficient processing handle large documentation sets quickly.
- **Namespace Organization**: Documents organized by namespace for logical separation and management.

## Searching Your Documentation

### Basic Search Example

```python
import os
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("docs-site-demo-starter-kit")

# Search using hosted embeddings
results = index.search(
    namespace="docs-content",
    query={
        "inputs": {"text": "How do I integrate with Spark?"},
        "top_k": 5
    }
)

# Process results
for hit in results.result.hits:
    print(f"Score: {hit['_score']:.4f}")
    print(f"Title: {hit['fields']['title']}")
    print(f"URL: {hit['fields']['url']}")
    print(f"Summary: {hit['fields'].get('summary', '')[:200]}...")
    print()
```

### Search Response Format

The hosted embeddings API returns results in this format:

```python
{
    'result': {
        'hits': [
            {
                '_id': 'document-identifier',
                '_score': 0.8234,  # Relevance score (0-1)
                'fields': {
                    'title': 'Document Title',
                    'url': 'path/to/document.html',
                    'format': 'text',
                    'content': 'Full document content...',
                    'summary': 'Document summary...',
                    'headings': 'Section headings...'
                }
            }
        ]
    },
    'usage': {
        'embed_total_tokens': 5,
        'read_units': 6
    }
}
```

### Advanced Search Function

```python
def search_docs(query: str, top_k: int = 5, namespace: str = "docs-content"):
    """
    Search documentation with error handling and clean output.
    """
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("docs-site-demo-starter-kit")
    
    try:
        results = index.search(
            namespace=namespace,
            query={
                "inputs": {"text": query},
                "top_k": top_k
            }
        )
        
        # Extract and format results
        search_results = []
        for hit in results.result.hits:
            fields = hit.get('fields', {})
            search_results.append({
                'score': hit.get('_score', 0),
                'title': fields.get('title', 'N/A'),
                'url': fields.get('url', 'N/A'),
                'summary': fields.get('summary', '')[:200],
                'content_preview': fields.get('content', '')[:150]
            })
        
        return search_results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

# Usage
results = search_docs("Apache Spark integration", top_k=3)
for result in results:
    print(f"📄 {result['title']} (Score: {result['score']:.4f})")
    print(f"   {result['summary']}")
    print()
```

## Configuration

### Pinecone Index Setup

Ensure your Pinecone index meets these requirements:

| Setting | Value |
|---------|-------|
| Dimensions | 1024 (matches llama-text-embed-v2) |
| Metric | cosine |
| Model | llama-text-embed-v2 (hosted) |
| Type | Serverless (recommended) |

### Environment Configuration

Set your Pinecone API key as an environment variable:

```bash
export PINECONE_API_KEY="your-api-key-here"
```

For persistent configuration, add to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
echo 'export PINECONE_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Document Metadata Structure

Each document uploaded to Pinecone includes structured metadata accessible in search results:

```json
{
  "title": "Document Title",
  "url": "path/to/document.html", 
  "format": "text",
  "content": "Full document text content...",
  "summary": "Brief document summary...",
  "headings": "Section | Subsection | Topic",
  "last_modified": "2024-01-15",
  "author": "Author Name",
  "tags": "tag1, tag2",
  "categories": "category",
  "description": "Document description"
}
```

## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_pinecone_setup.py` | Validate connection and configuration | `python scripts/test_pinecone_setup.py` |
| `send_to_pinecone_simple.py` | Upload documentation to Pinecone | `python scripts/send_to_pinecone_simple.py --namespace docs-content` |
| `query_pinecone_example.py` | Test search functionality | `python scripts/query_pinecone_example.py` |

## Command Reference

### Test Script Options

```bash
python scripts/test_pinecone_setup.py [OPTIONS]
```

```{list-table} Test Script Options
:widths: 25 25 50
:header-rows: 1

* - Option
  - Default
  - Description
* - `--index-name`
  - `docs-site-demo-starter-kit`
  - Pinecone index name
* - `--index-file`
  - `docs/_build/html/index.json`
  - Path to documentation index file
```

### Upload Script Options

```bash
python scripts/send_to_pinecone_simple.py [OPTIONS]
```

```{list-table} Upload Script Options
:widths: 25 25 50
:header-rows: 1

* - Option
  - Default
  - Description
* - `--index-file`
  - `docs/_build/html/index.json`
  - Path to documentation index file
* - `--index-name`
  - `docs-site-demo-starter-kit`
  - Pinecone index name
* - `--namespace`
  - Required
  - Pinecone namespace for documents
* - `--dry-run`
  - `false`
  - Preview without uploading
* - `--batch-size`
  - `50`
  - Documents per batch upload
```

### Make Targets

```{list-table} Available Make Targets  
:widths: 35 65
:header-rows: 1

* - Target
  - Description
* - `docs-pinecone-test`
  - Test Pinecone connection and configuration
* - `docs-pinecone-update`
  - Build documentation and upload to Pinecone
* - `docs-pinecone-upload`
  - Upload to Pinecone (no documentation build)
* - `docs-pinecone-upload-dry`
  - Preview upload without sending to Pinecone
```

## Troubleshooting

### Common Issues

**Index Not Found**

Error: `Index 'your-index-name' not found`

Solutions:
- Verify index name in script matches Pinecone console
- Check `PINECONE_API_KEY` environment variable
- Confirm index exists in your Pinecone project

**Dimension Mismatch**

Error: `Dimension mismatch: expected 1024, got XXX`

Solutions:
- Ensure index configured for 1024 dimensions
- Verify hosted embeddings (llama-text-embed-v2) enabled
- Recreate index with correct dimensions if needed

**API Key Invalid**

Error: `Authentication failed` or `Invalid API key`

Solutions:
- Verify `PINECONE_API_KEY` environment variable set
- Check API key in Pinecone console
- Ensure key has appropriate permissions

**Index File Missing**

Error: `Index file not found: docs/_build/html/index.json`

Solutions:
- Build documentation first: `make docs-html`
- Verify index extension enabled in Sphinx configuration
- Check file path and permissions

**No Search Results**

Error: Search returns empty results despite having uploaded documents

Solutions:
- Verify correct namespace in search query
- Check that documents were uploaded successfully
- Try broader search terms
- Confirm namespace exists: `python scripts/test_pinecone_setup.py`

### Debug Commands

For detailed troubleshooting, use these diagnostic commands:

```bash
# Full connection and configuration test
python scripts/test_pinecone_setup.py --index-name your-index-name

# Preview upload with custom settings
python scripts/send_to_pinecone_simple.py --dry-run --namespace test --batch-size 10

# Test search functionality
python scripts/query_pinecone_example.py

# Check environment variables
echo $PINECONE_API_KEY

# Verify index file exists
ls -la docs/_build/html/index.json
```

## Performance and Monitoring

### Upload Performance

Typical performance metrics for the integration:

- **Upload Speed**: 30-50 documents per second
- **Batch Processing**: Efficient handling of large documentation sets
- **No Local Compute**: All embedding generation happens on Pinecone servers
- **Automatic Retries**: Built-in error handling and retry logic

### Search Performance

- **Query Speed**: Sub-second response times for most queries
- **Hosted Embeddings**: No local embedding computation required
- **Scalable**: Handles concurrent searches efficiently
- **Usage Tracking**: Monitor embedding tokens and read units

### Success Indicators

When the integration works correctly, you should see:

- ✅ Connection test passes without errors
- ✅ Index dimensions confirmed as 1024  
- ✅ Documents upload with zero failures
- ✅ Namespace appears in Pinecone console
- ✅ Vector count increases in index statistics
- ✅ Search queries return relevant results with good scores (>0.2)

### Monitoring Uploads

The upload script provides detailed progress information:

```
🔗 Testing Pinecone connection...
✅ Connected to index: docs-site-demo-starter-kit
📊 Index stats:
   - Total vectors: 37
   - Dimension: 1024
   - Namespaces:
     - docs-content: 37 vectors
     
🚀 Starting upload to Pinecone...
📄 Found 37 documents to upload
⏳ Processing batch 1/1 (37 documents)
✅ Successfully uploaded 37 documents (0 failures)
🎉 Upload completed successfully!
```

## Integration with Documentation Workflow

### CI/CD Integration

Add Pinecone upload to your documentation deployment pipeline:

```yaml
# Example GitHub Actions workflow
- name: Build Documentation
  run: make docs-html

- name: Upload to Pinecone  
  run: make docs-pinecone-upload PINECONE_ARGS="--namespace docs-content"
  env:
    PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
```

### Website Integration

Add search to your documentation website:

```html
<!-- Search interface -->
<div id="search-container">
    <input type="text" id="search-input" placeholder="Search documentation...">
    <div id="search-results"></div>
</div>

<script>
async function searchDocs(query) {
    // Call your backend API that uses the Pinecone search
    const response = await fetch('/api/search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query, top_k: 5})
    });
    
    const results = await response.json();
    displayResults(results);
}

function displayResults(results) {
    const container = document.getElementById('search-results');
    container.innerHTML = results.map(result => `
        <div class="search-result">
            <h3><a href="${result.url}">${result.title}</a></h3>
            <p>${result.summary}</p>
            <small>Relevance: ${(result.score * 100).toFixed(1)}%</small>
        </div>
    `).join('');
}
</script>
```

### Automated Updates

For regularly updated documentation, set up automated uploads:

```bash
# Example cron job - update search index every 6 hours
0 */6 * * * cd /path/to/docs && make docs-pinecone-update PINECONE_ARGS="--namespace docs-content"
```

This integration provides a seamless way to enable semantic search capabilities for your documentation, leveraging Pinecone's advanced embedding models and vector search infrastructure. 