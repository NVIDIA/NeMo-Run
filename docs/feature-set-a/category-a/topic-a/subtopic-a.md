(section-category-topic-subtopic-a)=
# Subtopic A


## Tabs

::::{tab-set}

:::{tab-item} cURL
:sync: s-curl

```bash
# Basic API request example
curl -X POST \
  https://api.example.com/v1/process \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "batch_size": 1000,
    "timeout": "30s",
    "enable_logging": true,
    "output_format": "json",
    "data": [
      {"id": 1, "content": "Sample data"},
      {"id": 2, "content": "More sample data"}
    ]
  }'
```

:::


:::{tab-item} Python
:sync: s-python

```python
import requests
import json

# Configuration
api_url = "https://api.example.com/v1/process"
api_key = "YOUR_API_KEY"

# Request headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Request payload
payload = {
    "batch_size": 1000,
    "timeout": "30s",
    "enable_logging": True,
    "output_format": "json",
    "data": [
        {"id": 1, "content": "Sample data"},
        {"id": 2, "content": "More sample data"}
    ]
}

# Make the request
response = requests.post(api_url, headers=headers, json=payload)

# Handle the response
if response.status_code == 200:
    result = response.json()
    print("Success:", result)
else:
    print(f"Error {response.status_code}: {response.text}")
```

:::
::::


### Synced Tabs with Variables

Adding `:sync:` with a matching value enables syncing.

::::{tab-set}

:::{tab-item} cURL
:sync: s-curl

```bash
# Basic API request example
curl -X POST \
  https://api.example.com/v1/process \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "batch_size": 1000,
    "timeout": "30s",
    "enable_logging": true,
    "output_format": "json",
    "data": [
      {"id": 1, "content": "{{ product_name }}"},
      {"id": 2, "content": "{{ version }}"}
    ]
  }'
```

:::


:::{tab-item} Python
:sync: s-python

```python
import requests
import json

# Configuration
api_url = "https://api.example.com/v1/process"
api_key = "YOUR_API_KEY"

# Request headers
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Request payload
payload = {
    "batch_size": 1000,
    "timeout": "30s",
    "enable_logging": True,
    "output_format": "json",
    "data": [
        {"id": 1, "content": "{{ product_name }}"},
        {"id": 2, "content": "{{ product_name }}"}
    ]
}

# Make the request
response = requests.post(api_url, headers=headers, json=payload)

# Handle the response
if response.status_code == 200:
    result = response.json()
    print("Success:", result)
else:
    print(f"Error {response.status_code}: {response.text}")
```

:::
::::

## Demo Parameters

List tables enable you to control the individual `:widths:` of your columns.

```{list-table} Sample Configuration Options
:header-rows: 1
:widths: 20 30 25 25

* - Parameter
  - Description
  - Default Value
  - Valid Options
* - `batch_size`
  - Number of items to process in each batch
  - `1000`
  - Any positive integer
* - `timeout`
  - Maximum time to wait for operation completion
  - `30 seconds`
  - `1s` to `300s`
* - `enable_logging`
  - Whether to enable detailed logging output
  - `true`
  - `true`, `false`
* - `output_format`
  - Format for generated output files
  - `json`
  - `json`, `csv`, `parquet`
```
