(section-category-topic-subtopic-b)=
# Subtopic B

This subtopic demonstrates advanced MyST markdown and Sphinx-design features for technical writers to observe patterns and techniques.

(section-category-topic-subtopic-b-admonitions)=
## Admonitions & Callouts

Use admonitions sparingly to highlight important information without disrupting flow.

:::{note}
This is a general informational note. Use for surprising or unexpected behavior.
:::

:::{tip}
This is a helpful tip. Use to reveal positive software behavior users might not discover.
:::

:::{warning}
This is a warning. Use to identify risk of physical injury or data loss.
:::

:::{important}
This is for critical information that users must know.
:::

:::{seealso}
Check out the {ref}`section-category-topic-subtopic-a` for related examples.
:::

(section-category-topic-subtopic-b-dropdowns)=
## Dropdowns

Use dropdowns for lengthy code blocks or content that might distract from main flow.

:::{dropdown} Configuration Example
:icon: gear

This dropdown contains configuration details that don't interrupt the main narrative.

```yaml
# Complete configuration file
api:
  version: "v2"
  timeout: 30s
  retry_attempts: 3
  
database:
  host: "localhost"
  port: 5432
  name: "production_db"
  
logging:
  level: "INFO"
  format: "json"
  output: "stdout"
```
:::

:::{dropdown} Python Implementation Details
:icon: code-square
:color: primary

Here's the complete implementation with error handling:

```python
import logging
import requests
from typing import Dict, Optional, Any

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def make_request(self, endpoint: str, method: str = 'GET', 
                    data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an API request with proper error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            raise
```
:::

(section-category-topic-subtopic-b-advanced-grids)=
## Advanced Grid Layouts

Showcase different grid configurations and responsive behavior.

### Three-Column Feature Grid

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Security First
:class-header: sd-bg-primary sd-text-white
Security features built into every component with zero-trust architecture.
+++
{bdg-success}`Enterprise Ready` {bdg-info}`SOC 2 Compliant`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` High Performance
:class-header: sd-bg-success sd-text-white
Optimized for speed with sub-millisecond response times at scale.
+++
{bdg-primary}`99.9% Uptime` {bdg-secondary}`Auto-scaling`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Developer Friendly
:class-header: sd-bg-info sd-text-white
Comprehensive APIs, SDKs, and documentation for rapid integration.
+++
{bdg-warning}`OpenAPI 3.0` {bdg-info}`SDK Available`
:::
::::

### Two-Column Comparison Layout

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
:class: sd-border-1 sd-shadow-sm sd-p-3

**Traditional Approach** {octicon}`x-circle;1em;sd-text-danger`

- Manual configuration required
- Limited scalability options
- Complex deployment process
- Higher maintenance overhead
- Vendor lock-in risks

:::

:::{grid-item}
:class: sd-border-1 sd-shadow-lg sd-p-3 sd-bg-light

**Our Solution** {octicon}`check-circle;1em;sd-text-success`

- Zero-configuration deployment
- Automatic horizontal scaling
- One-click deployment anywhere
- Self-healing infrastructure
- Cloud-agnostic architecture

:::
::::

(section-category-topic-subtopic-b-complex-tabs)=
## Complex Tab Sets

Demonstrate various tab configurations with synchronization and different content types.

### Multi-Language Code Examples

::::{tab-set}

:::{tab-item} Python
:sync: example-lang

```python
# Python implementation
import asyncio
import aiohttp

async def fetch_data(session, url):
    async with session.get(url) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        data = await fetch_data(session, "https://api.example.com/data")
        print(f"Received: {data}")

# Run the async function
asyncio.run(main())
```

**Key Features:**
- Asynchronous processing
- Built-in error handling
- Session management

:::

:::{tab-item} Node.js
:sync: example-lang

```javascript
// Node.js implementation
const fetch = require('node-fetch');

async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}

// Usage
fetchData('https://api.example.com/data')
    .then(data => console.log('Received:', data))
    .catch(error => console.error('Error:', error));
```

**Key Features:**
- Promise-based architecture
- Comprehensive error handling
- Modern async/await syntax

:::

:::{tab-item} Go
:sync: example-lang

```go
// Go implementation
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type APIResponse struct {
    Data    interface{} `json:"data"`
    Status  string      `json:"status"`
    Message string      `json:"message"`
}

func fetchData(url string) (*APIResponse, error) {
    client := &http.Client{
        Timeout: 30 * time.Second,
    }
    
    resp, err := client.Get(url)
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()
    
    var result APIResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, fmt.Errorf("decode failed: %w", err)
    }
    
    return &result, nil
}

func main() {
    data, err := fetchData("https://api.example.com/data")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    fmt.Printf("Received: %+v\n", data)
}
```

**Key Features:**
- Strong type safety
- Explicit error handling
- High performance execution

:::
::::

### Platform-Specific Instructions

::::{tab-set}

:::{tab-item} Docker
:sync: platform-deploy

**Step 1: Create Dockerfile**

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

**Step 2: Build and Run**

```bash
docker build -t my-app .
docker run -p 3000:3000 my-app
```

:::

:::{tab-item} Kubernetes
:sync: platform-deploy

**Step 1: Create Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        ports:
        - containerPort: 3000
```

**Step 2: Create Service**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

:::

:::{tab-item} Serverless
:sync: platform-deploy

**Step 1: Configure Function**

```yaml
# serverless.yml
service: my-app

provider:
  name: aws
  runtime: nodejs18.x
  stage: ${opt:stage, 'dev'}
  region: us-east-1

functions:
  api:
    handler: src/handler.main
    events:
      - httpApi:
          path: '/{proxy+}'
          method: ANY
```

**Step 2: Deploy**

```bash
npm install -g serverless
serverless deploy --stage production
```

:::
::::

(section-category-topic-subtopic-b-advanced-tables)=
## Advanced Tables & Lists

### Comparison Matrix

```{list-table} Feature Comparison Matrix
:header-rows: 1
:stub-columns: 1
:widths: 25 20 20 20 15

* - Feature
  - Starter Plan
  - Professional
  - Enterprise
  - Custom
* - **API Calls/Month**
  - 10,000
  - 100,000
  - 1,000,000
  - Unlimited
* - **Storage (GB)**
  - 1
  - 10
  - 100
  - Unlimited
* - **Team Members**
  - 1
  - 5
  - 25
  - Unlimited
* - **SLA Guarantee**
  - 99%
  - 99.5%
  - 99.9%
  - 99.99%
* - **Support Level**
  - Community
  - Email
  - Priority
  - Dedicated
* - **Custom Integrations**
  - {octicon}`x;1em;sd-text-danger`
  - {octicon}`check;1em;sd-text-success`
  - {octicon}`check;1em;sd-text-success`
  - {octicon}`check;1em;sd-text-success`
* - **SSO Integration**
  - {octicon}`x;1em;sd-text-danger`
  - {octicon}`x;1em;sd-text-danger`
  - {octicon}`check;1em;sd-text-success`
  - {octicon}`check;1em;sd-text-success`
```

### Configuration Parameters

```{list-table} Configuration Parameters
:header-rows: 1
:widths: 20 30 15 15 20

* - Parameter
  - Description
  - Type
  - Default
  - Example
* - `max_connections`
  - Maximum concurrent database connections
  - Integer
  - `100`
  - `200`
* - `timeout_seconds`
  - Request timeout in seconds
  - Integer
  - `30`
  - `60`
* - `enable_caching`
  - Enable response caching
  - Boolean
  - `true`
  - `false`
* - `cache_ttl_minutes`
  - Cache time-to-live in minutes
  - Integer
  - `60`
  - `120`
* - `log_level`
  - Logging verbosity level
  - String
  - `INFO`
  - `DEBUG`
```

(section-category-topic-subtopic-b-mixed-content)=
## Mixed Content Blocks

Combine different MyST features for rich, informative sections.

### Implementation Guide

::::{dropdown} Prerequisites Checklist
:icon: list-unordered
:color: info

Before implementing this solution, ensure you have:

- [ ] Administrative access to your system
- [ ] Valid API credentials configured
- [ ] Network connectivity to required endpoints
- [ ] Minimum 4GB RAM available
- [ ] Python 3.8+ or Node.js 16+ installed

:::{tip}
Run the system requirements check script first: `python check_requirements.py`
:::
::::

::::{tab-set}

:::{tab-item} Quick Start
:sync: impl-type

**5-Minute Setup**

1. Install the package:
   ```bash
   pip install our-package
   ```

2. Initialize configuration:
   ```bash
   our-package init --interactive
   ```

3. Start the service:
   ```bash
   our-package start --daemon
   ```

```{note}
The quick start uses default settings. For production use, follow the complete setup.
```

:::

:::{tab-item} Complete Setup
:sync: impl-type

**Production-Ready Configuration**

1. **Environment Preparation**

   ```bash
   # Create isolated environment
   python -m venv production_env
   source production_env/bin/activate
   ```

2. **Secure Installation**

   ```bash
   # Install with security extras
   pip install our-package[security,monitoring]
   ```

3. **Configuration File**

    ````{dropdown} Complete configuration template
   :icon: file-code
   
   ```yaml
   # config/production.yml
   server:
     host: "0.0.0.0"
     port: 8080
     workers: 4
     
   database:
     url: "${DATABASE_URL}"
     pool_size: 20
     max_overflow: 30
     
   security:
     secret_key: "${SECRET_KEY}"
     token_expiry: "24h"
     rate_limit: "1000/hour"
     
   monitoring:
     metrics_enabled: true
     health_check_path: "/health"
     log_level: "INFO"
   ````

4. **Start with Monitoring**

   ```bash
   our-package start --config config/production.yml --monitor
   ```

:::
::::

:::{warning}
Always use environment variables for secrets in production. Never commit credentials to version control.
:::

(section-category-topic-subtopic-b-responsive-design)=
## Responsive Design Examples

Content that adapts to different screen sizes and contexts.

### Mobile-First Card Layout

::::{grid} 1 2 3 4
:gutter: 1 2 2 3
:class-container: sd-p-3

:::{grid-item-card} {octicon}`database;1.5em` Storage
:class-card: sd-text-center
:shadow: md

**500GB**  
Included storage
:::

:::{grid-item-card} {octicon}`cloud;1.5em` Bandwidth
:class-card: sd-text-center
:shadow: md

**1TB/month**  
Data transfer
:::

:::{grid-item-card} {octicon}`cpu;1.5em` Processing
:class-card: sd-text-center
:shadow: md

**2.4 GHz**  
8-core CPU
:::

:::{grid-item-card} {octicon}`shield;1.5em` Security
:class-card: sd-text-center
:shadow: md

**256-bit**  
Encryption
:::
::::

### Flexible Content Blocks

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
:class: sd-border-2 sd-border-primary sd-p-3 sd-rounded-2

#### Getting Started {octicon}`play;1em;sd-text-primary`

Perfect for developers new to the platform.

- Interactive tutorials
- Sample projects  
- Video walkthroughs
- Community support

**Time Investment:** 2-3 hours

:::

:::{grid-item}
:class: sd-border-2 sd-border-success sd-p-3 sd-rounded-2

#### Advanced Integration {octicon}`tools;1em;sd-text-success`

For teams building production systems.

- Architecture patterns
- Performance optimization
- Security best practices
- Enterprise features

**Time Investment:** 1-2 weeks

:::
::::

---

:::{seealso}
This comprehensive example demonstrates MyST markdown capabilities. For more patterns, explore:
- {ref}`section-category-topic-subtopic-a` for tabs and tables
- {ref}`section-category-topic` for grid layouts and comparisons
:::