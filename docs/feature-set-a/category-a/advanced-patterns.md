(feature-set-a-advanced-patterns)=
# Advanced MyST Patterns

This document showcases sophisticated MyST markdown patterns and features for creating rich, professional documentation.

(feature-set-a-advanced-patterns-figures)=
## Figures & Media

Demonstrate advanced figure handling, captions, and cross-references.

### Basic Figure with Caption

```{figure} https://placehold.co/600x400/png
:alt: System architecture showing microservices communication
:width: 600px
:align: center
:name: fig-architecture

System Architecture Overview - This diagram illustrates the communication flow between microservices in our distributed system.
```

As shown in {numref}`fig-architecture`, the system uses an event-driven architecture.

### Responsive Figure Grid

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
```{figure} https://placehold.co/300x400/28a745/ffffff?text=Mobile+UI
:alt: Mobile dashboard interface
:width: 100%
:name: fig-mobile

Mobile Interface - Optimized for touch interactions
```
:::

:::{grid-item}
```{figure} https://placehold.co/400x300/007bff/ffffff?text=Desktop+UI
:alt: Desktop dashboard interface  
:width: 100%
:name: fig-desktop

Desktop Interface - Full-featured admin panel
```
:::
::::

Compare the mobile interface ({numref}`fig-mobile`) with the desktop version ({numref}`fig-desktop`) to see responsive design principles in action.

(feature-set-a-advanced-patterns-math)=
## Mathematical Expressions

Showcase mathematical notation using MyST's math support.

### Inline Mathematics

The algorithm achieves {math}`O(n \log n)` time complexity, where {math}`n` represents the input size.

### Block Mathematics

```{math}
:label: eq-performance

\text{Response Time} = \frac{\text{Queue Length}}{\text{Service Rate}} + \text{Processing Time}
```

The performance equation {eq}`eq-performance` helps predict system behavior under load.

### Complex Mathematical Notation

```{math}
:label: eq-machine-learning

J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
```

The cost function {eq}`eq-machine-learning` includes L2 regularization to prevent overfitting.

### Mathematical Derivation Steps

::::{dropdown} Mathematical Proof
:icon: book
:color: info

**Theorem:** The gradient descent update rule minimizes the cost function.

**Proof:**

Starting with the cost function:
```{math}
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
```

Taking the partial derivative with respect to {math}`\theta_j`:
```{math}
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
```

The gradient descent update becomes:
```{math}
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
```

Therefore:
```{math}
\theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
```

This update rule moves {math}`\theta_j` in the direction of steepest descent. □
::::

(feature-set-a-advanced-patterns-glossary)=
## Glossary & Definitions

Create searchable glossaries and definition lists.

```{glossary}
API
  Application Programming Interface - a set of protocols and tools for building software applications.

Microservice
  An architectural approach where a single application is composed of many loosely coupled services.

JWT
  JSON Web Token - a compact, URL-safe means of representing claims between two parties.

Load Balancer
  A device or software that distributes network traffic across multiple servers.

Container
  A lightweight, portable execution environment that includes everything needed to run an application.

Kubernetes
  An open-source container orchestration platform for automating deployment and management.
```

### Using Glossary Terms

When building modern applications, you'll often use {term}`API`s to enable communication between {term}`Microservice`s. Authentication is typically handled using {term}`JWT` tokens, while a {term}`Load Balancer` distributes traffic across multiple {term}`Container` instances managed by {term}`Kubernetes`.

### Definition Lists

**Authentication Methods**
: Various approaches to verify user identity

**Authorization Levels**
: Different permission tiers for system access

**Rate Limiting**
: Controlling the frequency of API requests

**Circuit Breaker**
: Pattern to prevent cascading failures in distributed systems

(feature-set-a-advanced-patterns-footnotes)=
## Footnotes & Citations

Demonstrate scholarly citation patterns.

### Research Citations

Modern distributed systems rely on eventual consistency models[^1] to achieve high availability across geographically distributed data centers. The CAP theorem[^2] proves that it's impossible to simultaneously guarantee consistency, availability, and partition tolerance.

Performance optimization in microservices architectures often involves implementing circuit breaker patterns[^3] and bulkhead isolation[^4] to prevent cascading failures.

[^1]: Werner Vogels, "Eventually Consistent," Communications of the ACM, vol. 52, no. 1, pp. 40-44, 2009.

[^2]: Eric Brewer, "CAP Twelve Years Later: How the Rules Have Changed," Computer, vol. 45, no. 2, pp. 23-29, 2012.

[^3]: Michael Nygard, "Release It! Design and Deploy Production-Ready Software," Pragmatic Bookshelf, 2018.

[^4]: Netflix Technology Blog, "Fault Tolerance in a High Volume, Distributed System," https://netflixtechblog.com/fault-tolerance-in-a-high-volume-distributed-system-91ab4faae74a

### Inline Citations

According to the latest performance benchmarks[^benchmark], our optimization improvements resulted in a 300% increase in throughput.

[^benchmark]: Internal Performance Report Q3 2024, Engineering Team Analysis

(feature-set-a-advanced-patterns-code-execution)=
## Executable Code Blocks

Show code examples with execution results and interactive elements.

### Python Code with Output

```{code-block} python
:linenos:
:emphasize-lines: 3,7

def fibonacci(n):
    """Calculate nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    
    return b

# Example usage
result = fibonacci(10)
print(f"The 10th Fibonacci number is: {result}")
```

**Output:**
```
The 10th Fibonacci number is: 55
```

### Multi-Language Code Comparison

::::{tab-set}

:::{tab-item} Python
:sync: lang-compare

```{code-block} python
:caption: Binary search implementation

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Time complexity: O(log n)
# Space complexity: O(1)
```
:::

:::{tab-item} JavaScript
:sync: lang-compare

```{code-block} javascript
:caption: Binary search implementation

function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}

// Time complexity: O(log n)
// Space complexity: O(1)
```
:::

:::{tab-item} Go
:sync: lang-compare

```{code-block} go
:caption: Binary search implementation

func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    
    for left <= right {
        mid := (left + right) / 2
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}

// Time complexity: O(log n)
// Space complexity: O(1)
```
:::
::::

(feature-set-a-advanced-patterns-cross-refs)=
## Advanced Cross References

Demonstrate sophisticated internal linking and reference systems.

### Section References

- Architecture overview: {ref}`feature-set-a-advanced-patterns-figures`
- Mathematical foundations: {ref}`feature-set-a-advanced-patterns-math`
- Code implementations: {ref}`feature-set-a-advanced-patterns-code-execution`

### Document References

See our related documentation:
- {doc}`topic-a/index` - Core concepts and comparisons
- {doc}`topic-a/subtopic-a` - Tab-based examples and API usage
- {doc}`topic-a/subtopic-b` - Comprehensive MyST pattern showcase

### External References

```{seealso}
**Related Documentation:**
- {ref}`section-category-topic` - Topic overview with comparison tables
- {ref}`feature-set-a-tuts-series-a` - Interactive tutorial series
- {ref}`feature-set-a-tutorials-beginner` - Getting started guide

**External Resources:**
- [MyST Parser Syntax Guide](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html)
- [Sphinx Design Documentation](https://sphinx-design.readthedocs.io/)
```

(feature-set-a-advanced-patterns-directives)=
## Custom Directives & Roles

Showcase specialized MyST directives for enhanced content presentation.

### Version Information

```{versionadded} 2.1
Support for advanced mathematical notation in code blocks.
```

```{versionchanged} 2.0
The authentication system now supports OAuth 2.0 and JWT tokens.
```

```{deprecated} 1.8
The legacy authentication method will be removed in version 3.0. Use JWT tokens instead.
```

### Todo Items

:::{note}
**Development Task:** Add performance benchmarks for the new caching layer implementation.
:::

:::{note}
**Documentation Task:** Update the API documentation with new endpoint descriptions.
:::

### Code Documentation

```{function} calculate_performance_metrics(requests, duration)
Calculate key performance indicators for API endpoints.

:param requests: List of HTTP request objects
:type requests: List[Request]
:param duration: Time window for analysis in seconds  
:type duration: int
:returns: Dictionary containing performance metrics
:rtype: Dict[str, float]

**Example usage:**

.. code-block:: python

   metrics = calculate_performance_metrics(request_log, 3600)
   print(f"Average response time: {metrics['avg_response_time']}ms")
```

### Content Blocks

```{contents} Page Contents
:local:
:depth: 2
```

### Index Entries

```{index} single: Performance; Optimization
```

```{index} pair: API; Authentication
```

Performance optimization {index}`techniques <Performance>` are essential for scalable applications.

(feature-set-a-advanced-patterns-accessibility)=
## Accessibility Features

Demonstrate inclusive design patterns in documentation.

### Screen Reader Friendly Tables

```{list-table} API Endpoint Performance Metrics
:header-rows: 1
:stub-columns: 1
:widths: 25 20 20 20 15
:name: table-performance

* - Endpoint
  - Avg Response (ms)
  - 95th Percentile (ms)  
  - Requests/sec
  - Error Rate (%)
* - **GET /users**
  - 45
  - 120
  - 1,200
  - 0.1
* - **POST /users**
  - 85
  - 200
  - 800
  - 0.3
* - **GET /analytics**
  - 150
  - 400
  - 500
  - 0.2
* - **POST /data-export**
  - 2,500
  - 5,000
  - 50
  - 1.2
```

Reference {numref}`table-performance` for detailed performance characteristics of each endpoint.

### Alt Text for Visual Elements

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} {octicon}`accessibility;1.5em;sd-mr-1` Accessibility First
:class-header: sd-bg-success sd-text-white

All visual elements include descriptive alternative text for screen readers.
+++
{bdg-success}`WCAG 2.1 AA` {bdg-secondary}`Screen Reader Tested`
:::

:::{grid-item-card} {octicon}`device-mobile;1.5em;sd-mr-1` Mobile Optimized
:class-header: sd-bg-primary sd-text-white

Responsive design ensures content accessibility across all device sizes.
+++
{bdg-primary}`Touch Friendly` {bdg-secondary}`High Contrast`
:::
::::

### Keyboard Navigation

:::{note}
All interactive elements in this documentation support keyboard navigation:
- **Tab**: Move to next interactive element
- **Shift+Tab**: Move to previous interactive element  
- **Enter/Space**: Activate buttons and links
- **Arrow keys**: Navigate within tab sets and dropdowns
:::

### Color and Contrast

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item}
:class: sd-bg-success sd-text-white sd-p-3 sd-text-center

**Success State**  
High contrast ratio: 7.2:1
:::

:::{grid-item}
:class: sd-bg-warning sd-text-dark sd-p-3 sd-text-center

**Warning State**  
High contrast ratio: 8.1:1
:::

:::{grid-item}
:class: sd-bg-danger sd-text-white sd-p-3 sd-text-center

**Error State**  
High contrast ratio: 12.3:1
:::

:::{grid-item}
:class: sd-bg-info sd-text-white sd-p-3 sd-text-center

**Info State**  
High contrast ratio: 6.8:1
:::
::::

All color combinations meet WCAG 2.1 AA contrast requirements (minimum 4.5:1 ratio).

---

:::{seealso}
**Pattern Summary:**
This document demonstrates advanced MyST markdown patterns including:
- Mathematical notation and equations with cross-references
- Sophisticated figure handling and responsive layouts
- Glossary integration and definition lists
- Scholarly citations and footnotes
- Multi-language code examples with syntax highlighting
- Accessibility-focused design patterns
- Custom directives for version control and documentation

**Next Steps:**
- {ref}`section-category-topic-subtopic-b` - Additional MyST pattern examples  
- {ref}`feature-set-a-tuts-series-a` - Interactive learning experiences
- {doc}`../tutorials/index` - Hands-on tutorial collection
:::
