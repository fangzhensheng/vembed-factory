# API Documentation Template Guide

This template provides a standardized format for all API documentation in vembed-factory.

## Template Structure

### 1. Module Title & Description (Required)
```markdown
# Module Name

One-line description of the module's purpose.

## Overview

Detailed explanation of what this module does, its main components, and when to use it.
```

### 2. Key Classes/Functions Table (Required)
```markdown
### Key Classes

| Class | Purpose |
|-------|---------|
| `ClassName` | Brief description of what it does |
```

### 3. Quick Start Example (Required)
```markdown
## Quick Start

### Basic Usage

```python
from vembed.module.submodule import ClassName

# Initialize
obj = ClassName(param1=value1)

# Use
result = obj.method()
```
```

### 4. Common Use Cases (Optional but Recommended)
```markdown
## Common Use Cases

### Use Case 1: Description
```python
# Code example
```

### Use Case 2: Description
```python
# Code example
```
```

### 5. API Reference (Auto-generated)
```markdown
## API Reference

::: vembed.module.ClassName
::: vembed.module.function_name
```

### 6. FAQs & Troubleshooting (Optional)
```markdown
## FAQs

**Q: Common question?**
A: Answer with example.
```

---

## Documentation Quality Checklist

- [ ] Title is clear and descriptive
- [ ] Overview section explains purpose and use cases
- [ ] Key classes/functions are listed with descriptions
- [ ] Quick start example is complete and runnable
- [ ] Common use cases include practical examples
- [ ] Code examples follow project style guide
- [ ] Related modules/classes are referenced
- [ ] API Reference section uses proper MkDocs syntax
- [ ] All links are valid and relative

---

## Writing Guidelines

### DO's ✅
- Write for developers unfamiliar with the codebase
- Include practical, runnable examples
- Use clear, simple language
- Link to related modules
- Explain why, not just what
- Format code blocks with syntax highlighting

### DON'Ts ❌
- Don't assume knowledge of implementation details
- Don't make examples too complex or too simple
- Don't skip error handling in examples
- Don't forget to mention dependencies
- Don't leave broken links

---

## Module-Specific Tips

### For Data Modules
- Explain input/output formats
- Show example data structures
- Mention supported file types

### For Model Modules
- Show how to load/initialize models
- Explain encoder modes
- Include inference examples

### For Loss/Training Modules
- Show training loop example
- Explain hyperparameter impact
- Include convergence tips

### For Utility Modules
- Show common patterns
- Explain registry system if applicable
- Include error handling examples
