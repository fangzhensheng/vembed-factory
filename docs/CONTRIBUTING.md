# Contributing to vembed-factory

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them.

## Table of Contents

- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
- [Styleguides](#styleguides)
  - [Commit Messages](#commit-messages)

## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](README.md).

Before you ask a question, it is best to search for existing [Issues](https://github.com/fangzhensheng/vembed-factory/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/fangzhensheng/vembed-factory/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (nodejs, npm, etc), depending on what seems relevant.

## I Want To Contribute

### Reporting Bugs

Before creating bug reports, please check this list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible. Fill out [the required template](https://github.com/fangzhensheng/vembed-factory/issues/new?template=bug_report.md), the information it asks for helps us resolve issues faster.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for vembed-factory, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

### Your First Code Contribution

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Styleguides

### Code Style

We use `black`, `isort`, and `ruff` for code formatting and linting. Please ensure your code passes these checks before submitting a PR.

```bash
pip install black isort ruff
black .
isort .
ruff check .
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
