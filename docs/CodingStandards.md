# Coding Standards

## Introduction
We set coding standards which are a set of guidelines and best practices that define how our codes should be written, formatted, and organized within this project. Adhering to coding standards helps maintain consistency, readability, and maintainability across the codebase.

## Purpose
The purpose of this document is to outline the coding standards to be followed by developers working on this project. Consistent adherence to these standards ensures that the codebase remains clean, understandable, and easy to maintain.

Below are the coding standards strictly followed by everyone working on this project.

## General Guidelines
- **Readability**: Write code that is easy to read, understand, and maintain. Use descriptive names for variables, functions, and classes.
- **Consistency**: Follow consistent naming conventions, formatting styles, and code organization patterns throughout the project.
- **Modularity**: Write modular code with clear separation of concerns. Encapsulate related functionality into functions, classes, or modules.
- **Documentation**: Document code comprehensively using comments, docstrings, and README files. Explain the purpose, usage, and behavior of functions, classes, and modules.
- **Error Handling**: Implement robust error handling mechanisms to gracefully handle exceptions and failures. Use meaningful error messages and log errors appropriately.

## Coding Style
- **Python PEP 8**: Follow the Python Enhancement Proposal 8 (PEP 8) guidelines for code style, including indentation, line length, imports, and spacing.
- **Naming Conventions**: Use descriptive and meaningful names for variables, functions, classes, and modules. Follow the *snake_case* naming convention for variables and functions, and the *CamelCase* convention for classes.
- **Whitespace**: Use consistent whitespace and indentation. Indent Python code blocks with four spaces, JSON, YAML with two spaces, and separate functions and classes with two blank lines.
- **Imports**: Organize imports alphabetically and group them into three sections: standard library imports, third-party library imports, and local imports.
- **Comments**: Write clear and concise comments to explain the purpose, behavior, and usage of code. Use inline comments sparingly and focus on documenting complex or non-obvious logic.

## Best Practices
- **DRY Principle**: Don't Repeat Yourself (DRY). Avoid duplicating code and instead encapsulate reusable functionality into functions or classes.
- **Single Responsibility Principle (SRP)**: Each function, class, or module should have a single responsibility or purpose. Avoid creating monolithic or overly complex components.
- **Version Control**: Use Git version control system to track code changes, collaborate with team members, and manage code revisions effectively. Follow branching and commit conventions agreed upon by the team.
- **Testing**: Write unit tests to validate the functionality of individual components. Aim for high test coverage to ensure code correctness and reliability.
- **Code Reviews**: Conduct thorough code reviews to provide feedback, identify potential issues, and ensure adherence to coding standards. Ping your teamates in the Teams channels and ask for code reviews and checklists to streamline the review process.
- **Pull Requests**: Pull requests on the `main` branch are protected, and all merge branch requests has to be approved by atleast another member (except the person requesting) from the team. This is to ensure that all the above standards are strictly adhered to and consitency throughout the repository is maintained.

## Conclusion
Adhering to coding standards is essential for maintaining code quality, readability, and consistency across a software project. By following the guidelines outlined in this document, developers can contribute to a clean, maintainable codebase that facilitates collaboration and long-term maintainability.