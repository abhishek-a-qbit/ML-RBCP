# Contributing to Convection Solver

Thank you for your interest in contributing to the AI-Powered Rayleigh-Bénard Convection Solver! This document provides guidelines for contributing to the project.

## 🎯 How You Can Contribute

### 1. Report Bugs
- Use GitHub Issues to report bugs
- Include a clear description and steps to reproduce
- Provide system information (OS, Python version, package versions)
- Include error messages and screenshots if applicable

### 2. Suggest Enhancements
- Use GitHub Issues with the "enhancement" label
- Clearly describe the feature and its benefits
- Explain the use case and expected behavior

### 3. Submit Code
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations
- Test cases

## 🔧 Development Setup

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/convection-solver.git
cd convection-solver

# Add upstream remote
git remote add upstream https://github.com/originalauthor/convection-solver.git
```

### Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## 📝 Code Style Guidelines

### Python Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Maximum line length: 100 characters
- Use type hints where appropriate

### Example:

```python
def compute_nusselt_number(
    temperature_field: np.ndarray,
    height: float,
    delta_T: float
) -> float:
    """
    Compute the Nusselt number from temperature field.
    
    Parameters
    ----------
    temperature_field : np.ndarray
        2D array of temperature values
    height : float
        Domain height
    delta_T : float
        Temperature difference
        
    Returns
    -------
    float
        Nusselt number
    """
    # Implementation
    pass
```

### Formatting Tools

```bash
# Format code with black
black app.py

# Check style with flake8
flake8 app.py

# Type checking with mypy
mypy app.py
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_solvers.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Include edge cases

Example:

```python
def test_traditional_solver_convergence():
    """Test that traditional solver converges for standard case"""
    solver = TraditionalSolver(nx=30, ny=30, Ra=1e4, Pr=0.71, 
                               aspect_ratio=2.0, max_iter=1000)
    results = solver.solve()
    
    assert results is not None
    assert 'temperature' in results
    # Check boundary conditions
    assert np.allclose(results['temperature'][0, :], 1.0)
    assert np.allclose(results['temperature'][-1, :], 0.0)
```

## 📚 Documentation

### Docstrings
- Use NumPy style docstrings
- Include parameter types and descriptions
- Add examples where helpful

### README Updates
- Update README.md if adding new features
- Include usage examples
- Update installation instructions if needed

### Comments
- Explain *why*, not *what*
- Comment complex algorithms
- Keep comments up-to-date with code changes

## 🔄 Pull Request Process

### Before Submitting

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**
   ```bash
   pytest
   ```

3. **Format code**
   ```bash
   black .
   flake8 .
   ```

4. **Update documentation**
   - Add/update docstrings
   - Update README if needed
   - Add to CHANGELOG.md

### Submitting PR

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request on GitHub**
   - Use descriptive title
   - Fill out PR template
   - Link related issues
   - Add screenshots/examples if applicable

3. **PR Description Should Include:**
   - What changes were made
   - Why the changes are needed
   - How to test the changes
   - Any breaking changes

### Example PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How to test the changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
```

## 🎓 Contribution Areas

### High Priority
- [ ] Additional PINN architectures (Fourier Features, etc.)
- [ ] 3D convection problems
- [ ] Turbulent regime handling
- [ ] Experimental data validation
- [ ] Performance optimization

### Medium Priority
- [ ] More visualization options
- [ ] Additional convection types (mixed, forced)
- [ ] Better error analysis tools
- [ ] Uncertainty quantification
- [ ] Cloud deployment

### Documentation Needs
- [ ] Tutorial notebooks
- [ ] Theory documentation
- [ ] API documentation
- [ ] Video tutorials
- [ ] Use case examples

## 🤝 Code Review Process

### What Reviewers Look For
- Code quality and style
- Test coverage
- Documentation completeness
- Performance implications
- Breaking changes

### Response to Feedback
- Be receptive to suggestions
- Ask questions if unclear
- Make requested changes promptly
- Explain your reasoning when disagreeing

## 🌟 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Thanked in documentation
- Considered for co-authorship on publications (major contributions)

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Chat**: Join our Discord/Slack (if available)
- **Email**: maintainer@example.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

**Happy Contributing! 🌊**
