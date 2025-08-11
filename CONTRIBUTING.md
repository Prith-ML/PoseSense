# Contributing to PoseSense

Thank you for your interest in contributing to PoseSense! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. **Report Issues**
- Use the GitHub issue tracker to report bugs
- Include detailed descriptions and steps to reproduce
- Attach screenshots or error logs when possible
- Check existing issues before creating new ones

### 2. **Suggest Features**
- Open an issue with the "enhancement" label
- Describe the feature and its benefits
- Include use cases and examples
- Discuss implementation approaches

### 3. **Submit Code Changes**
- Fork the repository
- Create a feature branch: `git checkout -b feature/amazing-feature`
- Make your changes and test thoroughly
- Commit with clear messages: `git commit -m "Add amazing feature"`
- Push to your fork: `git push origin feature/amazing-feature`
- Open a Pull Request

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+
- Git
- Basic knowledge of computer vision and deep learning

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/PoseSense.git
cd PoseSense

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements.txt[dev]

# Run tests
python test_system.py
```

## 📋 Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Use type hints when possible

### Example Code Style
```python
def process_skeleton_frame(skeleton_data: np.ndarray, 
                          normalize: bool = True) -> np.ndarray:
    """
    Process a single skeleton frame for action recognition.
    
    Args:
        skeleton_data: Input skeleton data of shape (25, 3)
        normalize: Whether to normalize the skeleton
        
    Returns:
        Processed skeleton data
        
    Raises:
        ValueError: If skeleton data is invalid
    """
    if skeleton_data.shape != (25, 3):
        raise ValueError("Skeleton data must be of shape (25, 3)")
    
    # Process the skeleton
    processed = skeleton_data.copy()
    
    if normalize:
        processed = normalize_skeleton_scale(processed)
    
    return processed
```

## 🧪 Testing Guidelines

### Running Tests
- Always run `python test_system.py` before submitting changes
- Ensure all tests pass
- Add new tests for new functionality
- Test on different platforms if possible

### Test Coverage
- Aim for high test coverage
- Test edge cases and error conditions
- Include performance benchmarks for critical functions

## 📚 Documentation Standards

### Code Documentation
- Use clear, concise docstrings
- Include examples in docstrings
- Document all public APIs
- Keep README.md updated

### Commit Messages
- Use present tense: "Add feature" not "Added feature"
- Use imperative mood: "Move cursor to..." not "Moves cursor to..."
- Limit first line to 72 characters
- Reference issues when relevant: "Fix #123"

## 🚀 Areas for Contribution

### High Priority
- [ ] Add more action classes
- [ ] Improve model accuracy
- [ ] Optimize performance
- [ ] Add support for multiple cameras

### Medium Priority
- [ ] Create web interface
- [ ] Add data augmentation techniques
- [ ] Implement model ensemble methods
- [ ] Add export functionality

### Low Priority
- [ ] Improve documentation
- [ ] Add more visualization options
- [ ] Create tutorials and examples
- [ ] Optimize memory usage

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Operating system and version
   - Python version
   - Package versions (from `pip freeze`)

2. **Error Details**
   - Full error traceback
   - Steps to reproduce
   - Expected vs. actual behavior

3. **Additional Context**
   - Hardware specifications
   - Camera model (if relevant)
   - Screenshots or videos

## 💡 Feature Requests

When suggesting features:

1. **Describe the Problem**
   - What issue does this solve?
   - Who would benefit from this feature?

2. **Propose a Solution**
   - How should this work?
   - Any implementation ideas?

3. **Consider Impact**
   - How will this affect existing functionality?
   - What are the performance implications?

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Wiki**: For detailed documentation and tutorials

## 🎯 Recognition

Contributors will be recognized in:
- Project README.md
- Release notes
- Contributor hall of fame
- GitHub contributor statistics

## 📄 License

By contributing to PoseSense, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to PoseSense! Your help makes this project better for everyone. 🚀 