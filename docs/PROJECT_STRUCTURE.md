# PoseSense Project Structure

```
PoseSense/
├── 📁 src/                    # Source code package
│   ├── __init__.py            # Package initialization
│   ├── 📁 core/               # Core application components
│   │   ├── __init__.py        # Core module initialization
│   │   ├── liveApplicationCode.py  # Main application
│   │   ├── liveModelTraining.py    # Training script
│   │   └── config.py               # Configuration settings
│   ├── 📁 utils/              # Utility scripts and helpers
│   │   ├── __init__.py        # Utils module initialization
│   │   └── run_demo.py        # Quick start demo script
│   └── 📁 models/             # Pre-trained models
│       ├── __init__.py        # Models module initialization
│       └── pytorchModel.pth   # Pre-trained PyTorch model
│
├── 📁 docs/                   # Documentation
│   ├── README.md              # Detailed project documentation
│   ├── PROJECT_STRUCTURE.md   # This file
│   ├── CONTRIBUTING.md        # Contribution guidelines
│   ├── LICENSE                # MIT License
│   └── Dataset                # Dataset download instructions
│
├── 📁 tests/                  # Testing and diagnostics
│   └── test_system.py         # Comprehensive system testing
│
├── 📁 examples/               # Usage examples (future)
│
├── 📁 .github/                # GitHub configuration
│   └── 📁 workflows/          # CI/CD workflows
│       └── ci.yml             # Continuous integration
│
├── main.py                    # Main entry point
├── README.md                  # Project overview (root)
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
└── .gitignore                 # Git ignore patterns
```

## 📋 File Descriptions

### 🚀 Main Entry Point
- **`main.py`**: New main entry point that imports from restructured modules

### 📁 Source Code (`src/`)
- **`__init__.py`**: Makes src a Python package with version info and imports

#### Core Module (`src/core/`)
- **`liveApplicationCode.py`**: Main application with real-time action detection
- **`liveModelTraining.py`**: Script to train the LSTM model on skeleton data
- **`config.py`**: Centralized configuration for all system parameters

#### Utils Module (`src/utils/`)
- **`run_demo.py`**: User-friendly script to start the application

#### Models Module (`src/models/`)
- **`pytorchModel.pth`**: Pre-trained LSTM model weights (953KB)

### 📚 Documentation (`docs/`)
- **`README.md`**: Comprehensive project documentation and usage guide
- **`PROJECT_STRUCTURE.md`**: This file explaining the codebase organization
- **`CONTRIBUTING.md`**: Guidelines for contributors
- **`LICENSE`**: MIT License for open-source usage
- **`Dataset`**: Instructions for downloading the training dataset

### 🧪 Testing (`tests/`)
- **`test_system.py`**: Comprehensive system testing and diagnostics

### ⚙️ Configuration Files
- **`requirements.txt`**: Python package dependencies
- **`setup.py`**: Package installation and distribution setup
- **`.gitignore`**: Git ignore patterns for clean repository

## 🔄 Import Structure

### Before Restructuring
```python
# Direct imports from root
from liveApplicationCode import SkeletonLSTM
import config
```

### After Restructuring
```python
# Package imports from src
from src.core.liveApplicationCode import SkeletonLSTM
from src.core import config
```

## 🚀 Running the Application

### New Way (Recommended)
```bash
# From project root
python main.py
```

### Alternative Ways
```bash
# Direct module execution
python src/utils/run_demo.py

# Core application
python src/core/liveApplicationCode.py
```

## 🎯 Benefits of New Structure

### 1. **Professional Organization**
- Follows Python package conventions
- Clear separation of concerns
- Easy to navigate and understand

### 2. **Better Maintainability**
- Logical grouping of related files
- Easier to add new features
- Cleaner import statements

### 3. **Scalability**
- Easy to add new modules
- Clear structure for contributors
- Professional appearance for GitHub

### 4. **Import Management**
- Proper Python package structure
- No more relative path issues
- Clean dependency management

## 🔧 Migration Notes

If you're updating from the old structure:

1. **Update imports** to use `src.core.*` and `src.utils.*`
2. **Use main.py** as the new entry point
3. **Model path** is now `src/models/pytorchModel.pth`
4. **Test scripts** are in the `tests/` directory

## 📈 Future Expansion

The new structure makes it easy to add:

- **New action classes** in `src/core/`
- **Additional utilities** in `src/utils/`
- **More models** in `src/models/`
- **Web interface** in `src/web/`
- **API endpoints** in `src/api/`
- **Data processing** in `src/data/`

## 🎉 Getting Started

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Test System**: `python tests/test_system.py`
3. **Run Application**: `python main.py`
4. **Explore Code**: Navigate through the organized `src/` structure 