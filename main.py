#!/usr/bin/env python3
"""
PoseSense - Main Entry Point

This is the main entry point for the PoseSense application.
Run this file to start the live action detection system.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for PoseSense."""
    try:
        from src.utils.run_demo import main as run_demo
        run_demo()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Make sure you're running from the project root directory")
        print("📁 Current directory:", os.getcwd())
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running PoseSense: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 