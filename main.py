#!/usr/bin/env python3
"""
Motor Thermal Modeler - Main Entry Point
=========================================
Parametric motor geometry GUI with live preview.
Supports Lumped Parameter Thermal Network and FEA paths.

Usage:
    python main.py
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.motor_gui_v2 import main

if __name__ == "__main__":
    main()
