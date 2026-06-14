"""
Main GUI Application for Motor Thermal Modeling.
Provides parameter input, live geometry preview, and LPTN thermal simulation.
"""

import sys
import os
import math
import numpy as np
from typing import Optional, Dict, List

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QGroupBox, QFormLayout, QLabel, QLineEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QSplitter,
    QMessageBox, QTextEdit, QCheckBox, QGridLayout, QFrame,
    QStatusBar, QAction, QFileDialog, QSlider, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractSpinBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtGui import QFont, QPalette, QColor

# Import our modules
from geometry.motor_geometry import MotorGeometryParams
from geometry.plotter import MotorGeometryCanvas, get_geometry_data, COLORS
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from materials.material_db import (
    get_material, get_magnet_for_grade, get_insulation_temp,
    MATERIAL_CATALOG, MAGNET_GRADES, INSULATION_CLASSES,
)
from lpt_fe import (
    NetworkBuilderConfig, build_thermal_network, solve_steady_state,
    ThermalNetwork, k_slot_equivalent,
)
from lpt_fe.node import ThermalNode, ThermalResistance


class FloatInput(QDoubleSpinBox):
    """A labeled spin box for float parameter input."""
    def __init__(self, label_text, default, min_val=0.0, max_val=500.0,
                 step=0.5, unit="mm", decimals=2):
        super().__init__()
        self.setRange(min_val, max_val)
        self.setSingleStep(step)
        self.setDecimals(decimals)
        self.setValue(default)
        if unit:
            self.setSuffix(f" {unit}")
        self.setToolTip(f"{label_text}")
        self._label_text = label_text
        # Hide arrow buttons by setting button symbols to none
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)

    def label(self):
        return self._label_text

    def set_value_silent(self, val):
        """Set value without emitting signals (for programmatic updates)."""
        self.blockSignals(True)
        self.setValue(val)
        self.blockSignals(False)


class GeometryInputPanel(QWidget):
    """
    Panel with all geometry parameter inputs.
    Organized in collapsible groups for clarity.
    """

    paramChanged = None  # Will be connected externally

    def __init__(self, parent=None):
        super().__init__(parent)
        self._geo = MotorGeometryParams()  # default params
        self._widgets = {}  # For tracking all input widgets
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(5, 5, 5, 5)

        # Scroll area for the whole panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(6)

        # ============================================================
        # 1. Structure Type
        # ============================================================
        struct_group = QGroupBox("Motor Structure")
        struct_layout = QFormLayout(struct_group)

        self._widgets["structure_type"] = QComboBox()
        self._widgets["structure_type"].addItems([
            "Slotted (distributed)", "Slotted (concentrated)", "Slotless"
        ])
        self._widgets["structure_type"].currentTextChanged.connect(self._on_param_change)
        struct_layout.addRow("Type:", self._widgets["structure_type"])
        scroll_layout.addWidget(struct_group)

        # ============================================================
        # 2. Core Dimensions
        # ============================================================
        dim_group = QGroupBox("Core Dimensions")
        dim_layout = QFormLayout(dim_group)

        dim_params = [
            ("Rso", "Stator Outer Radius", 50.0, 0, 99999, 1),
            ("Rsi", "Stator Inner Radius", 30.0, 0, 99999, 1),
            ("stack_length", "Stack Length", 60.0, 0, 99999, 1),
            ("Rro", "Rotor Outer Radius", 29.0, 0, 99999, 0.5),
            ("Rri", "Rotor Inner Radius", 10.0, 0, 99999, 0.5),
            ("airgap_length", "Airgap", 1.0, 0, 99999, 0.1),
        ]
        for key, label, default, vmin, vmax, step in dim_params:
            sb = FloatInput(label, default, vmin, vmax, step)
            sb.valueChanged.connect(self._on_param_change)
            self._widgets[key] = sb
            dim_layout.addRow(f"{label}:", sb)

        scroll_layout.addWidget(dim_group)

        # ============================================================
        # 3. Slot & Winding Geometry (shown for all types)
        # ============================================================
        self._slot_group = QGroupBox("Slot & Winding Geometry")
        slot_layout = QFormLayout(self._slot_group)

        slot_params = [
            ("num_slots", "Number of Slots", 24, 0, 99999, 1, "int", False),
            ("slot_depth", "Winding / Slot Depth", 10.0, 0, 99999, 0.5, "float", True),
            ("slot_opening", "Slot Opening Width", 2.0, 0, 99999, 0.2, "float", False),
            ("tooth_width_min", "Tooth Width (min)", 3.0, 0, 99999, 0.5, "float", False),
            ("slot_wedge_height", "Wedge Height", 0.5, 0, 99999, 0.1, "float", False),
            ("winding_layers", "Winding Layers", 2, 0, 99999, 1, "int", False),
            ("turns_per_slot", "Turns / Slot", 20, 0, 99999, 1, "int", True),
            ("conductor_diameter", "Conductor Diameter", 1.0, 0, 99999, 0.1, "float", True),
            ("fill_factor", "Target Fill Factor", 0.45, 0, 99999, 0.01, "float", True),
        ]

        # Store slot-specific widget labels for show/hide
        self._slot_only_widgets = set()

        for params in slot_params:
            key = params[0]
            label = params[1]
            default = params[2]
            vmin = params[3]
            vmax = params[4]
            step = params[5]
            ptype = params[6]
            show_for_slotless = params[7]

            if ptype == "int":
                sb = QSpinBox()
                sb.setRange(int(vmin), int(vmax))
                sb.setValue(int(default))
                sb.setSingleStep(int(step))
                sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
            else:
                sb = QDoubleSpinBox()
                sb.setRange(vmin, vmax)
                sb.setValue(default)
                sb.setSingleStep(step)
                sb.setDecimals(2)
                sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
                if key != "fill_factor":
                    sb.setSuffix(" mm")

            sb.valueChanged.connect(self._on_param_change)
            self._widgets[key] = sb
            slot_layout.addRow(f"{label}:", sb)

            if not show_for_slotless:
                self._slot_only_widgets.add(key)

        scroll_layout.addWidget(self._slot_group)

        # ============================================================
        # 4. Magnet Configuration
        # ============================================================
        mag_group = QGroupBox("Magnet Configuration")
        mag_layout = QFormLayout(mag_group)

        self._widgets["num_poles"] = QSpinBox()
        self._widgets["num_poles"].setRange(0, 99999)
        self._widgets["num_poles"].setValue(8)
        self._widgets["num_poles"].setSingleStep(2)
        self._widgets["num_poles"].setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._widgets["num_poles"].valueChanged.connect(self._on_param_change)
        mag_layout.addRow("Number of Poles:", self._widgets["num_poles"])

        self._widgets["magnet_thickness"] = FloatInput("Magnet Thickness",
                                                        4.0, 0, 99999, 0.5)
        self._widgets["magnet_thickness"].valueChanged.connect(self._on_param_change)
        mag_layout.addRow("Magnet Thickness:", self._widgets["magnet_thickness"])

        self._widgets["magnet_span_ratio"] = FloatInput(
            "Magnet Span Ratio", 0.82, 0, 1.0, 0.01, "")
        self._widgets["magnet_span_ratio"].valueChanged.connect(self._on_param_change)
        mag_layout.addRow("Pole Arc Ratio:", self._widgets["magnet_span_ratio"])

        self._widgets["magnet_grade"] = QComboBox()
        self._widgets["magnet_grade"].addItems(sorted(MAGNET_GRADES.keys()))
        self._widgets["magnet_grade"].currentTextChanged.connect(self._on_grade_changed)
        mag_layout.addRow("Magnet Grade:", self._widgets["magnet_grade"])

        # Magnet max temp (read-only, auto-set from grade)
        self._widgets["magnet_max_temp"] = QLineEdit()
        self._widgets["magnet_max_temp"].setReadOnly(True)
        self._widgets["magnet_max_temp"].setStyleSheet("background-color: #f0f0f0; color: #555;")
        mag_layout.addRow("Max Magnet Temp:", self._widgets["magnet_max_temp"])

        scroll_layout.addWidget(mag_group)

        # Initialize magnet max temp from default grade
        self._on_grade_changed(self._widgets["magnet_grade"].currentText())

        # ============================================================
        # 5. Housing
        # ============================================================
        housing_group = QGroupBox("Housing")
        housing_layout = QFormLayout(housing_group)

        self._widgets["housing_wall_thickness"] = FloatInput(
            "Housing Wall", 5.0, 0, 99999, 0.5)
        self._widgets["housing_wall_thickness"].valueChanged.connect(self._on_param_change)
        housing_layout.addRow("Wall Thickness:", self._widgets["housing_wall_thickness"])

        self._widgets["housing_finned"] = QCheckBox("Finned Housing")
        self._widgets["housing_finned"].stateChanged.connect(self._on_param_change)
        housing_layout.addRow("", self._widgets["housing_finned"])

        scroll_layout.addWidget(housing_group)

        # ============================================================
        # 6. Insulation Class
        # ============================================================
        ins_group = QGroupBox("Insulation")
        ins_layout = QFormLayout(ins_group)

        self._widgets["insulation_class"] = QComboBox()
        self._widgets["insulation_class"].addItems(sorted(INSULATION_CLASSES.keys(), reverse=True))
        self._widgets["insulation_class"].currentTextChanged.connect(self._on_param_change)
        ins_layout.addRow("Class:", self._widgets["insulation_class"])

        scroll_layout.addWidget(ins_group)

        # ============================================================
        # Info / Computed display
        # ============================================================
        info_group = QGroupBox("Derived Quantities")
        self._info_display = QTextEdit()
        self._info_display.setReadOnly(True)
        self._info_display.setMaximumHeight(150)
        info_layout = QVBoxLayout(info_group)
        info_layout.addWidget(self._info_display)
        scroll_layout.addWidget(info_group)

        # Button row
        btn_layout = QHBoxLayout()
        self._btn_reset = QPushButton("Reset Defaults")
        self._btn_reset.clicked.connect(self._reset_defaults)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_reset)
        scroll_layout.addLayout(btn_layout)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def get_parameters(self) -> MotorGeometryParams:
        """Read all input widgets and return a MotorGeometryParams."""
        try:
            stype = self._widgets["structure_type"].currentText()
            is_slotted = stype != "Slotless"

            # Show/hide slot-only widgets
            for key in self._slot_only_widgets:
                self._widgets[key].setVisible(is_slotted)

            # Read magnet max temp from read-only label
            mag_temp_text = self._widgets["magnet_max_temp"].text().replace(" °C", "")
            magnet_max_temp = float(mag_temp_text) if mag_temp_text else 150.0

            geo = MotorGeometryParams(
                structure_type=stype,
                Rso=self._widgets["Rso"].value(),
                Rsi=self._widgets["Rsi"].value(),
                stack_length=self._widgets["stack_length"].value(),
                Rro=self._widgets["Rro"].value(),
                Rri=self._widgets["Rri"].value(),
                airgap_length=self._widgets["airgap_length"].value(),
                num_slots=self._widgets["num_slots"].value() if is_slotted else 0,
                slot_depth=self._widgets["slot_depth"].value(),  # winding thickness for slotless
                slot_opening=self._widgets["slot_opening"].value() if is_slotted else 0,
                tooth_width_min=self._widgets["tooth_width_min"].value() if is_slotted else 0,
                slot_wedge_height=self._widgets["slot_wedge_height"].value() if is_slotted else 0,
                winding_layers=self._widgets["winding_layers"].value() if is_slotted else 0,
                turns_per_slot=self._widgets["turns_per_slot"].value(),
                conductor_diameter=self._widgets["conductor_diameter"].value(),
                fill_factor=self._widgets["fill_factor"].value(),
                num_poles=self._widgets["num_poles"].value(),
                magnet_thickness=self._widgets["magnet_thickness"].value(),
                magnet_span_ratio=self._widgets["magnet_span_ratio"].value(),
                magnet_grade=self._widgets["magnet_grade"].currentText(),
                magnet_max_temp=magnet_max_temp,
                housing_wall_thickness=self._widgets["housing_wall_thickness"].value(),
                housing_finned=self._widgets["housing_finned"].isChecked(),
                shaft_radius=self._widgets["Rri"].value(),
                insulation_class=self._widgets["insulation_class"].currentText(),
            )
            return geo
        except Exception as e:
            self._info_display.setText(f"Parameter error: {e}")
            return None

    def _on_grade_changed(self, grade_name):
        """Update max temp display when magnet grade changes."""
        mat = get_magnet_for_grade(grade_name)
        max_temp = mat.max_temp if mat else 150
        self._widgets["magnet_max_temp"].setText(f"{max_temp:.0f} °C")
        self._on_param_change()

    def _on_param_change(self):
        """Called when any parameter changes."""
        try:
            geo = self.get_parameters()
            if geo is not None:
                self._geo = geo
                self._update_info_display(geo)
                if self.paramChanged:
                    self.paramChanged(geo)
        except Exception:
            pass  # Silently ignore transient errors during rapid editing

    def _update_info_display(self, geo):
        """Update the derived quantities display."""
        try:
            data = get_geometry_data(geo)
            html = "<table>"
            for k, v in data.items():
                html += f"<tr><td><b>{k}</b>:</td><td>{v}</td></tr>"
            html += "</table>"
            self._info_display.setHtml(html)
        except Exception as e:
            self._info_display.setText(f"Computation error: {e}")

    def _reset_defaults(self):
        """Reset all parameters to defaults."""
        default = MotorGeometryParams()
        for key, widget in self._widgets.items():
            if hasattr(default, key):
                val = getattr(default, key)
                if isinstance(widget, QDoubleSpinBox):
                    widget.blockSignals(True)
                    widget.setValue(float(val))
                    widget.blockSignals(False)
                elif isinstance(widget, QSpinBox):
                    widget.blockSignals(True)
                    widget.setValue(int(val))
                    widget.blockSignals(False)
                elif isinstance(widget, QComboBox):
                    widget.blockSignals(True)
                    idx = widget.findText(str(val))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                    widget.blockSignals(False)
                elif isinstance(widget, QCheckBox):
                    widget.blockSignals(True)
                    widget.setChecked(bool(val))
                    widget.blockSignals(False)
        # Trigger grade change to update max temp
        self._on_grade_changed(self._widgets["magnet_grade"].currentText())
        self._on_param_change()

    @property
    def current_geometry(self) -> MotorGeometryParams:
        return self._geo

    def apply_geometry(self, geo: MotorGeometryParams):
        """Set all widget values from a MotorGeometryParams object."""
        # Block signals during bulk update
        for key, widget in self._widgets.items():
            if hasattr(geo, key):
                val = getattr(geo, key)
                widget.blockSignals(True)
                if isinstance(widget, QDoubleSpinBox) or isinstance(widget, QSpinBox):
                    widget.setValue(float(val) if isinstance(val, float) else int(val))
                elif isinstance(widget, QComboBox):
                    idx = widget.findText(str(val))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(val))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(val))
                widget.blockSignals(False)
        # Trigger grade update for max temp
        self._on_grade_changed(self._widgets["magnet_grade"].currentText())
        self._on_param_change()


class MaterialInputPanel(QWidget):
    """Panel for material property assignments."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sw = QWidget()
        sl = QFormLayout(sw)

        # Material assignments for each component
        self._assignments = {}
        components = [
            "stator_core", "rotor_core", "magnet", "winding",
            "housing", "shaft", "slot_liner", "impregnation"
        ]
        labels = {
            "stator_core": "Stator Core",
            "rotor_core": "Rotor Core",
            "magnet": "Magnet",
            "winding": "Winding",
            "housing": "Housing",
            "shaft": "Shaft",
            "slot_liner": "Slot Liner",
            "impregnation": "Impregnation"
        }

        # Get relevant materials for each category
        steel_mats = [k for k in MATERIAL_CATALOG.keys() if "Steel" in k or "M19" in k or "M400" in k or "NO20" in k]
        magnet_mats = list(MAGNET_GRADES.keys())
        housing_mats = ["Al6061", "ADC12", "Steel_Struct"]
        winding_mats = ["Copper", "Winding_Eq"]

        defaults = {
            "stator_core": steel_mats[0] if steel_mats else "M19_24Ga",
            "rotor_core": steel_mats[0] if steel_mats else "M19_24Ga",
            "magnet": "N35SH",
            "winding": "Winding_Eq",
            "housing": "Al6061",
            "shaft": "Shaft_Steel",
            "slot_liner": "Slot_Liner",
            "impregnation": "Varnish",
        }

        for comp in components:
            gb = QGroupBox(labels[comp])
            gl = QFormLayout(gb)

            cb = QComboBox()
            if comp == "magnet":
                cb.addItems(magnet_mats)
            elif comp == "housing":
                cb.addItems(housing_mats)
            elif comp in ("stator_core", "rotor_core"):
                cb.addItems(steel_mats + ["Steel_Struct"])
            elif comp == "winding":
                cb.addItems(winding_mats)
            else:
                cb.addItems(list(MATERIAL_CATALOG.keys()))

            idx = cb.findText(defaults.get(comp, ""))
            if idx >= 0:
                cb.setCurrentIndex(idx)

            gl.addRow("Material:", cb)

            # Show material properties
            info = QLabel()
            info.setWordWrap(True)
            info.setStyleSheet("font-size: 9pt; color: #555;")
            gl.addRow(info)

            cb.currentTextChanged.connect(lambda txt, lbl=info: self._update_mat_info(lbl, txt))

            self._assignments[comp] = {"combo": cb, "info": info}
            sl.addWidget(gb)

        self._update_all_info()
        scroll.setWidget(sw)
        layout.addWidget(scroll)

    def _update_mat_info(self, label, mat_name):
        mat = get_material(mat_name)
        if mat:
            label.setText(
                f"k: {mat.k_radial:.1f} W/m·K (r) / {mat.k_axial:.1f} (z)\n"
                f"ρ: {mat.rho:.0f} kg/m³ | cp: {mat.cp:.0f} J/kg·K\n"
                f"Tmax: {mat.max_temp:.0f}°C"
            )

    def _update_all_info(self):
        for comp, widgets in self._assignments.items():
            self._update_mat_info(widgets["info"], widgets["combo"].currentText())

    def get_material_assignments(self):
        """Return dict of component -> material name."""
        return {comp: w["combo"].currentText() for comp, w in self._assignments.items()}


class LPTNInputPanel(QWidget):
    """
    Input panel for Lumped Parameter Thermal Network configuration.
    Provides all fields from NetworkBuilderConfig as interactive widgets.
    """

    runRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(5, 5, 5, 5)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.setSpacing(6)

        # ============================================================
        # 1. Cooling Configuration
        # ============================================================
        cool_group = QGroupBox("Cooling Configuration")
        cool_layout = QFormLayout(cool_group)

        self._cb_cooling_mode = QComboBox()
        self._cb_cooling_mode.addItems(["TENV (Natural)", "TEFC (Fan)", "Water Jacket"])
        cool_layout.addRow("Cooling Mode:", self._cb_cooling_mode)

        self._sb_ambient = QDoubleSpinBox()
        self._sb_ambient.setRange(-50, 200)
        self._sb_ambient.setValue(40.0)
        self._sb_ambient.setSuffix(" C")
        self._sb_ambient.setButtonSymbols(QAbstractSpinBox.NoButtons)
        cool_layout.addRow("Ambient Temp:", self._sb_ambient)

        self._sb_air_speed = QDoubleSpinBox()
        self._sb_air_speed.setRange(0, 20)
        self._sb_air_speed.setValue(3.0)
        self._sb_air_speed.setSuffix(" m/s")
        self._sb_air_speed.setButtonSymbols(QAbstractSpinBox.NoButtons)
        cool_layout.addRow("Air Speed:", self._sb_air_speed)

        self._sb_flow_rate = QDoubleSpinBox()
        self._sb_flow_rate.setRange(0, 100)
        self._sb_flow_rate.setValue(10.0)
        self._sb_flow_rate.setSuffix(" L/min")
        self._sb_flow_rate.setButtonSymbols(QAbstractSpinBox.NoButtons)
        cool_layout.addRow("Coolant Flow:", self._sb_flow_rate)

        self._sb_coolant_temp = QDoubleSpinBox()
        self._sb_coolant_temp.setRange(0, 100)
        self._sb_coolant_temp.setValue(50.0)
        self._sb_coolant_temp.setSuffix(" C")
        self._sb_coolant_temp.setButtonSymbols(QAbstractSpinBox.NoButtons)
        cool_layout.addRow("Coolant Temp:", self._sb_coolant_temp)

        sl.addWidget(cool_group)

        # ============================================================
        # 2. Operating Conditions
        # ============================================================
        op_group = QGroupBox("Operating Conditions")
        op_layout = QFormLayout(op_group)

        self._sb_speed = QDoubleSpinBox()
        self._sb_speed.setRange(0, 100000)
        self._sb_speed.setValue(3000)
        self._sb_speed.setSuffix(" RPM")
        self._sb_speed.setButtonSymbols(QAbstractSpinBox.NoButtons)
        op_layout.addRow("Speed:", self._sb_speed)

        sl.addWidget(op_group)

        # ============================================================
        # 3. Loss Inputs
        # ============================================================
        loss_group = QGroupBox("Loss Inputs [W]")
        loss_grid = QGridLayout(loss_group)
        loss_grid.setSpacing(4)

        self._loss_widgets = {}
        loss_labels = [
            ("copper_slot", "Copper (slot)", 60.0, 0, 5000),
            ("copper_end", "Copper (end)", 20.0, 0, 5000),
            ("iron_yoke", "Iron (yoke)", 15.0, 0, 5000),
            ("iron_teeth", "Iron (teeth)", 10.0, 0, 5000),
            ("magnet", "Magnet", 5.0, 0, 2000),
            ("mechanical", "Mechanical", 5.0, 0, 2000),
        ]
        for idx, (key, label, default, vmin, vmax) in enumerate(loss_labels):
            lbl = QLabel(label)
            sb = QDoubleSpinBox()
            sb.setRange(vmin, vmax)
            sb.setValue(default)
            sb.setSuffix(" W")
            sb.setDecimals(1)
            sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
            self._loss_widgets[key] = sb
            loss_grid.addWidget(lbl, idx, 0)
            loss_grid.addWidget(sb, idx, 1)

        sl.addWidget(loss_group)

        # ============================================================
        # 4. Dimensionality & Sector
        # ============================================================
        dim_group = QGroupBox("Model Options")
        dim_layout = QFormLayout(dim_group)

        self._cb_dimensionality = QComboBox()
        self._cb_dimensionality.addItems(["2D (Radial)", "3D (Radial+Axial)"])
        dim_layout.addRow("Dimensionality:", self._cb_dimensionality)

        self._cb_use_sector = QCheckBox("Use sector symmetry (1/N pole)")
        dim_layout.addRow("", self._cb_use_sector)

        self._sb_sector_n = QSpinBox()
        self._sb_sector_n.setRange(1, 32)
        self._sb_sector_n.setValue(1)
        self._sb_sector_n.setSuffix(" poles")
        self._sb_sector_n.setButtonSymbols(QAbstractSpinBox.NoButtons)
        dim_layout.addRow("Sector poles:", self._sb_sector_n)

        sl.addWidget(dim_group)

        # ============================================================
        # 5. Solver Settings
        # ============================================================
        solver_group = QGroupBox("Solver")
        solver_layout = QFormLayout(solver_group)

        self._sb_max_iter = QSpinBox()
        self._sb_max_iter.setRange(10, 500)
        self._sb_max_iter.setValue(50)
        self._sb_max_iter.setButtonSymbols(QAbstractSpinBox.NoButtons)
        solver_layout.addRow("Max iterations:", self._sb_max_iter)

        self._sb_tolerance = QDoubleSpinBox()
        self._sb_tolerance.setRange(0.01, 10)
        self._sb_tolerance.setValue(0.1)
        self._sb_tolerance.setSuffix(" C")
        self._sb_tolerance.setButtonSymbols(QAbstractSpinBox.NoButtons)
        solver_layout.addRow("Convergence:", self._sb_tolerance)

        self._sb_relaxation = QDoubleSpinBox()
        self._sb_relaxation.setRange(0.1, 1.0)
        self._sb_relaxation.setValue(0.5)
        self._sb_relaxation.setSingleStep(0.1)
        self._sb_relaxation.setButtonSymbols(QAbstractSpinBox.NoButtons)
        solver_layout.addRow("Relaxation:", self._sb_relaxation)

        sl.addWidget(solver_group)

        # ============================================================
        # 6. Run Button
        # ============================================================
        btn_layout = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run LPTN Simulation")
        self._btn_run.setMinimumHeight(40)
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #2d7d46; color: white; font-weight: bold; "
            "border-radius: 6px; padding: 8px; }"
            "QPushButton:hover { background-color: #3a9c5a; }"
        )
        self._btn_run.clicked.connect(self._on_run)
        btn_layout.addWidget(self._btn_run)
        sl.addLayout(btn_layout)

        sl.addStretch()
        scroll.setWidget(sw)
        layout.addWidget(scroll)

    def _on_run(self):
        self.runRequested.emit()

    def get_config(self) -> NetworkBuilderConfig:
        """Read widgets and return a NetworkBuilderConfig."""
        cfg = NetworkBuilderConfig()

        mode = self._cb_cooling_mode.currentText()
        if mode == "TENV (Natural)":
            cfg.cooling_mode = "TENV"
        elif mode == "TEFC (Fan)":
            cfg.cooling_mode = "TEFC"
        else:
            cfg.cooling_mode = "Water Jacket"

        cfg.ambient_temperature = self._sb_ambient.value()
        cfg.housing_air_speed = self._sb_air_speed.value()
        cfg.coolant_flow_rate = self._sb_flow_rate.value()
        cfg.coolant_temperature = self._sb_coolant_temp.value()
        cfg.speed_rpm = self._sb_speed.value()

        cfg.dimensionality = "3D" if "3D" in self._cb_dimensionality.currentText() else "2D"
        cfg.use_sector = self._cb_use_sector.isChecked()
        cfg.sector_n_poles = self._sb_sector_n.value()

        cfg.loss_copper_slot = self._loss_widgets["copper_slot"].value()
        cfg.loss_copper_end = self._loss_widgets["copper_end"].value()
        cfg.loss_iron_yoke = self._loss_widgets["iron_yoke"].value()
        cfg.loss_iron_teeth = self._loss_widgets["iron_teeth"].value()
        cfg.loss_magnet = self._loss_widgets["magnet"].value()
        cfg.loss_mechanical = self._loss_widgets["mechanical"].value()

        return cfg

    def apply_config(self, cfg: NetworkBuilderConfig):
        """Set all widgets from a NetworkBuilderConfig object."""
        self._cb_cooling_mode.blockSignals(True)
        mode_text = cfg.cooling_mode
        if mode_text == "TENV":
            self._cb_cooling_mode.setCurrentText("TENV (Natural)")
        elif mode_text == "TEFC":
            self._cb_cooling_mode.setCurrentText("TEFC (Fan)")
        else:
            self._cb_cooling_mode.setCurrentText("Water Jacket")
        self._cb_cooling_mode.blockSignals(False)

        self._sb_ambient.setValue(cfg.ambient_temperature)
        self._sb_air_speed.setValue(cfg.housing_air_speed)
        self._sb_flow_rate.setValue(cfg.coolant_flow_rate)
        self._sb_coolant_temp.setValue(cfg.coolant_temperature)
        self._sb_speed.setValue(cfg.speed_rpm)

        if cfg.dimensionality == "3D":
            self._cb_dimensionality.setCurrentText("3D (Radial+Axial)")
        else:
            self._cb_dimensionality.setCurrentText("2D (Radial)")

        self._cb_use_sector.setChecked(cfg.use_sector)
        self._sb_sector_n.setValue(cfg.sector_n_poles)

        self._loss_widgets["copper_slot"].setValue(cfg.loss_copper_slot)
        self._loss_widgets["copper_end"].setValue(cfg.loss_copper_end)
        self._loss_widgets["iron_yoke"].setValue(cfg.loss_iron_yoke)
        self._loss_widgets["iron_teeth"].setValue(cfg.loss_iron_teeth)
        self._loss_widgets["magnet"].setValue(cfg.loss_magnet)
        self._loss_widgets["mechanical"].setValue(cfg.loss_mechanical)


class ThermalNetworkCanvas(FigureCanvas):
    """
    Matplotlib canvas that draws a schematic thermal network diagram.
    Nodes with temperatures, resistances with R values, heat sources with W.
    """
    def __init__(self, parent=None, width=9, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor("#FAFAFA")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._net = None

    def draw_network(self, net: ThermalNetwork):
        """Draw a schematic diagram of the thermal network."""
        from matplotlib.patches import Circle, Patch
        from collections import defaultdict
        # ... rest of method ...


class ResultPanel(QWidget):
    """
    Panel showing detailed LPTN model results (in the Results tab).
    Displays:
      - Network schematic diagram
      - Node table: name, temp, loss, volume, capacitance
      - Resistance table: name, from-to, R, type, length, area, k, h
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._net = None
        self._undock_window = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Network info at top
        self._net_info = QLabel("No simulation data. Run LPTN first.")
        self._net_info.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 2px;")
        layout.addWidget(self._net_info)

        # Undock button
        btn_layout = QHBoxLayout()
        self._btn_undock = QPushButton("Undock Network Diagram")
        self._btn_undock.clicked.connect(self._undock_network)
        self._btn_undock.setEnabled(False)
        self._btn_undock.setStyleSheet(
            "QPushButton { background-color: #2c3e50; color: white; "
            "border-radius: 4px; padding: 4px 12px; font-size: 9pt; }"
            "QPushButton:hover { background-color: #34495e; }"
            "QPushButton:disabled { background-color: #cccccc; color: #888888; }"
        )
        btn_layout.addWidget(self._btn_undock)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Network schematic canvas
        self._net_canvas = ThermalNetworkCanvas(self, width=9, height=4, dpi=100)
        self._net_canvas.setMinimumHeight(280)
        layout.addWidget(self._net_canvas)

        # Node properties table
        node_group = QGroupBox("Node Properties")
        node_layout = QVBoxLayout(node_group)
        self._node_table = QTableWidget()
        self._node_table.setColumnCount(6)
        self._node_table.setHorizontalHeaderLabels([
            "Node", "T [C]", "Loss [W]", "Volume [cm3]", "Capacitance [J/K]", "Temp Dep."
        ])
        self._node_table.setAlternatingRowColors(True)
        self._node_table.horizontalHeader().setStretchLastSection(True)
        node_layout.addWidget(self._node_table)
        layout.addWidget(node_group)

        # Resistance properties table
        res_group = QGroupBox("Thermal Resistances")
        res_layout = QVBoxLayout(res_group)
        self._res_table = QTableWidget()
        self._res_table.setColumnCount(7)
        self._res_table.setHorizontalHeaderLabels([
            "Name", "From -> To", "R [K/W]", "Type", "Length [mm]", "Area [mm2]", "k/h [W/mK]"
        ])
        self._res_table.setAlternatingRowColors(True)
        self._res_table.horizontalHeader().setStretchLastSection(True)
        res_layout.addWidget(self._res_table)
        layout.addWidget(res_group)

    def display_network(self, net: ThermalNetwork, config: NetworkBuilderConfig = None):
        """Fill tables with network data."""
        self._net = net
        conv = getattr(net, 'solver_converged', None)
        iters = getattr(net, 'solver_iterations', None)
        info = f"Network: {net.name} | {len(net.nodes)} nodes, {len(net.resistances)} resistances"
        if iters:
            info += f" | {iters} iterations"
            info += " | Converged" if conv else " | Not converged"
        self._net_info.setText(info)

        # Enable undock button
        self._btn_undock.setEnabled(True)

        # Draw the schematic
        self._net_canvas.draw_network(net)

        # ---- Node table ----
        nodes = net.nodes
        self._node_table.setRowCount(len(nodes))
        for i, node in enumerate(nodes):
            self._node_table.setItem(i, 0, QTableWidgetItem(node.name))
            
            t_item = QTableWidgetItem(f"{node.temperature:.1f}")
            t_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._node_table.setItem(i, 1, t_item)
            
            p_item = QTableWidgetItem(f"{node.effective_loss:.2f}")
            p_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._node_table.setItem(i, 2, p_item)
            
            v_item = QTableWidgetItem(f"{node.volume * 1e6:.1f}")
            v_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._node_table.setItem(i, 3, v_item)
            
            c_item = QTableWidgetItem(f"{node.capacitance:.1f}")
            c_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._node_table.setItem(i, 4, c_item)
            
            td_item = QTableWidgetItem(
                "Yes" if node.loss_temperature_dependent else "No"
            )
            td_item.setTextAlignment(Qt.AlignCenter)
            self._node_table.setItem(i, 5, td_item)

        self._node_table.resizeColumnsToContents()

        # ---- Resistance table ----
        reses = net.resistances
        self._res_table.setRowCount(len(reses))
        for i, res in enumerate(reses):
            self._res_table.setItem(i, 0, QTableWidgetItem(res.name))
            
            # Get node names for the indices
            from_name = net.nodes[res.node_from].name if res.node_from < len(net.nodes) else f"n{res.node_from}"
            to_name = net.nodes[res.node_to].name if res.node_to < len(net.nodes) else f"n{res.node_to}"
            conn_item = QTableWidgetItem(f"{from_name} -> {to_name}")
            self._res_table.setItem(i, 1, conn_item)
            
            r_item = QTableWidgetItem(f"{res.effective_resistance:.4f}")
            r_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._res_table.setItem(i, 2, r_item)
            
            self._res_table.setItem(i, 3, QTableWidgetItem(res.resistance_type))
            
            len_item = QTableWidgetItem(f"{res.effective_length * 1000:.2f}")
            len_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._res_table.setItem(i, 4, len_item)
            
            area_item = QTableWidgetItem(f"{res.effective_area * 1e6:.2f}")
            area_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._res_table.setItem(i, 5, area_item)
            
            # Show k (for conduction) or h (for convection)
            if res.resistance_type == "convection" and res.h_coefficient > 0:
                kh_item = QTableWidgetItem(f"h={res.h_coefficient:.1f}")
            elif res.conductivity > 0:
                kh_item = QTableWidgetItem(f"k={res.conductivity:.3f}")
            else:
                kh_item = QTableWidgetItem("-")
            kh_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._res_table.setItem(i, 6, kh_item)

        self._res_table.resizeColumnsToContents()

    def _undock_network(self):
        """Open a separate window with the full-size network diagram."""
        if not self._net:
            return
        from PyQt5.QtWidgets import QMainWindow as QMWindow
        from PyQt5.QtCore import Qt as Qtt
        self._undock_window = QMWindow()
        self._undock_window.setWindowTitle("Thermal Network Diagram")
        self._undock_window.setMinimumSize(1000, 700)
        
        central = QWidget()
        layout = QVBoxLayout(central)
        
        # Larger canvas for the undocked view
        canvas = ThermalNetworkCanvas(self._undock_window, width=12, height=7, dpi=100)
        canvas.draw_network(self._net)
        layout.addWidget(canvas)
        
        # Close button
        close_btn = QPushButton("Close Window")
        close_btn.clicked.connect(self._undock_window.close)
        layout.addWidget(close_btn)
        
        self._undock_window.setCentralWidget(central)
        self._undock_window.show()
class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self._geo = MotorGeometryParams()
        self._lptn_results = None  # Store for annotation on geometry
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self):
        self.setWindowTitle("Motor Thermal Modeler - Geometry & Simulation Setup")
        self.setMinimumSize(1400, 900)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ============================================================
        # Left Panel: Input Tabs
        # ============================================================
        left_panel = QWidget()
        left_panel.setMinimumWidth(380)
        left_panel.setMaximumWidth(500)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)

        # Tab 1: Geometry
        self._geo_panel = GeometryInputPanel()
        self._tabs.addTab(self._geo_panel, "📐 Geometry")

        # Tab 2: Materials (placeholder for now)
        self._mat_panel = MaterialInputPanel()
        self._tabs.addTab(self._mat_panel, "🧪 Materials")

        # Tab 3: LPTN Thermal Simulation
        self._lptn_panel = LPTNInputPanel()
        self._tabs.addTab(self._lptn_panel, "🔥 LPTN")

        # Tab 4: Model Results
        self._result_panel = ResultPanel()
        self._tabs.addTab(self._result_panel, "📊 Results")

        left_layout.addWidget(self._tabs)

                # ============================================================
        # Right Panel: Geometry CAD (always visible)
        # ============================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)

        # Geometry CAD canvas
        self._canvas = MotorGeometryCanvas(self, width=9, height=8, dpi=100)

        # Controls below CAD
        controls_layout = QHBoxLayout()
        self._cb_labels = QCheckBox("Show Labels")
        self._cb_labels.setChecked(True)
        self._cb_labels.stateChanged.connect(self._redraw)
        self._cb_dims = QCheckBox("Show Dimensions")
        self._cb_dims.setChecked(True)
        self._cb_dims.stateChanged.connect(self._redraw)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._redraw)
        controls_layout.addWidget(self._cb_labels)
        controls_layout.addWidget(self._cb_dims)
        controls_layout.addWidget(self._btn_refresh)
        controls_layout.addStretch()

        # Status info label
        self._status_label = QLabel("Ready. Adjust parameters to update geometry.")
        self._status_label.setStyleSheet("font-size: 9pt; color: #666;")

        right_layout.addWidget(self._canvas)
        right_layout.addLayout(controls_layout)
        right_layout.addWidget(self._status_label)

        # ============================================================
        # Splitter between left and right
        # ============================================================
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)  # Left doesn't stretch
        splitter.setStretchFactor(1, 1)  # Right stretches
        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Import Parameters...", self._import_params)
        file_menu.addAction("Export Parameters...", self._export_params)
        file_menu.addSeparator()
        file_menu.addAction("Export Gmsh .geo...", self._export_gmsh)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        view_menu = menubar.addMenu("View")
        view_menu.addAction("Reset View", self._redraw)

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", lambda: QMessageBox.about(self,
            "About", "Motor Thermal Modeler v0.1\n\n"
            "Parametric geometry GUI for radial-flux motors.\n"
            "Supports Lumped Parameter Thermal Network and FEA paths."))

    def _connect_signals(self):
        self._geo_panel.paramChanged = self._on_geo_changed
        self._lptn_panel.runRequested.connect(self._run_lptn)
        QTimer.singleShot(100, self._redraw)

    def _on_geo_changed(self, geo: MotorGeometryParams):
        """Called when geometry parameters change."""
        self._geo = geo
        self._redraw()

    def _geo_to_v2(self) -> 'MotorGeometry':
        """
        Convert MotorGeometryParams (v1) to MotorGeometry (v2) for LPTN.
        Bridges the old GUI params to the new SlotGeometry-based format.
        """
        from geometry.motor_geometry_v2 import MotorGeometry as MotorGeometryV2
        from geometry.slot_geometry import SlotGeometry, slot_from_simple_params

        is_slotted = self._geo.structure_type in ("Slotted (distributed)", "Slotted (concentrated)")

        if is_slotted and self._geo.num_slots > 0:
            # Create slot geometry from simple parameters
            slot = slot_from_simple_params(
                Rsi=self._geo.Rsi,
                Rso=self._geo.Rso,
                num_slots=self._geo.num_slots,
                slot_opening=self._geo.slot_opening,
                slot_depth=self._geo.slot_depth,
                tooth_width_min=self._geo.tooth_width_min,
                wedge_height=self._geo.slot_wedge_height,
            )
        else:
            slot = SlotGeometry()  # default placeholder for slotless

        geo_v2 = MotorGeometryV2(
            structure_type="Slotted" if is_slotted else "Slotless",
            Rso=self._geo.Rso,
            Rsi=self._geo.Rsi,
            stack_length=self._geo.stack_length,
            Rro=self._geo.Rro,
            Rri=self._geo.Rri,
            airgap_length=self._geo.airgap_length,
            num_slots=self._geo.num_slots if is_slotted else 0,
            slot=slot if is_slotted else SlotGeometry(),
            num_poles=self._geo.num_poles,
            magnet_thickness=self._geo.magnet_thickness,
            magnet_span_ratio=self._geo.magnet_span_ratio,
            shaft_radius=self._geo.Rri,
            housing_wall_thickness=self._geo.housing_wall_thickness,
            fill_factor=self._geo.fill_factor if is_slotted else 0.50,
            winding_inner_radius=self._geo.Rsi,
            winding_outer_radius=self._geo.Rsi + self._geo.slot_depth if self._geo.slot_depth > 0 else self._geo.Rsi + 4.0,
        )
        return geo_v2

    def _run_lptn(self):
        """Run the LPTN simulation and update results."""
        try:
            geo_v2 = self._geo_to_v2()
            lptn_cfg = self._lptn_panel.get_config()
            net = build_thermal_network(geo_v2, lptn_cfg)
            T = solve_steady_state(net)
            self._display_lptn_results(net)
            self._result_panel.display_network(net, lptn_cfg)

            converged = getattr(net, 'solver_converged', True)
            iterations = getattr(net, 'solver_iterations', 1)
            max_temp = getattr(net, 'solver_max_temp', 0)
            status = "Converged" if converged else "Thermal runaway"
            self.statusBar().showMessage(
                f"LPTN: {status} | {iterations} iters | T_max={max_temp:.0f}C"
            )
        except Exception as e:
            QMessageBox.critical(self, "LPTN Error", f"Simulation failed:\n{e}")
            self.statusBar().showMessage(f"LPTN error: {e}")

    def _display_lptn_results(self, net: ThermalNetwork):
        """Fill the results table and update the Results tab with network schematic."""
        self._lptn_results = {}
        nodes = net.nodes
        n = len(nodes)
        
        # Build a compact results dict for status display
        for i, node in enumerate(nodes):
            self._lptn_results[node.name] = (node.temperature, node.effective_loss)

        # Push everything to the Results tab
        self._result_panel.display_network(net, self._lptn_panel.get_config())

    def _redraw(self):
        """Redraw the geometry plot, with temperature annotations if available."""
        try:
            self._canvas.draw_cross_section(
                self._geo,
                show_labels=self._cb_labels.isChecked(),
                show_dimensions=self._cb_dims.isChecked(),
                node_temperatures=self._lptn_results,
            )
            # Build status string
            parts = [
                f"Geometry: {self._geo.structure_type}",
                f"{self._geo.num_poles} poles",
                f"{self._geo.num_slots if self._geo.structure_type != 'Slotless' else '—'} slots",
                f"Rso={self._geo.Rso:.1f}",
                f"Lstk={self._geo.stack_length:.1f}",
            ]
            if self._lptn_results:
                # Find the hot component
                hot_item = max(self._lptn_results.items(), key=lambda x: x[1][0])
                parts.append(f"Hotspot: {hot_item[0]}={hot_item[1][0]:.0f}C")
            self._status_label.setText(" | ".join(parts))
        except Exception as e:
            self._status_label.setText(f"❌ Draw error: {e}")
            self.statusBar().showMessage(f"Error: {e}")

    def _import_params(self):
        """Import geometry + LPTN parameters from a JSON file."""
        fname, _ = QFileDialog.getOpenFileName(
            self, "Import Parameters", "",
            "JSON files (*.json);;Text files (*.txt);;All files (*)")
        if fname:
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    content = f.read()
                import json
                data = json.loads(content)

                # Check if it's the new combined format (has "geometry" and "lptn" keys)
                if "geometry" in data:
                    new_geo = MotorGeometryParams(**data["geometry"])
                    # Apply LPTN config if present
                    if "lptn" in data:
                        lptn_cfg = NetworkBuilderConfig.from_dict(data["lptn"])
                        self._lptn_panel.apply_config(lptn_cfg)
                else:
                    # Legacy JSON (flat keys) or text format
                    if fname.endswith(".json"):
                        new_geo = MotorGeometryParams(**data)
                    else:
                        # Legacy text format: parse "key: value" lines
                        geo_data = {}
                        for line in content.splitlines():
                            line = line.strip()
                            if ":" not in line or line.startswith("=") or line.startswith("#"):
                                continue
                            key, _, val = line.partition(":")
                            key = key.strip()
                            val = val.strip()
                            if key in ("", "_computed"):
                                continue
                            default_val = getattr(MotorGeometryParams(), key, None)
                            if default_val is None:
                                continue
                            if isinstance(default_val, bool):
                                geo_data[key] = val.lower() == "true"
                            elif isinstance(default_val, int):
                                geo_data[key] = int(float(val))
                            elif isinstance(default_val, float):
                                geo_data[key] = float(val)
                            else:
                                geo_data[key] = val
                        new_geo = MotorGeometryParams(**geo_data)

                # Apply to the panel widgets
                self._geo_panel.apply_geometry(new_geo)
                self._geo = new_geo
                self._redraw()
                self.statusBar().showMessage(f"Imported: {fname}")
            except Exception as e:
                QMessageBox.warning(self, "Import Error", f"Failed to import:\n{e}")
                self.statusBar().showMessage("Import failed")

    def _export_params(self):
        """Export geometry + LPTN parameters to a JSON file."""
        fname, _ = QFileDialog.getSaveFileName(
            self, "Export Parameters", "motor_params.json",
            "JSON files (*.json);;All files (*)")
        if fname:
            try:
                import json
                export_data = {
                    "motor_thermal_modeler": "1.0",
                    "geometry": self._geo.as_dict(),
                    "lptn": self._lptn_panel.get_config().to_dict(),
                }
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2)
                self.statusBar().showMessage(f"Exported to {fname}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    def _export_gmsh(self):
        """Generate and export a Gmsh .geo file."""
        from geometry.gmsh_export import generate_gmsh_geo
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Gmsh .geo file", "motor_geometry.geo",
            "Gmsh files (*.geo);;All files (*)")
        if fname:
            try:
                mat_assignments = self._mat_panel.get_material_assignments()
                geo_code = generate_gmsh_geo(self._geo, mat_assignments)
                with open(fname, "w") as f:
                    f.write(geo_code)
                self.statusBar().showMessage(f"Exported .geo to {fname}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))


def main():
    """Entry point for the application."""
    app = QApplication(sys.argv)

    # Apply style
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(30, 30, 30))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
