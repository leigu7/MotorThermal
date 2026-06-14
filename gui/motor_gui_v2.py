"""
Main GUI Application (v2) for Motor Thermal Modeling.
Uses improved geometry with proper slot definitions (JMAG/Maxwell style)
and slotless support.
"""

import sys
import math
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QGroupBox, QFormLayout, QLabel, QLineEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QSplitter,
    QMessageBox, QTextEdit, QCheckBox, QGridLayout, QFrame,
    QStatusBar, QAction, QFileDialog, QSlider, QScrollArea,
    QRadioButton, QButtonGroup,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from geometry.motor_geometry_v2 import MotorGeometry, get_geometry_summary
from geometry.slot_geometry import SlotGeometry as SlotDef
from geometry.plotter_v3 import MotorGeometryCanvasV3 as MotorGeometryCanvasV2, NavigationToolbar, COLORS
from materials.material_db import (
    MATERIAL_CATALOG, MAGNET_GRADES, INSULATION_CLASSES,
)


class FloatInput(QDoubleSpinBox):
    """Spin box with label and unit for float parameters."""
    def __init__(self, label, default, vmin=0.0, vmax=500.0,
                 step=0.5, unit="mm", decimals=2):
        super().__init__()
        self.setRange(vmin, vmax)
        self.setSingleStep(step)
        self.setDecimals(decimals)
        self.setValue(default)
        if unit:
            self.setSuffix(f" {unit}")
        self.setToolTip(label)
        self._label = label

    def set_value_silent(self, val):
        self.blockSignals(True)
        self.setValue(val)
        self.blockSignals(False)


class GeometryPanel(QWidget):
    """Input panel for all geometry parameters."""

    paramChanged = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._geo = MotorGeometry()
        self._widgets = {}
        self._setup_ui()

    def _make_float(self, label, key, default, vmin=0, vmax=500,
                    step=0.5, unit="mm", decimals=2):
        sb = FloatInput(label, default, vmin, vmax, step, unit, decimals)
        sb.valueChanged.connect(self._on_change)
        self._widgets[key] = sb
        return sb

    def _make_int(self, label, key, default, vmin=0, vmax=200, step=1):
        sb = QSpinBox()
        sb.setRange(vmin, vmax)
        sb.setValue(default)
        sb.setSingleStep(step)
        sb.valueChanged.connect(self._on_change)
        self._widgets[key] = sb
        return sb

    def _make_combo(self, key, items, default_idx=0):
        cb = QComboBox()
        cb.addItems(items)
        cb.setCurrentIndex(default_idx)
        cb.currentTextChanged.connect(self._on_change)
        self._widgets[key] = cb
        return cb

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
        # 1. Motor Structure Type
        # ============================================================
        gb = QGroupBox("Motor Structure")
        gl = QVBoxLayout(gb)
        
        self._widgets["structure_type"] = QComboBox()
        self._widgets["structure_type"].addItems(["Slotted", "Slotless"])
        self._widgets["structure_type"].currentTextChanged.connect(self._on_structure_change)
        gl.addWidget(QLabel("Type:"))
        gl.addWidget(self._widgets["structure_type"])
        sl.addWidget(gb)

        # ============================================================
        # 2. Core Dimensions
        # ============================================================
        gb = QGroupBox("Core Dimensions")
        gl = QFormLayout(gb)
        for label, key, default, vmin, vmax, step, unit in [
            ("Stator Outer Radius", "Rso", 50.0, 10, 500, 1, "mm"),
            ("Stator Inner Radius", "Rsi", 30.0, 5, 490, 1, "mm"),
            ("Stack Length", "stack_length", 60.0, 5, 500, 1, "mm"),
            ("Rotor Outer Radius", "Rro", 29.0, 5, 480, 0.5, "mm"),
            ("Rotor Inner Radius", "Rri", 10.0, 3, 200, 0.5, "mm"),
            ("Airgap", "airgap_length", 1.0, 0.1, 10, 0.1, "mm"),
        ]:
            gl.addRow(label + ":", self._make_float(label, key, default, vmin, vmax, step, unit))
        sl.addWidget(gb)

        # ============================================================
        # 3. Slot Geometry (Slotted only - JMAG/Maxwell style)
        # ============================================================
        self._slot_group = QGroupBox("Slot Geometry (Slotted)")
        gl = QFormLayout(self._slot_group)

        for label, key, default, vmin, vmax, step, unit in [
            ("Nº Slots", "num_slots_int", 24, 3, 96, 1, ""),
            ("Opening Width (Bs0)", "Bs0", 2.0, 0.5, 20, 0.2, "mm"),
            ("Shoulder Width (Bs1)", "Bs1", 4.0, 0.5, 25, 0.5, "mm"),
            ("Bottom Width (Bs2)", "Bs2", 6.0, 0.5, 30, 0.5, "mm"),
            ("Opening Height (Hs0)", "Hs0", 0.5, 0.1, 5, 0.1, "mm"),
            ("Shoulder Height (Hs1)", "Hs1", 0.5, 0.1, 10, 0.1, "mm"),
            ("Slot Body Height (Hs2)", "Hs2", 9.0, 0.5, 50, 0.5, "mm"),
            ("Fillet Radius (Rs)", "Rs_fillet", 0.5, 0, 5, 0.1, "mm"),
        ]:
            if key == "num_slots_int":
                w = self._make_int(label, key, default, vmin, vmax, step)
            else:
                w = self._make_float(label, key, default, vmin, vmax, step, unit)
            gl.addRow(label + ":", w)

        # Tooth type
        self._widgets["tooth_type_combo"] = QComboBox()
        self._widgets["tooth_type_combo"].addItems(["Parallel tooth", "Parallel slot"])
        self._widgets["tooth_type_combo"].currentTextChanged.connect(self._on_change)
        gl.addRow("Tooth Type:", self._widgets["tooth_type_combo"])

        sl.addWidget(self._slot_group)

        # ============================================================
        # 4. Winding (Slotless) - shown only for Slotless
        # ============================================================
        self._slotless_winding_group = QGroupBox("Winding Region (Slotless)")
        gl = QFormLayout(self._slotless_winding_group)
        for label, key, default, vmin, vmax, step, unit in [
            ("Winding Inner Radius", "winding_inner_radius", 30.0, 5, 490, 0.5, "mm"),
            ("Winding Outer Radius", "winding_outer_radius", 34.0, 5, 495, 0.5, "mm"),
            ("Winding Layers", "winding_layers_int", 2, 1, 4, 1, ""),
            ("Conductor Diameter", "conductor_diameter", 1.0, 0.1, 5, 0.1, "mm"),
            ("Turns per Phase", "turns_per_slot_int", 20, 1, 200, 1, ""),
            ("Fill Factor", "fill_factor_float", 0.45, 0.2, 0.75, 0.01, ""),
        ]:
            if key in ("winding_layers_int", "turns_per_slot_int"):
                w = self._make_int(label, key, default, vmin, vmax, step)
            elif key == "fill_factor_float":
                w = self._make_float(label, key, default, vmin, vmax, step, "", 3)
            else:
                w = self._make_float(label, key, default, vmin, vmax, step, unit)
            gl.addRow(label + ":", w)

        sl.addWidget(self._slotless_winding_group)

        # ============================================================
        # 5. Magnet Configuration
        # ============================================================
        gb = QGroupBox("Magnet Configuration")
        gl = QFormLayout(gb)

        self._widgets["num_poles"] = QSpinBox()
        self._widgets["num_poles"].setRange(2, 64)
        self._widgets["num_poles"].setValue(8)
        self._widgets["num_poles"].setSingleStep(2)
        self._widgets["num_poles"].valueChanged.connect(self._on_change)
        gl.addRow("Number of Poles:", self._widgets["num_poles"])

        gl.addRow("Magnet Thickness:",
                   self._make_float("", "magnet_thickness", 4.0, 0.5, 30, 0.5))
        gl.addRow("Pole Arc Ratio:",
                   self._make_float("", "magnet_span_ratio", 0.82, 0.4, 0.95, 0.01, "", 2))
        gl.addRow("Magnet Grade:",
                   self._make_combo("magnet_grade", sorted(MAGNET_GRADES.keys()), 0))
        gl.addRow("Max Temp:",
                   self._make_float("", "magnet_max_temp", 150, 60, 350, 5, "°C"))
        sl.addWidget(gb)

        # ============================================================
        # 6. Housing & Shaft
        # ============================================================
        gb = QGroupBox("Housing & Shaft")
        gl = QFormLayout(gb)
        gl.addRow("Housing Wall Thickness:",
                   self._make_float("", "housing_wall_thickness", 5.0, 1, 50, 0.5))
        gl.addRow("Shaft Radius:",
                   self._make_float("", "shaft_radius", 10.0, 3, 100, 0.5))

        self._widgets["housing_finned"] = QCheckBox("Finned Housing")
        self._widgets["housing_finned"].stateChanged.connect(self._on_change)
        gl.addRow("", self._widgets["housing_finned"])
        sl.addWidget(gb)

        # ============================================================
        # 7. Insulation
        # ============================================================
        gb = QGroupBox("Insulation")
        gl = QFormLayout(gb)
        gl.addRow("Class:",
                   self._make_combo("insulation_class",
                                    sorted(INSULATION_CLASSES.keys(), reverse=True), 0))
        sl.addWidget(gb)

        # ============================================================
        # 8. Info display
        # ============================================================
        gb = QGroupBox("Derived Quantities")
        self._info_display = QTextEdit()
        self._info_display.setReadOnly(True)
        self._info_display.setMaximumHeight(180)
        gl2 = QVBoxLayout(gb)
        gl2.addWidget(self._info_display)
        sl.addWidget(gb)

        # Buttons
        btn_layout = QHBoxLayout()
        btn = QPushButton("🔄 Update Preview")
        btn.clicked.connect(self._on_change)
        btn_layout.addWidget(btn)
        btn = QPushButton("↺ Reset")
        btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(btn)
        sl.addLayout(btn_layout)

        sl.addStretch()
        scroll.setWidget(sw)
        layout.addWidget(scroll)

        # Initial visibility
        self._update_visibility()

    def _on_structure_change(self, text):
        """Handle structure type change."""
        self._update_visibility()
        self._on_change()

    def _update_visibility(self):
        """Show/hide slot vs slotless panels."""
        is_slotted = self._widgets["structure_type"].currentText() == "Slotted"
        self._slot_group.setVisible(is_slotted)
        self._slotless_winding_group.setVisible(not is_slotted)

    def get_geometry(self) -> Optional[MotorGeometry]:
        """Read all inputs and return a MotorGeometry."""
        try:
            stype = self._widgets["structure_type"].currentText()
            is_slotted = stype == "Slotted"

            # Common params
            kwargs = {
                "structure_type": stype,
                "Rso": self._widgets["Rso"].value(),
                "Rsi": self._widgets["Rsi"].value(),
                "stack_length": self._widgets["stack_length"].value(),
                "Rro": self._widgets["Rro"].value(),
                "Rri": self._widgets["Rri"].value(),
                "airgap_length": self._widgets["airgap_length"].value(),
                "num_poles": self._widgets["num_poles"].value(),
                "magnet_thickness": self._widgets["magnet_thickness"].value(),
                "magnet_span_ratio": self._widgets["magnet_span_ratio"].value(),
                "magnet_grade": self._widgets["magnet_grade"].currentText(),
                "magnet_max_temp": self._widgets["magnet_max_temp"].value(),
                "housing_wall_thickness": self._widgets["housing_wall_thickness"].value(),
                "shaft_radius": self._widgets["shaft_radius"].value(),
                "housing_finned": self._widgets["housing_finned"].isChecked(),
                "insulation_class": self._widgets["insulation_class"].currentText(),
            }

            if is_slotted:
                kwargs.update({
                    "num_slots": self._widgets["num_slots_int"].value(),
                    "conductor_diameter": self._widgets["conductor_diameter"].value(),
                    "turns_per_slot": self._widgets["turns_per_slot_int"].value(),
                    "fill_factor": self._widgets["fill_factor_float"].value(),
                })
                # Slot geometry
                kwargs["slot"] = SlotDef(
                    Hs0=self._widgets["Hs0"].value(),
                    Hs1=self._widgets["Hs1"].value(),
                    Hs2=self._widgets["Hs2"].value(),
                    Bs0=self._widgets["Bs0"].value(),
                    Bs1=self._widgets["Bs1"].value(),
                    Bs2=self._widgets["Bs2"].value(),
                    Rs_fillet=self._widgets["Rs_fillet"].value(),
                    tooth_type=self._widgets["tooth_type_combo"].currentText(),
                )
            else:
                kwargs.update({
                    "winding_inner_radius": self._widgets["winding_inner_radius"].value(),
                    "winding_outer_radius": self._widgets["winding_outer_radius"].value(),
                    "conductor_diameter": self._widgets["conductor_diameter"].value(),
                    "turns_per_slot": self._widgets["turns_per_slot_int"].value(),
                    "fill_factor": self._widgets["fill_factor_float"].value(),
                })

            return MotorGeometry(**kwargs)

        except Exception as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            return None

    def _on_change(self):
        """Called when any parameter changes."""
        geo = self.get_geometry()
        if geo is not None:
            self._geo = geo
            self._update_info(geo)
            if self.paramChanged:
                self.paramChanged(geo)

    def _update_info(self, geo):
        """Update derived quantities display."""
        try:
            info = get_geometry_summary(geo)
            html = "<table>"
            for k, v in info.items():
                html += f"<tr><td><b>{k}</b>:</td><td>{v}</td></tr>"
            html += "</table>"
            self._info_display.setHtml(html)
        except Exception as e:
            self._info_display.setText(f"Error: {e}")

    def _reset_defaults(self):
        """Reset to default values."""
        default = MotorGeometry()
        for key, widget in self._widgets.items():
            # Map widget keys to geometry field names
            field_key = key.replace("_int", "").replace("_float", "")
            if hasattr(default, field_key):
                val = getattr(default, field_key)
                if isinstance(widget, QDoubleSpinBox):
                    widget.set_value_silent(float(val) if val else 0)
                elif isinstance(widget, QSpinBox):
                    widget.blockSignals(True)
                    widget.setValue(int(val) if val else 0)
                    widget.blockSignals(False)
            # Handle slot sub-fields
            if hasattr(default, 'slot') and hasattr(default.slot, field_key):
                val = getattr(default.slot, field_key)
                if isinstance(widget, QDoubleSpinBox):
                    widget.set_value_silent(float(val))
        self._on_change()

    @property
    def current_geometry(self):
        return self._geo


class MaterialPanel(QWidget):
    """Material assignment panel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._assignments = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sw = QWidget()
        sl = QFormLayout(sw)

        components = [
            ("stator_core", "Stator Core"),
            ("rotor_core", "Rotor Core"),
            ("magnet", "Magnet"),
            ("winding", "Winding"),
            ("housing", "Housing"),
            ("shaft", "Shaft"),
            ("slot_liner", "Slot Liner"),
            ("impregnation", "Impregnation"),
        ]

        mat_groups = {
            "stator_core": ["M19_24Ga", "M400_50A", "NO20", "Steel_Struct"],
            "rotor_core": ["M19_24Ga", "M400_50A", "Steel_Struct"],
            "magnet": sorted(MAGNET_GRADES.keys()),
            "winding": ["Winding_Eq", "Copper"],
            "housing": ["Al6061", "ADC12", "Steel_Struct"],
            "shaft": ["Shaft_Steel", "Steel_Struct"],
            "slot_liner": ["Slot_Liner"],
            "impregnation": ["Varnish"],
        }

        for key, label in components:
            gb = QGroupBox(label)
            gl = QFormLayout(gb)

            cb = QComboBox()
            items = mat_groups.get(key, list(MATERIAL_CATALOG.keys()))
            cb.addItems(items)
            cb.setCurrentIndex(0)

            info = QLabel()
            info.setWordWrap(True)
            info.setStyleSheet("font-size: 9pt; color: #555;")

            gl.addRow("Material:", cb)
            gl.addRow(info)

            cb.currentTextChanged.connect(lambda txt, lbl=info: self._update_info(lbl, txt))

            self._assignments[key] = {"combo": cb, "info": info}
            sl.addWidget(gb)

        self._update_all_info()
        scroll.setWidget(sw)
        layout.addWidget(scroll)

    def _update_info(self, label, mat_name):
        from materials.material_db import get_material
        mat = get_material(mat_name)
        if mat:
            label.setText(
                f"k: {mat.k_radial:.1f} W/m·K (r) / {mat.k_axial:.1f} (z)\n"
                f"ρ: {mat.rho:.0f} kg/m³ | cp: {mat.cp:.0f} J/kg·K\n"
                f"Tmax: {mat.max_temp:.0f}°C"
            )

    def _update_all_info(self):
        for comp, w in self._assignments.items():
            self._update_info(w["info"], w["combo"].currentText())

    def get_assignments(self):
        return {comp: w["combo"].currentText() for comp, w in self._assignments.items()}


class MainWindowV2(QMainWindow):
    """Main application window (v2)."""

    def __init__(self):
        super().__init__()
        self._geo = MotorGeometry()
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self):
        self.setWindowTitle("Motor Thermal Modeler v2 - Geometry & Simulation Setup")
        self.setMinimumSize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # === Left Panel ===
        left_panel = QWidget()
        left_panel.setMinimumWidth(400)
        left_panel.setMaximumWidth(520)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)

        self._tabs = QTabWidget()

        # Geometry tab
        self._geo_panel = GeometryPanel()
        self._tabs.addTab(self._geo_panel, "📐 Geometry")

        # Materials tab
        self._mat_panel = MaterialPanel()
        self._tabs.addTab(self._mat_panel, "🧪 Materials")

        # Operating conditions (placeholder)
        op_panel = QWidget()
        opl = QVBoxLayout(op_panel)
        opl.addWidget(QLabel("⚡ Operating conditions & losses\n(Coming in next phase)"))
        opl.addStretch()
        self._tabs.addTab(op_panel, "⚡ Operating")

        # Simulation tab
        sim_panel = QWidget()
        siml = QVBoxLayout(sim_panel)

        gb = QGroupBox("Simulation Path")
        gbl = QVBoxLayout(gb)
        self._btn_lpt = QPushButton("🔧 Run Lumped Parameter Network")
        self._btn_lpt.setMinimumHeight(40)
        self._btn_fea = QPushButton("📐 FEA (Gmsh)")
        self._btn_fea.setMinimumHeight(40)
        gbl.addWidget(self._btn_lpt)
        gbl.addWidget(self._btn_fea)
        siml.addWidget(gb)

        gb2 = QGroupBox("Export")
        gb2l = QVBoxLayout(gb2)
        self._btn_export_params = QPushButton("📄 Export Parameters (.txt)")
        self._btn_export_geo = QPushButton("🔷 Export Gmsh (.geo)")
        self._btn_export_report = QPushButton("📊 Generate Report")
        gb2l.addWidget(self._btn_export_params)
        gb2l.addWidget(self._btn_export_geo)
        gb2l.addWidget(self._btn_export_report)
        siml.addWidget(gb2)
        siml.addStretch()
        self._tabs.addTab(sim_panel, "🚀 Simulation")

        left_layout.addWidget(self._tabs)

        # === Right Panel: Plot ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)

        self._canvas = MotorGeometryCanvasV2(self, width=9, height=9, dpi=100)

        # Navigation toolbar for zoom/pan/fit
        self._nav_toolbar = NavigationToolbar(self._canvas, self)
        self._nav_toolbar.setStyleSheet("font-size: 8pt;")

        controls = QHBoxLayout()
        self._cb_labels = QCheckBox("Labels")
        self._cb_labels.setChecked(True)
        self._cb_labels.stateChanged.connect(self._redraw)
        self._cb_dims = QCheckBox("Dimensions")
        self._cb_dims.setChecked(True)
        self._cb_dims.stateChanged.connect(self._redraw)
        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self._redraw)
        btn_fit = QPushButton("⊞ Fit")
        btn_fit.clicked.connect(self._fit_view)

        controls.addWidget(self._cb_labels)
        controls.addWidget(self._cb_dims)
        controls.addWidget(btn_refresh)
        controls.addWidget(btn_fit)
        controls.addStretch()

        self._status_label = QLabel("Ready. Adjust parameters to update geometry.")
        self._status_label.setStyleSheet("font-size: 9pt; color: #666;")

        right_layout.addWidget(self._nav_toolbar)
        right_layout.addWidget(self._canvas)
        right_layout.addLayout(controls)
        right_layout.addWidget(self._status_label)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.statusBar().showMessage("Ready")

    def _setup_menu(self):
        menubar = self.menuBar()
        fm = menubar.addMenu("File")
        fm.addAction("Export Parameters...", self._export_params)
        fm.addAction("Export Gmsh .geo...", self._export_gmsh)
        fm.addSeparator()
        fm.addAction("Quit", self.close)

        vm = menubar.addMenu("View")
        vm.addAction("Reset View", self._redraw)
        vm.addAction("Fit to Window", self._fit_view)

        hm = menubar.addMenu("Help")
        hm.addAction("Parameter Reference", self._show_param_ref)
        hm.addAction("About", lambda: QMessageBox.about(self,
            "About", "Motor Thermal Modeler v2\n\n"
            "Parametric geometry GUI for radial-flux motors.\n"
            "JMAG/Maxwell inspired slot geometry.\n"
            "Paths: Lumped Parameter Network & FEA (Gmsh)."))

    def _fit_view(self):
        """Reset axes to fit the geometry."""
        if self._geo:
            max_r = self._geo.Rso + self._geo.housing_wall_thickness + 12
            self._canvas.ax.set_xlim(-max_r, max_r)
            self._canvas.ax.set_ylim(-max_r, max_r)
            self._canvas.draw()

    def _show_param_ref(self):
        """Show the parameter definition reference in a scrollable dialog."""
        text = self._build_param_ref_text()
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Parameter Reference")
        dlg.setText(text)
        dlg.setDetailedText(
            "Tip: Use the navigation toolbar (Zoom, Pan, Home) to explore the geometry."
        )
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setMinimumWidth(500)
        dlg.exec_()

    def _build_param_ref_text(self):
        """Build parameter reference text."""
        return """CORE DIMENSIONS
  Rso ................ Stator Outer Radius [mm]
  Rsi ................ Stator Inner Radius (Bore) [mm]
  Lstk ............... Axial Stack Length [mm]
  Rro ................ Rotor Outer Radius [mm]
  Rri ................ Rotor Inner Radius [mm]
  Airgap ............. Mechanical airgap [mm]

SLOT GEOMETRY (JMAG/Maxwell Convention) — Only for Slotted type
  Bs0 ................ Slot Opening Width at bore [mm]
  Bs1 ................ Shoulder / Wedge Width [mm]
  Bs2 ................ Slot Bottom Width [mm]
  Hs0 ................ Opening Height (tooth tip) [mm]
  Hs1 ................ Shoulder / Wedge Height [mm]
  Hs2 ................ Slot Body Height [mm]
  Rs_fillet .......... Slot bottom fillet radius [mm]

  Slot taper: Bs0 (bore) -> Bs1 -> Bs2 (bottom)
  Total slot depth = Hs0 + Hs1 + Hs2

SLOTLESS PARAMETERS — Only for Slotless type
  Winding Inner R .... Inner radius of winding band [mm]
  Winding Outer R .... Outer radius of winding band [mm]

MAGNET CONFIGURATION
  Nº Poles ........... Number of magnetic poles
  Mag Thickness ...... Radial magnet thickness [mm]
  Pole Arc Ratio ..... Magnet span / pole pitch (0.4-0.95)
  Magnet Grade ....... NdFeB / SmCo grade selection
  Max Temp ........... Max magnet operating temp [°C]

HOUSING & SHAFT
  Housing Wall ....... Radial housing wall thickness [mm]
  Shaft Radius ....... Rotor shaft radius [mm]"""

    def _connect_signals(self):
        self._geo_panel.paramChanged = self._on_geo_changed
        self._btn_export_params.clicked.connect(self._export_params)
        self._btn_export_geo.clicked.connect(self._export_gmsh)
        QTimer.singleShot(100, self._redraw)

    def _on_geo_changed(self, geo):
        self._geo = geo
        self._redraw()

    def _redraw(self):
        try:
            self._canvas.draw_section(
                self._geo,
                show_labels=self._cb_labels.isChecked(),
                show_dimensions=self._cb_dims.isChecked(),
            )
            parts = self._geo.structure_type
            if parts == "Slotted":
                parts += f" | {self._geo.num_slots} slots"
            self._status_label.setText(
                f"Geometry: {parts} | {self._geo.num_poles} poles | "
                f"Lstk={self._geo.stack_length} | Rso={self._geo.Rso}"
            )
        except Exception as e:
            self._status_label.setText(f"❌ Error: {e}")

    def _export_params(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "motor_params.txt", "Text files (*.txt);;All files (*)")
        if fname:
            try:
                d = self._geo.as_dict()
                with open(fname, "w") as f:
                    f.write("=== Motor Thermal Model - Parameters ===\n\n")
                    for k, v in d.items():
                        if k not in ("_computed", "slot") or k == "slot":
                            if k == "slot" and v:
                                f.write(f"\n--- Slot Geometry ---\n")
                                for sk, sv in v.items():
                                    f.write(f"  {sk}: {sv}\n")
                            elif k != "_computed":
                                f.write(f"{k}: {v}\n")
                    f.write("\n=== Derived Quantities ===\n")
                    for k, v in self._geo.computed.items():
                        f.write(f"{k}: {v}\n")
                self.statusBar().showMessage(f"Exported to {fname}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))

    def _export_gmsh(self):
        QMessageBox.information(self, "Coming Soon",
                                "Gmsh export will be implemented in the next phase.")
        # from geometry.gmsh_export_v2 import generate_gmsh_geo_v2
        # fname, _ = QFileDialog.getSaveFileName(
        #     self, "Save Gmsh .geo file", "motor_geometry.geo",
        #     "Gmsh files (*.geo);;All files (*)")
        # if fname:
        #     try:
        #         mats = self._mat_panel.get_assignments()
        #         code = generate_gmsh_geo_v2(self._geo, mats)
        #         with open(fname, "w") as f:
        #             f.write(code)
        #         self.statusBar().showMessage(f"Exported .geo to {fname}")
        #     except Exception as e:
        #         QMessageBox.warning(self, "Export Error", str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.WindowText, QColor(30, 30, 30))
    app.setPalette(palette)

    window = MainWindowV2()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
