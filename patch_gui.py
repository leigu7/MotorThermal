"""Patch the motor_gui_main.py file to add thermal network schematic to ResultPanel."""
import sys

with open('gui/motor_gui_main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add matplotlib imports after the lpt_fe.node import line
old_import = 'from lpt_fe.node import ThermalNode, ThermalResistance'
new_import = '''from lpt_fe.node import ThermalNode, ThermalResistance

# For thermal network schematic
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Figure = None
    FigureCanvas = None'''

content = content.replace(old_import, new_import, 1)
print('1. Added matplotlib imports OK')

# 2. Replace the entire ResultPanel class
idx_start = content.find('class ResultPanel(QWidget):')
idx_end = content.find('class MainWindow(QMainWindow):')

new_result_panel = '''class ResultPanel(QWidget):
    """
    Panel showing detailed LPTN model results.
    - Network schematic diagram (matplotlib)
    - Node table: name, temp, loss, volume, capacitance
    - Resistance table: name, from->to, R, type, length, area, k, h
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._net = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._net_info = QLabel("No simulation data. Run LPTN first.")
        self._net_info.setStyleSheet("font-weight: bold; font-size: 11pt; padding: 2px;")
        layout.addWidget(self._net_info)

        # Schematic canvas
        if HAS_MATPLOTLIB:
            self._schematic_fig = Figure(figsize=(8, 3), dpi=90)
            self._schematic_fig.set_facecolor('#f8f8f8')
            self._schematic_canvas = FigureCanvas(self._schematic_fig)
            self._schematic_canvas.setMaximumHeight(250)
            self._schematic_ax = self._schematic_fig.add_subplot(111)
            self._schematic_ax.axis('off')
            layout.addWidget(self._schematic_canvas)
        else:
            self._schematic_canvas = None
            lbl = QLabel("Thermal network schematic requires matplotlib.")
            lbl.setStyleSheet("color: #888; font-style: italic; padding: 10px;")
            layout.addWidget(lbl)

        # Node table
        node_group = QGroupBox("Node Properties")
        node_layout = QVBoxLayout(node_group)
        self._node_table = QTableWidget()
        self._node_table.setColumnCount(6)
        self._node_table.setHorizontalHeaderLabels([
            "Node", "T [C]", "Loss [W]", "Volume [cm3]", "Capacitance [J/K]", "Temp Dep."
        ])
        self._node_table.setAlternatingRowColors(True)
        self._node_table.setMaximumHeight(180)
        self._node_table.horizontalHeader().setStretchLastSection(True)
        node_layout.addWidget(self._node_table)
        layout.addWidget(node_group)

        # Resistance table
        res_group = QGroupBox("Thermal Resistances")
        res_layout = QVBoxLayout(res_group)
        self._res_table = QTableWidget()
        self._res_table.setColumnCount(7)
        self._res_table.setHorizontalHeaderLabels([
            "Name", "From -> To", "R [K/W]", "Type", "Length [mm]", "Area [mm2]", "k/h [W/mK]"
        ])
        self._res_table.setAlternatingRowColors(True)
        self._res_table.horizontalHeader().setStretchLastSection(True)
        self._res_table.setMaximumHeight(180)
        res_layout.addWidget(self._res_table)
        layout.addWidget(res_group)

    def _draw_schematic(self, net):
        """Draw thermal network schematic using matplotlib."""
        if not HAS_MATPLOTLIB or self._schematic_canvas is None:
            return
        ax = self._schematic_ax
        ax.clear()
        ax.axis('off')

        nodes = net.nodes
        if len(nodes) == 0:
            self._schematic_canvas.draw()
            return

        name_to_node = {n.name: n for n in nodes}
        layout_order = [
            "ambient", "housing", "stator_yoke", "stator_tooth",
            "slot_winding", "stator_tip", "magnet", "rotor_core", "shaft",
            "end_winding",
        ]
        ordered = [n for n in layout_order if n in name_to_node]
        n_ord = len(ordered)
        if n_ord == 0:
            return

        y_pos = {name: 1.0 - (i + 0.5) / n_ord for i, name in enumerate(ordered)}
        node_xy = {}
        for name in ordered:
            y = y_pos[name]
            if name == "ambient":
                x = 0.08
            elif name == "end_winding":
                x = 0.92
            elif name in ("magnet", "rotor_core", "shaft"):
                x = 0.30
            elif name == "slot_winding":
                x = 0.70
            else:
                x = 0.50
            node_xy[name] = (x, y)

        # Draw resistances as lines
        for res in net.resistances:
            fn = net.nodes[res.node_from].name
            tn = net.nodes[res.node_to].name
            if fn in node_xy and tn in node_xy:
                x1, y1 = node_xy[fn]
                x2, y2 = node_xy[tn]
                color = '#e74c3c' if res.resistance_type == "convection" else '#2c3e50'
                lw = max(0.5, min(3.0, 1.0 / (res.effective_resistance + 0.01)))
                ax.plot([x1, x2], [y1, y2], '-', color=color, lw=lw, alpha=0.5, zorder=1)

        # Draw nodes as circles
        for name, (x, y) in node_xy.items():
            node = name_to_node[name]
            radius = max(0.015, min(0.065, 0.008 + node.capacitance / 25000))
            t = node.temperature
            if node.fixed_temperature is not None:
                fc = '#3498db'
            elif t < 80:
                fc = '#2ecc71'
            elif t < 120:
                fc = '#f39c12'
            elif t < 180:
                fc = '#e67e22'
            else:
                fc = '#e74c3c'
            circle = plt.Circle((x, y), radius, facecolor=fc,
                                edgecolor='black', linewidth=1.0, alpha=0.85, zorder=2)
            ax.add_patch(circle)
            ax.annotate(f"{node.name}\\n{t:.0f}C", (x, y - radius - 0.025),
                        fontsize=6, ha='center', va='top', zorder=3)

        ax.plot([], [], '-', color='#2c3e50', label='Conduction', alpha=0.6)
        ax.plot([], [], '-', color='#e74c3c', label='Convection', alpha=0.6)
        ax.legend(loc='lower center', fontsize=7, ncol=2,
                  framealpha=0.7, handlelength=1.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 1.15)
        ax.set_title("Equivalent Thermal Network", fontsize=10, fontweight='bold')
        self._schematic_fig.tight_layout()
        self._schematic_canvas.draw()

    def display_network(self, net, config):
        """Fill tables with network data and draw schematic."""
        self._net = net
        conv = getattr(net, 'solver_converged', None)
        iters = getattr(net, 'solver_iterations', None)
        info = f"Network: {net.name} | {len(net.nodes)} nodes, {len(net.resistances)} resistances"
        if iters:
            info += f" | {iters} iterations"
            info += " | Converged" if conv else " | Not converged"
        self._net_info.setText(info)

        # Node table
        nodes = net.nodes
        self._node_table.setRowCount(len(nodes))
        for i, node in enumerate(nodes):
            self._node_table.setItem(i, 0, QTableWidgetItem(node.name))
            ti = QTableWidgetItem(f"{node.temperature:.1f}")
            ti.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._node_table.setItem(i, 1, ti)
            pi = QTableWidgetItem(f"{node.effective_loss:.2f}")
            pi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._node_table.setItem(i, 2, pi)
            vi = QTableWidgetItem(f"{node.volume * 1e6:.1f}")
            vi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._node_table.setItem(i, 3, vi)
            ci = QTableWidgetItem(f"{node.capacitance:.1f}")
            ci.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._node_table.setItem(i, 4, ci)
            tdi = QTableWidgetItem("Yes" if node.loss_temperature_dependent else "No")
            tdi.setTextAlignment(Qt.AlignCenter)
            self._node_table.setItem(i, 5, tdi)
        self._node_table.resizeColumnsToContents()

        # Resistance table
        reses = net.resistances
        self._res_table.setRowCount(len(reses))
        for i, res in enumerate(reses):
            self._res_table.setItem(i, 0, QTableWidgetItem(res.name))
            fn = net.nodes[res.node_from].name if res.node_from < len(net.nodes) else f"n{res.node_from}"
            tn = net.nodes[res.node_to].name if res.node_to < len(net.nodes) else f"n{res.node_to}"
            self._res_table.setItem(i, 1, QTableWidgetItem(f"{fn} -> {tn}"))
            ri = QTableWidgetItem(f"{res.effective_resistance:.4f}")
            ri.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._res_table.setItem(i, 2, ri)
            self._res_table.setItem(i, 3, QTableWidgetItem(res.resistance_type))
            li = QTableWidgetItem(f"{res.effective_length * 1000:.2f}")
            li.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._res_table.setItem(i, 4, li)
            ai = QTableWidgetItem(f"{res.effective_area * 1e6:.2f}")
            ai.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._res_table.setItem(i, 5, ai)
            if res.resistance_type == "convection" and res.h_coefficient > 0:
                khi = QTableWidgetItem(f"h={res.h_coefficient:.1f}")
            elif res.conductivity > 0:
                khi = QTableWidgetItem(f"k={res.conductivity:.3f}")
            else:
                khi = QTableWidgetItem("—")
            khi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._res_table.setItem(i, 6, khi)
        self._res_table.resizeColumnsToContents()

        # Draw schematic
        self._draw_schematic(net)


'''

if idx_start > 0 and idx_end > 0:
    content = content[:idx_start] + new_result_panel + content[idx_end:]
    print('2. Replaced ResultPanel class OK')
else:
    print('2. ERROR: Could not find boundaries. idx_start=%d idx_end=%d' % (idx_start, idx_end))

with open('gui/motor_gui_main.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Written to gui/motor_gui_main.py')
print('Done')
