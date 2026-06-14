"""Debug why network diagram is not visible."""
from PyQt5.QtWidgets import QApplication
a = QApplication([])
from gui.motor_gui_main import MainWindow
from lpt_fe import NetworkBuilderConfig, build_thermal_network, solve_steady_state

mw = MainWindow()
rp = mw._result_panel

# Show tab info
print('Number of tabs:', mw._tabs.count())
for i in range(mw._tabs.count()):
    txt = mw._tabs.tabText(i).encode('ascii', errors='replace').decode('ascii')
    widget = mw._tabs.widget(i)
    print(f'  Tab {i}: "{txt}" -> {type(widget).__name__}')

# Run simulation
cfg = mw._lptn_panel.get_config()
geo_v2 = mw._geo_to_v2()
net = build_thermal_network(geo_v2, cfg)
T = solve_steady_state(net)
mw._display_lptn_results(net)

# Force show the window and results tab
mw.show()
mw._tabs.setCurrentIndex(3)
a.processEvents()

print()
print('After Results tab selected:')
print(f'  _result_panel visible: {rp.isVisible()}')
print(f'  _net_canvas visible: {rp._net_canvas.isVisible()}')
print(f'  _net_canvas size: {rp._net_canvas.width()}x{rp._net_canvas.height()}')
print(f'  _net_info text: {rp._net_info.text()}')
print(f'  _btn_undock enabled: {rp._btn_undock.isEnabled()}')

# Check matplotlib figure
fig = rp._net_canvas.figure
print(f'  Axes count: {len(fig.axes)}')
if fig.axes:
    ax = fig.axes[0]
    print(f'  Title: {ax.get_title()}')
    print(f'  Artists in axes: {len(ax.get_children())}')

print('\nCanvas layout test:')
# Check if canvas has a valid FigureCanvas
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
print(f'  Is FigureCanvas: {isinstance(rp._net_canvas, FigureCanvasQTAgg)}')

# Check the layout hierarchy
print('\nHierarchy:')
from PyQt5.QtWidgets import QWidget
w = rp._net_canvas
while w:
    print(f'  {type(w).__name__} visible={w.isVisible()} size={w.width()}x{w.height()}')
    w = w.parentWidget()

input("Press Enter to close...")
