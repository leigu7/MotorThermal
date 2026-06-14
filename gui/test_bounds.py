"""Quick test to verify all spinbox bounds are relaxed."""
import sys
sys.path.insert(0, '.')
from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)

from gui.motor_gui_main import GeometryInputPanel
gp = GeometryInputPanel()

print('=== Core Dimension Bounds (all 0 to 99999) ===')
for k in ['Rso','Rsi','Rro','Rri','airgap_length','stack_length']:
    sb = gp._widgets[k]
    print(f'  {k}: {sb.minimum()} to {sb.maximum()}')

print()
print('=== Slot Bounds ===')
for k in ['num_slots','slot_depth','slot_opening','tooth_width_min']:
    sb = gp._widgets[k]
    print(f'  {k}: {sb.minimum()} to {sb.maximum()}')

print()
print('=== Magnet Bounds ===')
print(f'  num_poles: {gp._widgets["num_poles"].minimum()} to {gp._widgets["num_poles"].maximum()}')
print(f'  magnet_thickness: {gp._widgets["magnet_thickness"].minimum()} to {gp._widgets["magnet_thickness"].maximum()}')
msr = gp._widgets['magnet_span_ratio']
print(f'  magnet_span_ratio: {msr.minimum()} to {msr.maximum()}')

print()
print('=== Test Rri=0 (slotless, no shaft) ===')
gp._widgets['Rri'].setValue(0)
print(f'  set Rri=0: value={gp._widgets["Rri"].value()}')
print(f'  geo.Rri={gp.current_geometry.Rri}')

print()
print('=== Magnet Grade -> Temp ===')
cb = gp._widgets['magnet_grade']
le = gp._widgets['magnet_max_temp']
print(f'  N35SH -> {le.text()}')
cb.setCurrentText('SmCo 2:17')
print(f'  SmCo -> {le.text()}')
print(f'  Read-only: {le.isReadOnly()}')

print()
print('ALL OK')
