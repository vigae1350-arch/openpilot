"""python3 -m openpilot.selfdrive.jetlink_adapter on|off|status: switch the phone link."""
import sys

from openpilot.selfdrive.jetlink_adapter import _cli

sys.exit(_cli(sys.argv[1:]))
