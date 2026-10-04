#!/usr/bin/env python3
"""
carrot-jetlink: build the comma-side warp JIT jetlink's large model needs.

zoompilot builds this with scons; here the phone link's owner (jetlinkphoned)
runs it once, offroad, in a child process, never in the boot path. It compiles
only when the warp for this device's camera is missing, or with --force (after
a tinygrad change). A failure leaves no pickle: modeld then stays on the small
model and the next boot tries again.
"""
import argparse
import sys


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--force', action='store_true')
  args = p.parse_args()

  from openpilot.selfdrive import jetlink_adapter as ja
  ja.use_our_jetlink()
  op = ja.adapter()
  cam_w, cam_h, model_w, model_h = op.camera()
  out = ja.warp_path(cam_w, cam_h, model_w, model_h)
  if out.is_file() and not args.force:
    print(f"jetlink warp present: {out}")
    return 0
  from jetlink.openpilot.warp import main as build
  build(['--adapter', ja.__name__, '--camera', f'{cam_w}x{cam_h}', '--model', f'{model_w}x{model_h}',
         '--output', str(out)])
  return 0 if out.is_file() else 1


if __name__ == '__main__':
  sys.exit(main())
