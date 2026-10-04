"""
carrot-jetlink: jetlink on CARROT (carrot-wip), for a phone running the large model.

Ported from zoompilot's openpilot/sunnypilot/jetlink_adapter (MIT, Zeph
Leggett) to carrot's tree. jetlink (jetlink_repo, linked in as `jetlink`)
runs comma's large driving model on an attached phone, Jetson or Mac over the
comma's USB-C port and never imports openpilot; this module is the one place
that holds every openpilot import it needs, and the hooks in manager and
modeld call the functions at the bottom.

carrot-wip already carries its own Jetlink link for a Jetson or Mac (jetlinkd,
openpilot.selfdrive.modeld.jetlink, protocol v2, Cinque v2). The phone apps
speak upstream's current protocol (v3), so this phone link is a separate path,
and the two never run together: with JetlinkLink set (the phone link on),
carrot's jetlinkd and its JoiningModel stand down and this module's owner
(jetlinkphoned) and joining model run instead. With JetlinkLink unset or 0,
carrot-wip runs exactly as shipped and nothing here is imported past the
standard library.

Safety, because the comma must always boot and drive:
- every hook is guarded: a failure here leaves carrot's own small model driving;
- the owner never touches the USB-C port's input current for a phone (only the
  iOS mode does, as upstream); a comma powered from that port keeps its power;
- the owner counts boots: if it starts on three boots in a row that each end
  within two minutes, it turns the phone link off by itself (failsafe).

What differs from zoompilot:
- carrot has no sunnypilot model manager, so there is no big-model pick or
  catalog: jetlink runs its own default large model (Cinque Terre V3, the one
  carrot's eGPU runs), fetched from comma's public LFS.
- carrot's Params registry does not declare jetlink's keys, and manager's
  Params.clear_all deletes every undeclared file in the params directory at
  each start and each onroad/offroad change. So jetlink's own keys live as
  plain files in a directory of their own beside it, /data/params/jetlink-phone,
  written the way params.cc writes them (temp file, fsync, rename); IsOffroad
  is read from openpilot's params directory. Nothing native needs rebuilding.
  Switch the phone link with: python3 -m openpilot.selfdrive.jetlink_adapter on|off|status
- carrot's eGPU (comma's chestnut board on the USB-C port) plays chestnut's
  part: while one is fitted the link stays off.
- carrot's ModelState.run takes prepare_only where jetlink passes
  after_enqueue; SmallModel and CarrotModel translate between the two.

The top level is the standard library only: manager imports this module to
build the process list and runs it as jetlinkd, the resident gadget owner.
"""
from __future__ import annotations

import functools
import json
import os
import tempfile
import threading
from collections import namedtuple
from pathlib import Path

API = 1
OWNER = 'jetlinkphoned'
MODES = ('off', 'usb', 'ios')

_Keys = namedtuple('_Keys', 'link offroad progress spec pointers big_model catalog')
KEYS = _Keys(link='JetlinkLink', offroad='IsOffroad', progress='AcceleratorProgress', spec='JetlinkSpec',
             pointers='JetlinkModelPointers', big_model=None, catalog=None)

# keys carrot's params_keys.h does not declare: kept as files in _params_dir()
FILE_KEYS = {KEYS.link: 'int', KEYS.progress: 'json', KEYS.spec: 'json', KEYS.pointers: 'json'}

# comma's chestnut, running and in its ROM: carrot's eGPU (modeld.helpers.USBGPU_USB_IDS)
CHESTNUT_IDS = frozenset({(0xADD1, 0x0001), (0x3801, 0x0001), (0x174C, 0x2464), (0x174C, 0x2463)})

WARP_DIR = Path(__file__).resolve().parent / 'models'
OWNER_LOG = Path('/data/log/jetlink-owner.log')
_AGNOS = os.path.isfile('/AGNOS')
STATE_DIR: Path | None = None   # the failsafe's records; None: beside jetlink's keys
STABLE_AFTER_S = 120     # a boot the owner survives this long is a good boot
FAILSAFE_BOOTS = 3       # this many short boots in a row turn the phone link off


def _op_params_dir() -> Path:
  """openpilot's params directory (Params' own files: IsOffroad and the rest)."""
  prefix = os.environ.get('OPENPILOT_PREFIX', '')
  root = os.environ.get('PARAMS_ROOT')
  if root is None:
    root = '/data/params' if _AGNOS else os.path.join(os.environ.get('HOME', ''), '.comma' + prefix, 'params')
  return Path(root) / os.environ.get('OPENPILOT_PREFIX', 'd')


def _params_dir() -> Path:
  """jetlink's own key files: beside openpilot's params directory, never in it,
  where Params.clear_all would delete them."""
  return _op_params_dir().parent / 'jetlink-phone'


def _state_dir() -> Path:
  return STATE_DIR if STATE_DIR is not None else _params_dir()


def _keys() -> dict:
  """jetlink's key names. jetlink reads the link and IsOffroad off one
  directory, so IsOffroad is named by its path relative to ours."""
  return dict(KEYS._asdict(), offroad=os.path.relpath(_op_params_dir() / 'IsOffroad', _params_dir()))


def _file_get(key: str):
  try:
    raw = (_params_dir() / key).read_bytes()
  except OSError:
    return None
  try:
    if FILE_KEYS[key] == 'int':
      return int(raw)
    return json.loads(raw) if raw.strip() else None
  except (ValueError, TypeError):
    return None


def _file_put(key: str, value) -> None:
  if FILE_KEYS[key] == 'int':
    data = str(int(value)).encode()
  else:
    data = value if isinstance(value, bytes) else json.dumps(value).encode()
  directory = _params_dir()
  directory.mkdir(parents=True, exist_ok=True)
  fd, tmp = tempfile.mkstemp(prefix='.tmp_value_', dir=directory)
  try:
    with os.fdopen(fd, 'wb') as f:
      f.write(data)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, directory / key)
  except BaseException:
    try:
      os.unlink(tmp)
    except OSError:
      pass
    raise


def _file_remove(key: str) -> None:
  try:
    (_params_dir() / key).unlink()
  except FileNotFoundError:
    pass


def phone_mode() -> bool:
  """Is the phone link switched on? Standard library only, never raises."""
  try:
    return 0 < int((_params_dir() / KEYS.link).read_bytes()) < len(MODES)
  except (OSError, ValueError):
    return False


def carrot_link_allowed(started: bool, params, CP) -> bool:
  """manager's rule for carrot's own jetlinkd: off while the phone link is on."""
  return not phone_mode()


def warp_path(cam_w: int, cam_h: int, model_w: int, model_h: int) -> Path:
  return WARP_DIR / f'warp_{cam_w}x{cam_h}_{model_w}x{model_h}_tinygrad.pkl'


def owner_config():
  from jetlink.openpilot.interface import Keys, OwnerConfig

  from openpilot.common.basedir import BASEDIR
  return OwnerConfig(params_dir=_params_dir(), keys=Keys(**_keys()), chestnut_ids=CHESTNUT_IDS,
                     adapter=__name__, cwd=Path(BASEDIR), env={'PYTHONPATH': BASEDIR}, log_file=OWNER_LOG)


# -- the owner's failsafe and the warp build ----------------------------------

def _boot_id() -> str:
  try:
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  except OSError:
    return ''


def _read_state() -> dict:
  try:
    state = json.loads((_state_dir() / 'boots.json').read_text())
    return state if isinstance(state, dict) else {}
  except (OSError, ValueError):
    return {}


def _write_state(state: dict) -> None:
  directory = _state_dir()
  directory.mkdir(parents=True, exist_ok=True)
  tmp = directory / 'boots.json.tmp'
  tmp.write_text(json.dumps(state))
  os.replace(tmp, directory / 'boots.json')


def failsafe_tripped() -> bool:
  """Record this boot as started; True when the last FAILSAFE_BOOTS boots each
  started the owner and ended before it was up STABLE_AFTER_S. Then the phone
  link is turned off and stays off until it is switched on again."""
  boot = _boot_id()
  state = _read_state()
  short = [b for b in state.get('short', []) if isinstance(b, str)]
  if boot and boot not in short:
    short = (short + [boot])[-FAILSAFE_BOOTS:]
  if len(short) >= FAILSAFE_BOOTS:
    _file_put(KEYS.link, 0)
    _write_state({'short': [], 'tripped': boot})
    (_state_dir() / 'FAILSAFE').write_text(f"phone link turned off after {FAILSAFE_BOOTS} short boots in a row\n")
    return True
  _write_state({'short': short})
  return False


def _mark_stable() -> None:
  import time
  time.sleep(STABLE_AFTER_S)
  try:
    _write_state({'short': []})
  except OSError:
    pass


def _tinygrad_stamp() -> str:
  import subprocess
  from openpilot.common.basedir import BASEDIR
  try:
    return subprocess.run(['git', '-C', BASEDIR, 'rev-parse', 'HEAD:tinygrad_repo'], capture_output=True,
                          text=True, timeout=30).stdout.strip()
  except Exception:
    return ''


def _offroad() -> bool:
  try:
    return (_op_params_dir() / 'IsOffroad').read_bytes().strip() in (b'1', b'')
  except OSError:
    return True


def _build_warp_when_offroad() -> None:
  """Build the comma-side warp once, offroad, when it is missing or tinygrad
  changed. In a child process at low priority: a failure only means the small
  model keeps driving, and the next boot tries again."""
  import subprocess
  import sys
  import time
  from openpilot.common.basedir import BASEDIR
  stamp_file = WARP_DIR / 'tinygrad.stamp'
  stamp = _tinygrad_stamp()
  try:
    if any(WARP_DIR.glob('warp_*.pkl')) and stamp and stamp_file.read_text().strip() == stamp:
      return
  except OSError:
    pass
  time.sleep(30)   # let the boot settle first
  while not _offroad():
    time.sleep(30)
  cmd = [sys.executable, '-m', 'openpilot.selfdrive.jetlink_adapter.build_warp', '--force']
  env = dict(os.environ, PYTHONPATH=BASEDIR)
  try:
    OWNER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(OWNER_LOG, 'a') as log:
      log.write("jetlink-phone: building the warp\n")
      log.flush()
      done = subprocess.run(cmd, cwd=BASEDIR, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900,
                            preexec_fn=lambda: os.nice(10))
    if done.returncode == 0 and stamp:
      stamp_file.write_text(stamp)
  except Exception as e:
    _log_failure(f"warp build failed: {e}", e)


def main() -> None:
  """jetlinkphoned: hold the USB gadget until manager stops this process."""
  if not phone_mode():
    return
  try:
    _params_dir().mkdir(parents=True, exist_ok=True)   # jetlink reads IsOffroad through it
    if failsafe_tripped():
      _log_failure("phone link turned off by the failsafe (short boots in a row)", None)
      return
  except OSError:
    pass
  threading.Thread(target=_mark_stable, name='jetlink-stable', daemon=True).start()
  threading.Thread(target=_build_warp_when_offroad, name='jetlink-warp', daemon=True).start()
  from jetlink.openpilot.owner import main as run_owner
  run_owner(owner_config())


def adapter() -> Adapter:
  return Adapter()


class Adapter:
  """jetlink.openpilot.interface.Openpilot over carrot."""

  catalog_selector = 0   # no model manager

  def __init__(self):
    from jetlink.openpilot.interface import Keys

    from openpilot.common.basedir import BASEDIR
    from openpilot.common.swaglog import cloudlog
    self.keys = Keys(**_keys())
    self.log = cloudlog
    self.basedir = Path(BASEDIR)
    try:
      _params_dir().mkdir(parents=True, exist_ok=True)
    except OSError:
      pass
    self._stores: dict[Path, object] = {}

  # -- params ---------------------------------------------------------------

  def params_dir(self) -> Path:
    return _params_dir()

  def _params(self):
    where = _params_dir()
    store = self._stores.get(where)
    if store is None:
      from openpilot.common.params import Params
      store = self._stores[where] = Params()
    return store

  def _op_key(self, key: str) -> str:
    return 'IsOffroad' if key == self.keys.offroad else key

  def get(self, key: str):
    if key is None:
      return None
    key = self._op_key(key)
    if key in FILE_KEYS:
      return _file_get(key)
    try:
      return self._params().get(key)
    except Exception:
      return None

  def put(self, key: str, value, *, block: bool = False) -> None:
    key = self._op_key(key)
    if key in FILE_KEYS:
      _file_put(key, value)
      return
    self._params().put(key, value)

  def remove(self, key: str) -> None:
    key = self._op_key(key)
    if key in FILE_KEYS:
      _file_remove(key)
      return
    self._params().remove(key)

  # -- the device -------------------------------------------------------------

  def chestnut_present(self) -> bool:
    from openpilot.selfdrive.modeld.helpers import usbgpu_present
    return usbgpu_present()

  def camera(self) -> tuple[int, int, int, int]:
    from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
    from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE
    from openpilot.system.hardware import HARDWARE
    camera = _os_fisheye if HARDWARE.get_device_type() == "mici" else _ar_ox_fisheye
    return camera.width, camera.height, *MEDMODEL_INPUT_SIZE

  def warp_path(self, cam_w: int, cam_h: int, model_w: int, model_h: int) -> Path:
    return warp_path(cam_w, cam_h, model_w, model_h)

  def model_root(self) -> Path:
    from openpilot.system.hardware.hw import Paths
    root = getattr(Paths, 'model_root', None)
    if root is not None:
      return Path(root())
    return Path('/data/media/0/models' if _AGNOS else os.path.join(os.environ.get('HOME', ''), '.comma', 'media', 'models'))

  # -- modeld -----------------------------------------------------------------

  def model_face(self):
    from jetlink.openpilot.interface import ModelFace

    from openpilot.selfdrive.modeld.constants import ModelConstants
    from openpilot.selfdrive.modeld.modeld import LAT_SMOOTH_SECONDS, LONG_SMOOTH_SECONDS, get_action_from_model
    from openpilot.selfdrive.modeld.parse_model_outputs import Parser
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    return ModelFace(parser=Parser, frame_size=lambda w, h: get_nv12_info(w, h)[3], desire_len=ModelConstants.DESIRE_LEN,
                     constants=ModelConstants, lat_smooth_seconds=LAT_SMOOTH_SECONDS,
                     long_smooth_seconds=LONG_SMOOTH_SECONDS, get_action_from_model=get_action_from_model)

  def engagement(self):
    """The large model swaps in only while nothing is in control."""
    import openpilot.cereal.messaging as messaging
    services = ('selfdriveState', 'carState', 'carControl')
    sm = messaging.SubMaster(list(services))

    def engaged(timeout_ms: int) -> bool:
      sm.update(timeout_ms)
      valid = all(sm.seen[s] and sm.alive[s] and sm.valid[s] for s in services)
      cc = sm['carControl']
      return not valid or sm['selfdriveState'].enabled or cc.latActive or cc.longActive
    return engaged

  def event(self, name: str, **fields) -> None:
    self.log.event(name, **fields)

  # -- the build --------------------------------------------------------------

  def make_warp(self, cam_w: int, cam_h: int, model_w: int, model_h: int):
    # compile_modeld first: it patches tinygrad as it loads
    import openpilot.selfdrive.modeld.compile_modeld as cm
    from tinygrad.device import Device

    from openpilot.selfdrive.modeld.constants import ModelConstants
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    # carrot's warp graph reads its device from WARP_DEV; jetlink loads the
    # pickle against Device.DEFAULT, so capture it there
    if cm.WARP_DEV is None:
      cm.WARP_DEV = Device.DEFAULT
    nv12 = cm.NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
    frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
    return cm.make_warp(nv12, model_w, model_h, frame_skip), nv12.size


# -- carrot's ModelState <-> jetlink ------------------------------------------

class SmallModel:
  """carrot's ModelState as jetlink's joining model drives it: jetlink calls
  run(bufs, transforms, inputs, after_enqueue), carrot's takes prepare_only.
  Reads and writes go through to the model (input_queues, npy, prev_desire,
  vision_input_names, usbgpu)."""

  def __init__(self, model):
    object.__setattr__(self, '_model', model)

  def run(self, bufs, transforms, inputs, after_enqueue=None):
    out = self._model.run(bufs, transforms, inputs, False)
    if after_enqueue is not None:
      after_enqueue()
    return out

  def __getattr__(self, name):
    return getattr(self._model, name)

  def __setattr__(self, name, value):
    setattr(self._model, name, value)


class CarrotModel:
  """jetlink's joining model as carrot's modeld drives it."""

  usbgpu = False

  def __init__(self, joined):
    object.__setattr__(self, '_joined', joined)

  def run(self, bufs, transforms, inputs, prepare_only=False):
    # a dropped camera frame: carrot advances the small model's history without
    # publishing. jetlink's models keep their own history per frame sent, so
    # the frame runs and is published as zoompilot's modeld does
    return self._joined.run(bufs, transforms, inputs)

  def __getattr__(self, name):
    return getattr(self._joined, name)

  def __setattr__(self, name, value):
    setattr(self._joined, name, value)


# -- what the hooks call ------------------------------------------------------

class _Absent:
  def __init__(self, why: str | None):
    self.why = why

  def enabled(self) -> bool:
    return False

  def status(self):
    return None

  def reason(self) -> str | None:
    if self.why is None:
      return None
    try:
      on = 0 < int((_params_dir() / KEYS.link).read_bytes()) < len(MODES)
    except (OSError, ValueError):
      on = False
    return self.why if on else None

  def prepare(self) -> bool:
    return False

  def attach(self, small, cam_w: int, cam_h: int):
    return None


_bound = None
_binding = threading.Lock()


def _api():
  global _bound
  if _bound is None:
    with _binding:
      if _bound is None:
        _bound = _bind()
  return _bound


def _bind():
  try:
    import jetlink
    if getattr(jetlink, '__file__', None) is None:
      return _Absent(None)
    import jetlink.openpilot as jl
  except ModuleNotFoundError as e:
    if e.name == 'jetlink':
      return _Absent(None)
    return _unusable(f"jetlink failed to load: {e}", e)
  except Exception as e:
    return _unusable(f"jetlink failed to load: {type(e).__name__}: {e}", e)
  api = getattr(jl, 'API', None)
  if api != API:
    return _unusable(f"jetlink package API {api}, this build expects {API}")
  try:
    return jl.bind(Adapter())
  except Exception as e:
    return _unusable(f"jetlink failed to start: {type(e).__name__}: {e}", e)


def _unusable(why: str, error: Exception | None = None) -> _Absent:
  _log_failure(why, error)
  return _Absent(why)


_failed_hooks: dict[str, str] = {}


def _log_failure(what: str, error: Exception | None) -> None:
  try:
    from openpilot.common.swaglog import cloudlog
    cloudlog.error("jetlink: %s", what, exc_info=error)
  except Exception:
    pass


def _guarded(default):
  def wrap(hook):
    @functools.wraps(hook)
    def call(*args, **kwargs):
      try:
        result = hook(*args, **kwargs)
      except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if _failed_hooks.get(hook.__name__) != error:
          _failed_hooks[hook.__name__] = error
          _log_failure(f"{hook.__name__}() failed", e)
        return default
      _failed_hooks.pop(hook.__name__, None)
      return result
    return call
  return wrap


@_guarded(False)
def should_run(started: bool, params, CP) -> bool:
  """manager's rule for jetlinkphoned: the phone link is on and no eGPU is fitted."""
  return phone_mode() and _api().enabled()


@_guarded(None)
def status():
  return _api().status()


@_guarded(None)
def reason() -> str | None:
  return _api().reason()


@_guarded(False)
def prepare() -> bool:
  """modeld, before config_realtime_process: will the link join this modeld?"""
  return phone_mode() and _api().prepare()


@_guarded(None)
def attach(small, cam_w: int, cam_h: int):
  """modeld, once the camera is up and carrot's small ModelState is built:
  the model to run (small driving until the link has joined), or None."""
  if not phone_mode():
    return None
  wrapped = SmallModel(small)
  joined = _api().attach(wrapped, cam_w, cam_h)
  # attach() hands the small model back when the joining model could not be built
  return None if joined is None or joined is wrapped else CarrotModel(joined)


def _cli(argv: list[str]) -> int:
  """python3 -m openpilot.selfdrive.jetlink_adapter on|off|status"""
  cmd = argv[0] if argv else 'status'
  if cmd in ('on', 'off'):
    _file_put(KEYS.link, 1 if cmd == 'on' else 0)
    if cmd == 'on':
      for name in ('boots.json', 'FAILSAFE'):
        try:
          (_state_dir() / name).unlink()
        except OSError:
          pass
    print(f"phone link {cmd}. Reboot to apply (sudo reboot).")
    return 0
  if cmd == 'status':
    print(f"phone link: {'on' if phone_mode() else 'off'}  ({_params_dir() / KEYS.link})")
    if (_state_dir() / 'FAILSAFE').exists():
      print("failsafe: tripped -> " + (_state_dir() / 'FAILSAFE').read_text().strip())
    warps = sorted(WARP_DIR.glob('warp_*.pkl'))
    print(f"warp: {warps[0].name if warps else 'not built yet'}")
    print(f"log: {OWNER_LOG}")
    return 0
  print(_cli.__doc__)
  return 2

