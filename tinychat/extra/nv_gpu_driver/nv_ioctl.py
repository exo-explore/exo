# type: ignore
import ctypes, ctypes.util, struct, platform, pathlib, re, time, os, signal
from tinygrad.helpers import from_mv, to_mv, getenv
from hexdump import hexdump
start = time.perf_counter()

# *** ioctl lib ***
libc = ctypes.CDLL(ctypes.util.find_library("c"))
processor = platform.processor()
IOCTL_SYSCALL = {"aarch64": 0x1d, "x86_64":16}[processor]
MMAP_SYSCALL = {"aarch64": 0xde, "x86_64":0x09}[processor]

def get_struct(argp, stype):
  return ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(stype)).contents

def dump_struct(st):
  print("\t", st.__class__.__name__, end=" { ")
  for v in type(st)._fields_: print(f"{v[0]}={getattr(st, v[0])}", end=" ")
  print("}")

def format_struct(s):
  sdats = []
  for field in s._fields_:
    dat = getattr(s, field[0])
    if isinstance(dat, int): sdats.append(f"{field[0]}:0x{dat:X}")
    else: sdats.append(f"{field[0]}:{dat}")
  return sdats

real_func_pool = {}
def install_hook(c_function, python_function):
  orig_func = (ctypes.c_char*4096)()
  python_function_addr = ctypes.cast(ctypes.byref(python_function), ctypes.POINTER(ctypes.c_ulong)).contents.value
  # AARCH64 trampoline to ioctl
  if processor == "aarch64":
    # 0x0000000000000000:  70 00 00 10    adr x16, #0xc
    # 0x0000000000000004:  10 02 40 F9    ldr x16, [x16]
    # 0x0000000000000008:  00 02 1F D6    br  x16
    tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
    tramp += struct.pack("Q", python_function_addr)
  elif processor == "x86_64":
    # 0x0000000000000000:  49 BB aa aa aa aa aa aa aa aa    movabs r11, <address>
    # 0x000000000000000a:  41 FF E3                         jmp    r11
    tramp = b"\x49\xBB" + struct.pack("Q", python_function_addr) + b"\x41\xFF\xE3"
  else:
    raise Exception(f"processor {processor} not supported")

  # get real ioctl address
  ioctl_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))

  # hook ioctl
  ret = libc.mprotect(ctypes.c_ulong((ioctl_address.contents.value//0x1000)*0x1000), 0x2000, 7)
  assert ret == 0
  ret = libc.mprotect(ctypes.c_ulong((ctypes.addressof(orig_func)//0x1000)*0x1000), 0x3000, 7)
  assert ret == 0
  libc.memcpy(orig_func, ioctl_address.contents, 0x1000)
  libc.memcpy(ioctl_address.contents, ctypes.create_string_buffer(tramp), len(tramp))
  return orig_func

# *** ioctl lib end ***
import tinygrad.runtime.autogen.nv_gpu as nv_gpu
nvescs = {getattr(nv_gpu, x):x for x in dir(nv_gpu) if x.startswith("NV_ESC")}
nvcmds = {getattr(nv_gpu, x):(x, getattr(nv_gpu, "struct_"+x+"_PARAMS", getattr(nv_gpu, "struct_"+x.replace("_CMD_", "_")+"_PARAMS", None))) for x in dir(nv_gpu) if \
          x.startswith("NV") and x[6:].startswith("_CTRL_") and isinstance(getattr(nv_gpu, x), int)}

def get_classes():
  hdrpy = (pathlib.Path(__file__).parent.parent.parent / "tinygrad/runtime/autogen/nv_gpu.py").read_text()
  clss = re.search(r'NV01_ROOT.*?NV_SEMAPHORE_SURFACE = \(0x000000da\) # macro', hdrpy, re.DOTALL).group()
  pattern = r'([0-9a-zA-Z_]*) = +\((0x[0-9a-fA-F]+)\)'
  matches = re.findall(pattern, clss, re.MULTILINE)
  return {int(num, base=16):name for name, num in matches}
nvclasses = get_classes()
nvuvms = {getattr(nv_gpu, x):x for x in dir(nv_gpu) if x.startswith("UVM_") and nv_gpu.__dict__.get(x+"_PARAMS")}
nvqcmds = {int(getattr(nv_gpu, x)):x for x in dir(nv_gpu) if x[:7] in {"NVC6C0_", "NVC56F_", "NVC6B5_"} and isinstance(getattr(nv_gpu, x), int)}

global_ioctl_id = 0
gpus_user_modes = []
gpus_mmio = []
gpus_fifo = []

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def ioctl(fd, request, argp):
  global global_ioctl_id, gpus_user_modes, gpus_mmio
  global_ioctl_id += 1
  st = time.perf_counter()
  ret = libc.syscall(IOCTL_SYSCALL, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))
  et = time.perf_counter()-st
  fn = os.readlink(f"/proc/self/fd/{fd}")
  #print(f"ioctl {request:8x} {fn:20s}")
  idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF
  print(f"#{global_ioctl_id}: ", end="")
  if itype == ord(nv_gpu.NV_IOCTL_MAGIC):
    if nr == nv_gpu.NV_ESC_RM_CONTROL:
      s = get_struct(argp, nv_gpu.NVOS54_PARAMETERS)
      if s.cmd in nvcmds:
        name, struc = nvcmds[s.cmd]
        print(f"NV_ESC_RM_CONTROL    cmd={name:30s} hClient={s.hClient}, hObject={s.hObject}, flags={s.flags}, params={s.params}, paramsSize={s.paramsSize}, status={s.status}")
        if struc is not None: dump_struct(get_struct(s.params, struc))
        elif hasattr(nv_gpu, name+"_PARAMS"): dump_struct(get_struct(argp, getattr(nv_gpu, name+"_PARAMS")))
        elif name == "NVA06C_CTRL_CMD_GPFIFO_SCHEDULE": dump_struct(get_struct(argp, nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS))
      else:
        print("unhandled cmd", hex(s.cmd))
      # format_struct(s)
      # print(f"{(st-start)*1000:7.2f} ms +{et*1000.:7.2f} ms : {ret:2d} = {name:40s}", ' '.join(format_struct(s)))
    elif nr == nv_gpu.NV_ESC_RM_ALLOC:
      s = get_struct(argp, nv_gpu.NVOS21_PARAMETERS)
      print(f"NV_ESC_RM_ALLOC    hClass={nvclasses.get(s.hClass, 'unk'):30s}, hRoot={s.hRoot}, hObjectParent={s.hObjectParent}, pAllocParms={s.pAllocParms}, hObjectNew={s.hObjectNew}")
      if s.pAllocParms is not None:
        if s.hClass == nv_gpu.NV01_DEVICE_0: dump_struct(get_struct(s.pAllocParms, nv_gpu.NV0080_ALLOC_PARAMETERS))
        if s.hClass == nv_gpu.FERMI_VASPACE_A: dump_struct(get_struct(s.pAllocParms, nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS))
        if s.hClass == nv_gpu.NV50_MEMORY_VIRTUAL: dump_struct(get_struct(s.pAllocParms, nv_gpu.NV_MEMORY_ALLOCATION_PARAMS))
        if s.hClass == nv_gpu.NV1_MEMORY_USER: dump_struct(get_struct(s.pAllocParms, nv_gpu.NV_MEMORY_ALLOCATION_PARAMS))
        if s.hClass == nv_gpu.NV1_MEMORY_SYSTEM: dump_struct(get_struct(s.pAllocParms, nv_gpu.NV_MEMORY_ALLOCATION_PARAMS))
        if s.hClass == nv_gpu.AMPERE_CHANNEL_GPFIFO_A:
          sx = get_struct(s.pAllocParms, nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS)
          dump_struct(sx)
          gpus_fifo.append((sx.gpFifoOffset, sx.gpFifoEntries))
        if s.hClass == nv_gpu.KEPLER_CHANNEL_GROUP_A: dump_struct(get_struct(s.pAllocParms, nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS))
      if s.hClass == nv_gpu.TURING_USERMODE_A: gpus_user_modes.append(s.hObjectNew)
    elif nr == nv_gpu.NV_ESC_RM_MAP_MEMORY:
      # nv_ioctl_nvos33_parameters_with_fd
      s = get_struct(argp, nv_gpu.NVOS33_PARAMETERS)
      print(f"NV_ESC_RM_MAP_MEMORY   hClient={s.hClient}, hDevice={s.hDevice}, hMemory={s.hMemory}, length={s.length} flags={s.flags} pLinearAddress={s.pLinearAddress}")
    elif nr == nv_gpu.NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO:
      s = get_struct(argp, nv_gpu.NVOS56_PARAMETERS)
      print(f"NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO   hClient={s.hClient}, hDevice={s.hDevice}, hMemory={s.hMemory}, pOldCpuAddress={s.pOldCpuAddress} pNewCpuAddress={s.pNewCpuAddress} status={s.status}")
    elif nr == nv_gpu.NV_ESC_RM_ALLOC_MEMORY:
      s = get_struct(argp, nv_gpu.nv_ioctl_nvos02_parameters_with_fd)
      print(f"NV_ESC_RM_ALLOC_MEMORY  fd={s.fd}, hRoot={s.params.hRoot}, hObjectParent={s.params.hObjectParent}, hObjectNew={s.params.hObjectNew}, hClass={s.params.hClass}, flags={s.params.flags}, pMemory={s.params.pMemory}, limit={s.params.limit}, status={s.params.status}")
    elif nr == nv_gpu.NV_ESC_ALLOC_OS_EVENT:
      s = get_struct(argp, nv_gpu.nv_ioctl_nvos02_parameters_with_fd)
    elif nr == nv_gpu.NV_ESC_REGISTER_FD:
      s = get_struct(argp, nv_gpu.nv_ioctl_register_fd_t)
      print(f"NV_ESC_REGISTER_FD  fd={s.ctl_fd}")
    elif nr in nvescs:
      print(nvescs[nr])
    else:
      print("unhandled NR", nr)
  elif fn.endswith("nvidia-uvm"):
    print(f"{nvuvms.get(request, f'UVM UNKNOWN {request=}')}")
    if nvuvms.get(request) is not None: dump_struct(get_struct(argp, getattr(nv_gpu, nvuvms.get(request)+"_PARAMS")))
    if nvuvms.get(request) == "UVM_MAP_EXTERNAL_ALLOCATION":
      st = get_struct(argp, getattr(nv_gpu, nvuvms.get(request)+"_PARAMS"))
      for i in range(st.gpuAttributesCount):
        print("perGpuAttributes[{i}] = ", end="")
        dump_struct(st.perGpuAttributes[i])
    print("ok")

  if getenv("IOCTL") >= 2: print("ioctl", f"{idir=} {size=} {itype=} {nr=} {fd=} {ret=}", fn)
  return ret

@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
def _mmap(addr, length, prot, flags, fd, offset):
  mmap_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
  orig_mmap = mmap_type(ctypes.addressof(orig_mmap_mv))
  ret = orig_mmap(addr, length, prot, flags, fd, offset)
  # ll = os.readlink(f"/proc/self/fd/{fd}") if fd >= 0 else ""
  print(f"mmap {addr=}, {length=}, {prot=}, {flags=}, {fd=}, {offset=} {ret=}")
  return ret

install_hook(libc.ioctl, ioctl)
if getenv("IOCTL") >= 3: orig_mmap_mv = install_hook(libc.mmap, _mmap)

import collections
old_gpputs = collections.defaultdict(int)
def _dump_gpfifo(mark):
  print("_dump_gpfifo:", mark)
  for start,size in gpus_fifo:
    gpfifo_controls = nv_gpu.AmpereAControlGPFifo.from_address(start+size*8)
    gpfifo = to_mv(start, gpfifo_controls.GPPut * 8).cast("Q")
    if old_gpputs[start] == gpfifo_controls.GPPut: continue

    print(f"gpfifo {start}: {gpfifo_controls.GPPut=}")
    for i in range(old_gpputs[start], gpfifo_controls.GPPut):
      addr = ((gpfifo[i % size] & ((1 << 40)-1)) >> 2) << 2
      pckt_cnt = (gpfifo[i % size]>>42)&((1 << 20)-1)

      print(f"\t{i}: 0x{gpfifo[i % size]:x}: addr:0x{addr:x} packets:{pckt_cnt} sync:{(gpfifo[i % size] >> 63) & 0x1} fetch:{gpfifo[i % size] & 0x1}")
      old_gpputs[start] = gpfifo_controls.GPPut
      _dump_qmd(addr, pckt_cnt)

import types
def _dump_qmd(address, packets):
  gpfifo = to_mv(address, packets * 4).cast("I")

  i = 0
  while i < packets:
    dat = gpfifo[i]
    typ = (dat>>28) & 0xF
    if typ == 0: break
    size = (dat>>16) & 0xFFF
    subc = (dat>>13) & 7
    mthd = (dat<<2) & 0x7FFF
    method_name = nvqcmds.get(mthd, f"unknown method #{mthd}")
    print(f"\t\t{method_name}, {typ=} {size=} {subc=} {mthd=}")
    for j in range(size): print(f"\t\t\t{j}: {gpfifo[i+j+1]} | 0x{gpfifo[i+j+1]:x}")
    if mthd == 792:
      for x in dir(nv_gpu):
        if x.startswith("NVC6C0_QMDV03_00_"):
          vv = getattr(nv_gpu, x)
          bits = None
          if isinstance(vv, tuple) and len(vv) == 2:
            bits = vv
          if isinstance(vv, types.FunctionType):
            bits = vv(0)
          
          if bits is not None:
            res = 0
            for bt in range(bits[1], bits[0]+1): res |= ((gpfifo[i + 3 + bt // 32] >> (bt % 32)) & 0x1) << (bt - bits[1])
            if res != 0: print(f"{x}, {hex(res)} | {bin(res)}")

      const_addr = gpfifo[i+35] + ((gpfifo[i+36] & 0xffff) << 32)
      const_len = ((gpfifo[i+36] >> 19))
      # hexdump(to_mv(const_addr, const_len))

    i += size + 1

# IOCTL=1 PTX=1 CUDA=1 python3 test/test_ops.py TestOps.test_tiny_add