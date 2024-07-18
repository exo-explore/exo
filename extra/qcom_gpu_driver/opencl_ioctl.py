import ctypes, ctypes.util, struct, fcntl, re
from hexdump import hexdump
from tinygrad.runtime.ops_gpu import CLDevice, CLAllocator
import pathlib, sys
sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())

ops = {}
import xml.etree.ElementTree as ET
xml = ET.parse(pathlib.Path(__file__).parent / "adreno_pm4.xml")
for child in xml.getroot():
  if 'name' in child.attrib and child.attrib['name'] == "adreno_pm4_type3_packets":
    for sc in child:
      if 'name' in sc.attrib and ('variants' not in sc.attrib or sc.attrib['variants'] != "A2XX"):
        ops[int(sc.attrib['value'], 0x10)] = sc.attrib['name']
#print(ops)
#exit(0)

from extra.qcom_gpu_driver import msm_kgsl
def ioctls_from_header():
  hdr = (pathlib.Path(__file__).parent.parent.parent / "extra/qcom_gpu_driver/msm_kgsl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(IOCTL_KGSL_[A-Z0-9_]+)\s+_IOWR?\(KGSL_IOC_TYPE,\s+(0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)
  return {int(nr, 0x10):(name, getattr(msm_kgsl, "struct_"+sname)) for name, nr, sname in matches}

nrs = ioctls_from_header()

# https://github.com/ensc/dietlibc/blob/master/include/sys/aarch64-ioctl.h

def get_struct(argp, stype):
  return ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(stype)).contents

def format_struct(s):
  sdats = []
  for field_name, field_type in s._fields_:
    if field_name in {"__pad", "PADDING_0"}: continue
    dat = getattr(s, field_name)
    if isinstance(dat, int): sdats.append(f"{field_name}:0x{dat:X}")
    else: sdats.append(f"{field_name}:{dat}")
  return sdats

import mmap
mmaped = {}
def get_mem(addr, vlen):
  for k,v in mmaped.items():
    if k <= addr and addr < k+len(v):
      return v[addr-k:addr-k+vlen]

def hprint(vals):
  ret = []
  for v in vals:
    if v > 31: ret.append(f"{v:#x}")
    else: ret.append(f"{v}")
  return f"({','.join(ret)})"

ST6_SHADER = 0
ST6_CONSTANTS = 1

def parse_cmd_buf(dat):
  ptr = 0
  while ptr < len(dat):
    cmd = struct.unpack("I", dat[ptr:ptr+4])[0]
    if (cmd>>24) == 0x70:
      # packet with opcode and opcode specific payload (replace pkt3)
      opcode, size = ((cmd>>16)&0x7F), cmd&0x3FFF
      vals = struct.unpack("I"*size, dat[ptr+4:ptr+4+4*size])
      print(f"{ptr:3X} -- typ 7: {size=:3d}, {opcode=:#x} {ops[opcode]}", hprint(vals))
      if ops[opcode] == "CP_LOAD_STATE6_FRAG":
        dst_off = vals[0] & 0x3FFF
        state_type = (vals[0]>>14) & 0x3
        state_src = (vals[0]>>16) & 0x3
        state_block = (vals[0]>>18) & 0xF  # 13 = SB4_CS_SHADER
        num_unit = vals[0]>>22
        print(f"{num_unit=} {state_block=} {state_src=} {state_type=} {dst_off=}")

        from extra.disassemblers.adreno import disasm_raw
        if state_type == ST6_SHADER: disasm_raw(get_mem(((vals[2] << 32) | vals[1]), 0x180))
        if state_type == ST6_CONSTANTS: hexdump(get_mem(((vals[2] << 32) | vals[1]), min(0x180, num_unit*4)))
        pass
      ptr += 4*size
    elif (cmd>>28) == 0x4:
      # write one or more registers (replace pkt0)
      offset, size = ((cmd>>8)&0x7FFFF), cmd&0x7F
      vals = struct.unpack("I"*size, dat[ptr+4:ptr+4+4*size])
      print(f"{ptr:3X} -- typ 4: {size=:3d}, {offset=:#x}", hprint(vals))
      ptr += 4*size
    else:
      print("unk", hex(cmd))
    ptr += 4

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def ioctl(fd, request, argp):
  ret = libc.syscall(0x1d, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))

  idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF
  if nr in nrs and itype == 9:
    name, stype = nrs[nr]
    s = get_struct(argp, stype)
    print(f"{ret:2d} = {name:40s}", ' '.join(format_struct(s)))
    if name == "IOCTL_KGSL_GPUOBJ_INFO":
      mmaped[s.gpuaddr] = mmap.mmap(fd, s.size, offset=s.id*0x1000)
    if name == "IOCTL_KGSL_GPU_COMMAND":
      for i in range(s.numcmds):
        cmd = get_struct(s.cmdlist+s.cmdsize*i, msm_kgsl.struct_kgsl_command_object)
        print(f"cmd {i}:", format_struct(cmd))
        #hexdump(get_mem(cmd.gpuaddr, cmd.size))
        parse_cmd_buf(get_mem(cmd.gpuaddr, cmd.size))
      for i in range(s.numobjs):
        obj = get_struct(s.objlist+s.objsize*i, msm_kgsl.struct_kgsl_command_object)
        print(f"obj {i}:", format_struct(obj))
        print(format_struct(msm_kgsl.struct_kgsl_cmdbatch_profiling_buffer.from_buffer_copy(get_mem(obj.gpuaddr, obj.size))))
        #hexdump(get_mem(obj.gpuaddr, obj.size))
  else:
    #print(f"ioctl({fd=}, (dir:{idir}, size:0x{size:3X}, type:{itype:d}, nr:0x{nr:2X}), {argp=:X}) = {ret=}")
    pass

  return ret

def install_hook(c_function, python_function):
  # AARCH64 trampoline to ioctl
  tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
  tramp += struct.pack("Q", ctypes.cast(ctypes.byref(python_function), ctypes.POINTER(ctypes.c_ulong)).contents.value)

  # get real ioctl address
  ioctl_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))

  # hook ioctl
  libc = ctypes.CDLL(ctypes.util.find_library("libc"))
  ret = libc.mprotect(ctypes.c_ulong((ioctl_address.contents.value//0x1000)*0x1000), 0x2000, 7)
  assert ret == 0
  libc.memcpy(ioctl_address.contents, ctypes.create_string_buffer(tramp), len(tramp))

libc = ctypes.CDLL(ctypes.util.find_library("libc"))
install_hook(libc.ioctl, ioctl)

"""
print("***** init device")
dev = CLDevice()
print("***** alloc")
alloc = CLAllocator(dev)
a = alloc._alloc(16)
#alloc._alloc(0x2000)
ba = bytearray(b"hello")
print(f"***** copyin {ctypes.addressof((ctypes.c_char * len(ba)).from_buffer(ba)):#x}")
alloc.copyin(a, memoryview(ba))
dev.synchronize()
print("***** copyout")
mv2 = memoryview(bytearray(b"nopeo"))
alloc.copyout(mv2, a)
dev.synchronize()
print("***** done", bytes(mv2))
exit(0)
"""

print("***** import tinygrad")
from tinygrad import Tensor, Device, TinyJit
print("***** access GPU")
dev = Device["GPU"]
print("***** create tensor a")
a = Tensor([1.,2.]*200).realize()
print("***** create tensor b")
b = Tensor([3.,4.]*200).realize()
@TinyJit
def add(a, b): return (a+b).realize()
for i in range(4):
  print(f"***** add tensors {i}")
  c = add(a, b)
  #dev.synchronize()
  c = add(b, a)
  dev.synchronize()
#print("***** copy out")
#print(c.numpy())
