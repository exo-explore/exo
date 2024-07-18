#!/bin/bash -e

# setup instructions for clang2py
if [[ ! $(clang2py -V) ]]; then
  pushd .
  cd /tmp
  sudo apt-get install -y --no-install-recommends clang
  pip install --upgrade pip setuptools
  pip install clang==14.0.6
  git clone https://github.com/geohot/ctypeslib.git
  cd ctypeslib
  pip install --user .
  clang2py -V
  popd
fi

BASE=tinygrad/runtime/autogen/

fixup() {
  sed -i '1s/^/# mypy: ignore-errors\n/' $1
  sed -i 's/ *$//' $1
  grep FIXME_STUB $1 || true
}

patch_dlopen() {
  path=$1; shift
  name=$1; shift
  cat <<EOF | sed -i "/import ctypes.*/r /dev/stdin" $path
PATHS_TO_TRY = [
$(for p in "$@"; do echo "  $p,"; done)
]
def _try_dlopen_$name():
  library = ctypes.util.find_library("$name")
  if library: return ctypes.CDLL(library)
  for candidate in PATHS_TO_TRY:
    try: return ctypes.CDLL(candidate)
    except OSError: pass
  raise RuntimeError("library $name not found")
EOF
}

process_cdefines() {
  local input_file="$1"
  local output_file="$2"

  sed -E '
    # Remove single-line comments
    s/[[:space:]]*\/\*.*\*\///g

    # Remove multi-line comments
    /\/\*/,/\*\//d

    /.*DT_MIPS_NUM.*/d

    # Remove lines ending with backslash (multi-line macros)
    /\\$/d

    # Convert C integer literals (remove U suffix)
    s/\b([0-9]+)U\b/\1/g

    # Convert C types to Python ctypes
    s/\bunsigned char\b/ctypes.c_ubyte/g
    s/\bsigned char\b/ctypes.c_byte/g
    s/\bunsigned short\b/ctypes.c_ushort/g
    s/\bshort\b/ctypes.c_short/g
    s/\bunsigned int\b/ctypes.c_uint/g
    s/\bint\b/ctypes.c_int/g
    s/\bunsigned long\b/ctypes.c_ulong/g
    s/\blong\b/ctypes.c_long/g
    s/\bfloat\b/ctypes.c_float/g
    s/\bdouble\b/ctypes.c_double/g

    # Function-like macros with parameters
    /^#define[[:space:]]+([[:alnum:]_]+)[[:space:]]*\(([^)]*)\)[[:space:]]+(.+)/ {
      s//def \1(\2): return \3/
      p
      d
    }

    # Simple #define statements (including those with parentheses)
    /^#define[[:space:]]+([[:alnum:]_]+)[[:space:]]+(.+)/ {
      s//\1 = \2/
      p
      d
    }

    # Drop all other lines
    d
  ' "$input_file" >> "$output_file"
}

generate_opencl() {
  clang2py /usr/include/CL/cl.h -o $BASE/opencl.py -l /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -k cdefstum
  fixup $BASE/opencl.py
  # hot patches
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/opencl.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libOpenCL.so.1')\ctypes.CDLL(ctypes.util.find_library('OpenCL'))\g" $BASE/opencl.py
  python3 -c "import tinygrad.runtime.autogen.opencl"
}

generate_hip() {
  clang2py /opt/rocm/include/hip/hip_ext.h /opt/rocm/include/hip/hiprtc.h \
  /opt/rocm/include/hip/hip_runtime_api.h /opt/rocm/include/hip/driver_types.h \
  --clang-args="-D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -x c++" -o $BASE/hip.py -l /opt/rocm/lib/libamdhip64.so
  echo "hipDeviceProp_t = hipDeviceProp_tR0600" >> $BASE/hip.py
  echo "hipGetDeviceProperties = hipGetDevicePropertiesR0600" >> $BASE/hip.py
  fixup $BASE/hip.py
  # we can trust HIP is always at /opt/rocm/lib
  #sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/hip.py
  #sed -i "s\ctypes.CDLL('/opt/rocm/lib/libhiprtc.so')\ctypes.CDLL(ctypes.util.find_library('hiprtc'))\g" $BASE/hip.py
  #sed -i "s\ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')\ctypes.CDLL(ctypes.util.find_library('amdhip64'))\g" $BASE/hip.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/hip.py
  sed -i "s\'/opt/rocm/\os.getenv('ROCM_PATH', '/opt/rocm/')+'/\g" $BASE/hip.py
  python3 -c "import tinygrad.runtime.autogen.hip"
}

generate_comgr() {
  clang2py /opt/rocm/include/amd_comgr/amd_comgr.h \
  --clang-args="-D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -x c++" -o $BASE/comgr.py -l /opt/rocm/lib/libamd_comgr.so
  fixup $BASE/comgr.py
  sed -i "s\import ctypes\import ctypes, ctypes.util, os\g" $BASE/comgr.py
  patch_dlopen $BASE/comgr.py amd_comgr "'/opt/rocm/lib/libamd_comgr.so'" "os.getenv('ROCM_PATH', '')+'/lib/libamd_comgr.so'"
  sed -i "s\ctypes.CDLL('/opt/rocm/lib/libamd_comgr.so')\_try_dlopen_amd_comgr()\g" $BASE/comgr.py
  python3 -c "import tinygrad.runtime.autogen.comgr"
}

generate_kfd() {
  clang2py /usr/include/linux/kfd_ioctl.h -o $BASE/kfd.py -k cdefstum
  fixup $BASE/kfd.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/kfd.py
  python3 -c "import tinygrad.runtime.autogen.kfd"
}

generate_cuda() {
  clang2py /usr/include/cuda.h -o $BASE/cuda.py -l /usr/lib/x86_64-linux-gnu/libcuda.so
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/cuda.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so')\ctypes.CDLL(ctypes.util.find_library('cuda'))\g" $BASE/cuda.py
  fixup $BASE/cuda.py
  python3 -c "import tinygrad.runtime.autogen.cuda"
}

generate_nvrtc() {
  clang2py /usr/local/cuda/include/nvrtc.h /usr/local/cuda/include/nvJitLink.h -o $BASE/nvrtc.py -l /usr/local/cuda/lib64/libnvrtc.so -l /usr/local/cuda/lib64/libnvJitLink.so
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/nvrtc.py
  sed -i "s\ctypes.CDLL('/usr/local/cuda/lib64/libnvrtc.so')\ctypes.CDLL(ctypes.util.find_library('nvrtc'))\g" $BASE/nvrtc.py
  sed -i "s\ctypes.CDLL('/usr/local/cuda/lib64/libnvJitLink.so')\ctypes.CDLL(ctypes.util.find_library('nvJitLink'))\g" $BASE/nvrtc.py
  fixup $BASE/nvrtc.py
  python3 -c "import tinygrad.runtime.autogen.nvrtc"
}

generate_nv() {
  NVKERN_COMMIT_HASH=d6b75a34094b0f56c2ccadf14e5d0bd515ed1ab6
  NVKERN_SRC=/tmp/open-gpu-kernel-modules-$NVKERN_COMMIT_HASH
  if [ ! -d "$NVKERN_SRC" ]; then
    git clone https://github.com/tinygrad/open-gpu-kernel-modules $NVKERN_SRC
    pushd .
    cd $NVKERN_SRC
    git reset --hard $NVKERN_COMMIT_HASH
    popd
  fi

  clang2py \
    extra/nv_gpu_driver/clc6c0qmd.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/cl0080.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/cl2080_notification.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clc56f.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clc56f.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clc56f.h \
    $NVKERN_SRC/src/nvidia/generated/g_allclasses.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/class/clc6c0.h \
    $NVKERN_SRC/kernel-open/nvidia-uvm/clc6b5.h \
    $NVKERN_SRC/kernel-open/nvidia-uvm/uvm_ioctl.h \
    $NVKERN_SRC/kernel-open/nvidia-uvm/uvm_linux_ioctl.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv_escape.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl-numbers.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv-ioctl-numa.h \
    $NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include/nv-unix-nvos-params-wrappers.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/alloc/alloc_channel.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/nvos.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrl0000/*.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrl0080/*.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrl2080/*.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrl83de/*.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrlcb33.h \
    $NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl/ctrla06c.h \
    --clang-args="-include $NVKERN_SRC/src/common/sdk/nvidia/inc/nvtypes.h -I$NVKERN_SRC/src/common/inc -I$NVKERN_SRC/kernel-open/nvidia-uvm -I$NVKERN_SRC/kernel-open/common/inc -I$NVKERN_SRC/src/common/sdk/nvidia/inc -I$NVKERN_SRC/src/nvidia/arch/nvalloc/unix/include -I$NVKERN_SRC/src/common/sdk/nvidia/inc/ctrl" \
    -o $BASE/nv_gpu.py -k cdefstum
  fixup $BASE/nv_gpu.py
  sed -i "s\(0000000001)\1\g" $BASE/nv_gpu.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/nv_gpu.py
  sed -i 's/#\?\s\([A-Za-z0-9_]\+\) = MW ( \([0-9]\+\) : \([0-9]\+\) )/\1 = (\2 , \3)/' $BASE/nv_gpu.py # NVC6C0_QMDV03_00 processing
  sed -i 's/#\sdef NVC6C0_QMD\([A-Za-z0-9_()]\+\):/def NVC6C0_QMD\1:/' $BASE/nv_gpu.py
  sed -i 's/#\s*return MW(\([0-9i()*+]\+\):\([0-9i()*+]\+\))/    return (\1 , \2)/' $BASE/nv_gpu.py
  sed -i 's/#\?\s*\(.*\)\s*=\s*\(NV\)\?BIT\(32\)\?\s*(\s*\([0-9]\+\)\s*)/\1 = (1 << \4)/' $BASE/nv_gpu.py # name = BIT(x) -> name = (1 << x)
  sed -i "s/UVM_\([A-Za-z0-9_]\+\) = \['i', '(', '\([0-9]\+\)', ')'\]/UVM_\1 = \2/" $BASE/nv_gpu.py # UVM_name = ['i', '(', '<num>', ')'] -> UVM_name = <num>

  # Parse status codes
  sed -n '1i\
nv_status_codes = {}
/^NV_STATUS_CODE/ { s/^NV_STATUS_CODE(\([^,]*\), *\([^,]*\), *"\([^"]*\)") *.*$/\1 = \2\nnv_status_codes[\1] = "\3"/; p }' $NVKERN_SRC/src/common/sdk/nvidia/inc/nvstatuscodes.h >> $BASE/nv_gpu.py

  python3 -c "import tinygrad.runtime.autogen.nv_gpu"
}

generate_amd() {
  # clang2py broken when pass -x c++ to prev headers
  clang2py extra/hip_gpu_driver/sdma_registers.h \
    --clang-args="-I/opt/rocm/include -x c++" \
    -o $BASE/amd_gpu.py

  sed 's/^\(.*\)\(\s*\/\*\)\(.*\)$/\1 #\2\3/; s/^\(\s*\*\)\(.*\)$/#\1\2/' extra/hip_gpu_driver/nvd.h >> $BASE/amd_gpu.py # comments
  sed 's/^\(.*\)\(\s*\/\*\)\(.*\)$/\1 #\2\3/; s/^\(\s*\*\)\(.*\)$/#\1\2/' extra/hip_gpu_driver/sdma_v6_0_0_pkt_open.h >> $BASE/amd_gpu.py # comments
  sed -i 's/#\s*define\s*\([^ \t]*\)(\([^)]*\))\s*\(.*\)/def \1(\2): return \3/' $BASE/amd_gpu.py # #define name(x) (smth) -> def name(x): return (smth)
  sed -i '/#\s*define\s\+\([^ \t]\+\)\s\+\([^ ]\+\)/s//\1 = \2/' $BASE/amd_gpu.py # #define name val -> name = val

  sed -e '/^reg/s/^\(reg[^ ]*\) [^ ]* \([^ ]*\) .*/\1 = \2/' \
    -e '/^ix/s/^\(ix[^ ]*\) [^ ]* \([^ ]*\) .*/\1 = \2/' \
    -e '/^[ \t]/d' \
    extra/hip_gpu_driver/gc_11_0_0.reg >> $BASE/amd_gpu.py

  fixup $BASE/amd_gpu.py
  sed -i "s\import ctypes\import ctypes, os\g" $BASE/amd_gpu.py
  python3 -c "import tinygrad.runtime.autogen.amd_gpu"
}

generate_hsa() {
  clang2py \
    /opt/rocm/include/hsa/hsa.h \
    /opt/rocm/include/hsa/hsa_ext_amd.h \
    /opt/rocm/include/hsa/amd_hsa_signal.h \
    /opt/rocm/include/hsa/amd_hsa_queue.h \
    /opt/rocm/include/hsa/amd_hsa_kernel_code.h \
    /opt/rocm/include/hsa/hsa_ext_finalize.h /opt/rocm/include/hsa/hsa_ext_image.h \
    /opt/rocm/include/hsa/hsa_ven_amd_aqlprofile.h \
    --clang-args="-I/opt/rocm/include" \
    -o $BASE/hsa.py -l /opt/rocm/lib/libhsa-runtime64.so

  fixup $BASE/hsa.py
  sed -i "s\import ctypes\import ctypes, ctypes.util, os\g" $BASE/hsa.py
  sed -i "s\ctypes.CDLL('/opt/rocm/lib/libhsa-runtime64.so')\ctypes.CDLL(os.getenv('ROCM_PATH')+'/lib/libhsa-runtime64.so' if os.getenv('ROCM_PATH') else ctypes.util.find_library('hsa-runtime64'))\g" $BASE/hsa.py
  python3 -c "import tinygrad.runtime.autogen.hsa"
}

generate_io_uring() {
  clang2py \
    /usr/include/liburing.h \
    /usr/include/linux/io_uring.h \
    -o $BASE/io_uring.py

  # clang2py can't parse defines
  sed -r '/^#define __NR_io_uring/ s/^#define __(NR_io_uring[^ ]+) (.*)$/\1 = \2/; t; d' /usr/include/asm-generic/unistd.h >> $BASE/io_uring.py # io_uring syscalls numbers
  sed -r '/^#define\s+([^ \t]+)\s+([^ \t]+)/ s/^#define\s+([^ \t]+)\s*([^/]*).*$/\1 = \2/; s/1U/1/g; s/0ULL/0/g; t; d' /usr/include/linux/io_uring.h >> $BASE/io_uring.py # #define name (val) -> name = val

  fixup $BASE/io_uring.py
}

generate_libc() {
  clang2py \
    $(dpkg -L libc6-dev | grep sys/mman.h) \
    $(dpkg -L libc6-dev | grep sys/syscall.h) \
    /usr/include/elf.h \
    /usr/include/unistd.h \
    -o $BASE/libc.py

  process_cdefines "/usr/include/elf.h" "$BASE/libc.py"

  sed -i "s\import ctypes\import ctypes, ctypes.util, os\g" $BASE/libc.py
  sed -i "s\FIXME_STUB\libc\g" $BASE/libc.py
  sed -i "s\FunctionFactoryStub()\ctypes.CDLL(ctypes.util.find_library('c'))\g" $BASE/libc.py

  fixup $BASE/libc.py
}

if [ "$1" == "opencl" ]; then generate_opencl
elif [ "$1" == "hip" ]; then generate_hip
elif [ "$1" == "comgr" ]; then generate_comgr
elif [ "$1" == "cuda" ]; then generate_cuda
elif [ "$1" == "nvrtc" ]; then generate_nvrtc
elif [ "$1" == "hsa" ]; then generate_hsa
elif [ "$1" == "kfd" ]; then generate_kfd
elif [ "$1" == "nv" ]; then generate_nv
elif [ "$1" == "amd" ]; then generate_amd
elif [ "$1" == "io_uring" ]; then generate_io_uring
elif [ "$1" == "libc" ]; then generate_libc
elif [ "$1" == "all" ]; then generate_opencl; generate_hip; generate_comgr; generate_cuda; generate_nvrtc; generate_hsa; generate_kfd; generate_nv; generate_amd; generate_io_uring; generate_libc
else echo "usage: $0 <type>"
fi
