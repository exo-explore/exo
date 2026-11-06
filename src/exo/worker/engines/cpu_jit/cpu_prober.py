import logging
import os
import platform
import subprocess

from exo.shared.types.cpu_jit import (
    CPUCacheGeometry,
    CPUCapabilityInfo,
    InstructionSetLevel,
)

logger = logging.getLogger(__name__)


class CPUIDInstructionProber:
    """Probes hardware CPU vector instruction sets with robust cross-platform fallbacks."""

    @classmethod
    def probe(cls, override_flags: set[str] | None = None) -> CPUCapabilityInfo:
        """Probe the CPU vector flags.

        If `override_flags` is provided, hardware detection is bypassed and the provided
        feature flags are evaluated instead (useful for testing legacy hardware fallbacks).
        """
        arch = platform.machine().lower()
        system = platform.system().lower()
        flags: set[str] = set()

        if override_flags is not None:
            flags = {f.lower() for f in override_flags}
        else:
            try:
                flags = cls._detect_flags(system, arch)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                logger.debug("Failed to detect hardware CPU flags: %s", exc)
                flags = set()

        has_avx512 = any(
            f in flags
            for f in ("avx512f", "avx512", "avx512vl", "avx512bw", "avx512cd")
        )
        has_avx2 = "avx2" in flags
        has_avx = "avx" in flags
        has_fma = any(f in flags for f in ("fma", "fma3", "fma4"))
        has_sse4_2 = "sse4_2" in flags or "sse4.2" in flags or "sse42" in flags
        has_sse4_1 = (
            "sse4_1" in flags
            or "sse4.1" in flags
            or "sse41" in flags
            or has_sse4_2
        )
        has_neon = any(
            f in flags for f in ("neon", "asimd", "arm_neon", "fp_armv8")
        )

        active_level: InstructionSetLevel = "SCALAR"
        if has_avx512:
            active_level = "AVX512"
        elif has_avx2:
            active_level = "AVX2"
        elif has_avx:
            active_level = "AVX"
        elif has_sse4_2:
            active_level = "SSE4_2"
        elif has_sse4_1:
            active_level = "SSE4_1"
        elif has_neon:
            active_level = "ARM_NEON"
        else:
            active_level = "SCALAR"

        physical_cores = os.cpu_count() or 1
        logical_cores = os.cpu_count() or 1

        vendor = cls._detect_vendor(system)

        return CPUCapabilityInfo(
            architecture=arch,
            vendor=vendor,
            active_level=active_level,
            supported_instruction_sets=sorted(flags),
            has_avx512=has_avx512,
            has_avx2=has_avx2,
            has_avx=has_avx,
            has_fma=has_fma,
            has_sse4_2=has_sse4_2,
            has_sse4_1=has_sse4_1,
            has_neon=has_neon,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
        )

    @classmethod
    def _detect_vendor(cls, system: str) -> str:
        proc_str = platform.processor().lower()
        if "intel" in proc_str:
            return "GenuineIntel"
        if "amd" in proc_str:
            return "AuthenticAMD"
        if "arm" in proc_str or "apple" in proc_str:
            return "Apple/ARM"
        return "GenericCPU"

    @classmethod
    def _detect_flags(cls, system: str, arch: str) -> set[str]:
        flags: set[str] = set()

        if system == "linux":
            flags = cls._detect_linux_flags()
        elif system == "darwin":
            flags = cls._detect_darwin_flags()
        elif system == "windows":
            flags = cls._detect_windows_flags(arch)
        else:
            flags = set()

        # If arch is Apple Silicon or modern 64-bit ARM, ensure NEON is flagged
        if any(a in arch for a in ("arm64", "aarch64")):
            flags.add("neon")
            flags.add("asimd")

        return flags

    @classmethod
    def _detect_linux_flags(cls) -> set[str]:
        flags: set[str] = set()
        cpuinfo_path = "/proc/cpuinfo"
        if os.path.exists(cpuinfo_path):
            with open(cpuinfo_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_lower = line.lower()
                    if line_lower.startswith(("flags", "features")):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            for flag in parts[1].split():
                                flags.add(flag.strip().lower())
                        break
        return flags

    @classmethod
    def _detect_darwin_flags(cls) -> set[str]:
        flags: set[str] = set()
        try:
            cmd = ["sysctl", "-a"]
            res = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=1.0
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    line_lower = line.lower()
                    if "hw.optional.avx512" in line_lower and "1" in line_lower:
                        flags.add("avx512")
                    if "hw.optional.avx2_0" in line_lower and "1" in line_lower:
                        flags.add("avx2")
                    if "hw.optional.avx1_0" in line_lower and "1" in line_lower:
                        flags.add("avx")
                    if "hw.optional.sse4_2" in line_lower and "1" in line_lower:
                        flags.add("sse4_2")
                    if "hw.optional.sse4_1" in line_lower and "1" in line_lower:
                        flags.add("sse4_1")
                    if (
                        "hw.optional.arm" in line_lower
                        or "hw.optional.neon" in line_lower
                    ):
                        flags.add("neon")
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("sysctl execution failed: %s", exc)
        return flags

    @classmethod
    def _detect_windows_flags(cls, arch: str) -> set[str]:
        flags: set[str] = set()
        proc_id = os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
        if "intel" in proc_id or "amd" in proc_id:
            flags.add("sse")
            flags.add("sse2")
            flags.add("sse3")
            flags.add("ssse3")
            flags.add("sse4_1")
            flags.add("sse4_2")
            if not any(old in proc_id for old in ("core 2", "pentium", "celeron")):
                flags.add("avx")
                flags.add("avx2")
                flags.add("fma")
        elif "arm" in arch:
            flags.add("neon")
            flags.add("asimd")
        return flags


class CacheTopologyProber:
    """Probes and constructs hardware CPU cache topology geometry."""

    @classmethod
    def get_cache_geometry(
        cls,
        l1d_override: int | None = None,
        l2_override: int | None = None,
        l3_override: int | None = None,
    ) -> CPUCacheGeometry:
        """Returns the hardware CPU cache geometry in bytes."""
        if (
            l1d_override is not None
            and l2_override is not None
            and l3_override is not None
        ):
            return CPUCacheGeometry(
                l1d_cache_bytes=l1d_override,
                l2_cache_bytes=l2_override,
                l3_cache_bytes=l3_override,
                cache_line_bytes=64,
            )

        l1d = 32 * 1024  # 32 KB standard
        l2 = 256 * 1024  # 256 KB standard
        l3 = 8 * 1024 * 1024  # 8 MB standard

        system = platform.system().lower()
        if system == "linux":
            l1d, l2, l3 = cls._read_linux_cache_sysfs(l1d, l2, l3)

        return CPUCacheGeometry(
            l1d_cache_bytes=l1d_override or l1d,
            l2_cache_bytes=l2_override or l2,
            l3_cache_bytes=l3_override or l3,
            cache_line_bytes=64,
        )

    @classmethod
    def _read_linux_cache_sysfs(
        cls, default_l1: int, default_l2: int, default_l3: int
    ) -> tuple[int, int, int]:
        try:
            base = "/sys/devices/system/cpu/cpu0/cache"
            if not os.path.isdir(base):
                return default_l1, default_l2, default_l3

            l1, l2, l3 = default_l1, default_l2, default_l3
            for index_dir in os.listdir(base):
                idx_path = os.path.join(base, index_dir)
                level_file = os.path.join(idx_path, "level")
                size_file = os.path.join(idx_path, "size")
                type_file = os.path.join(idx_path, "type")

                if os.path.exists(level_file) and os.path.exists(size_file):
                    with open(level_file, "r", encoding="utf-8") as f:
                        level = f.read().strip()
                    with open(size_file, "r", encoding="utf-8") as f:
                        size_str = f.read().strip().upper()

                    cache_type = ""
                    if os.path.exists(type_file):
                        with open(type_file, "r", encoding="utf-8") as f:
                            cache_type = f.read().strip().lower()

                    bytes_val = 0
                    if size_str.endswith("K"):
                        bytes_val = int(size_str[:-1]) * 1024
                    elif size_str.endswith("M"):
                        bytes_val = int(size_str[:-1]) * 1024 * 1024
                    elif size_str.isdigit():
                        bytes_val = int(size_str)

                    if bytes_val > 0:
                        if level == "1" and "data" in cache_type:
                            l1 = bytes_val
                        elif level == "2":
                            l2 = bytes_val
                        elif level == "3":
                            l3 = bytes_val

            return l1, l2, l3
        except (OSError, ValueError) as exc:
            logger.debug("Failed to read sysfs cache topology: %s", exc)
            return default_l1, default_l2, default_l3
