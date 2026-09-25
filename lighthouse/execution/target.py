import os
import platform
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class RegisterInfo:
    width_bits: int
    count: int


class TargetInfo:
    """
    Struct to hold target architecture and feature information.
    Since this is used in a JIT context, we can safely assume the host
    architecture is the target architecture if not specified.

    Attributes:
        arch (str): The target architecture.
        features (list[str]): The list of CPU features (available on the target machine).
        filter (list[str]): The list of allowed features, if any (subset of `features`).
    """

    _cached_host: "TargetInfo | None" = None
    _override_features_stack: list[list[str] | None] = []
    _override_arch_stack: list[str | None] = []
    _override_core_count_stack: list[int | None] = []
    _override_l2_bytes_stack: list[int | None] = []
    # Fallback when the L2 size cannot be detected.
    DEFAULT_L2_CACHE_BYTES = 1024 * 1024

    def __init__(
        self,
        arch: str | None = None,
        features: list[str] | None = None,
        filter: list[str] | None = None,
        core_count: int | None = None,
        l2_cache_bytes: int | None = None,
    ):
        if arch is None and self.__class__._override_arch_stack:
            arch = self.__class__._override_arch_stack[-1]
        if features is None and self.__class__._override_features_stack:
            override = self.__class__._override_features_stack[-1]
            if override is not None:
                features = list(override)
        if core_count is None and self.__class__._override_core_count_stack:
            core_count = self.__class__._override_core_count_stack[-1]
        if l2_cache_bytes is None and self.__class__._override_l2_bytes_stack:
            l2_cache_bytes = self.__class__._override_l2_bytes_stack[-1]

        self.arch = arch if arch is not None else platform.machine()
        self.features = features if features is not None else self._get_feature_list()
        self._core_count = self._resolve_core_count(core_count)
        self._l2_cache_bytes = l2_cache_bytes
        # Pre-filter, if requested.
        if filter is not None:
            self.features = self.has_features(filter)

    @classmethod
    def host(cls) -> "TargetInfo":
        """Return a cached host TargetInfo honoring active test overrides."""
        if (
            cls._override_features_stack
            or cls._override_arch_stack
            or cls._override_core_count_stack
            or cls._override_l2_bytes_stack
        ):
            return cls()
        if cls._cached_host is None:
            cls._cached_host = cls()
        return cls._cached_host

    @classmethod
    def reset_host_cache(cls) -> None:
        """Clear cached host target information."""
        cls._cached_host = None

    @classmethod
    @contextmanager
    def override(
        cls,
        *,
        features: list[str] | None = None,
        arch: str | None = None,
        core_count: int | None = None,
        l2_cache_bytes: int | None = None,
    ):
        """Temporarily override auto-detected host target info for tests."""
        cls._override_features_stack.append(
            None if features is None else list(features)
        )
        cls._override_arch_stack.append(arch)
        cls._override_core_count_stack.append(core_count)
        cls._override_l2_bytes_stack.append(l2_cache_bytes)
        cls.reset_host_cache()
        try:
            yield
        finally:
            cls._override_features_stack.pop()
            cls._override_arch_stack.pop()
            cls._override_core_count_stack.pop()
            cls._override_l2_bytes_stack.pop()
            cls.reset_host_cache()

    def _get_feature_list(self) -> list[str]:
        """Get CPU features from the host system."""
        if platform.system() == "Darwin":
            return self._get_feature_list_darwin()
        return self._get_feature_list_linux()

    @staticmethod
    def _get_feature_list_linux() -> list[str]:
        """Get features from lscpu (Linux)."""
        flags = subprocess.run(
            "lscpu | grep Flags",
            capture_output=True,
            text=True,
            shell=True,
        ).stdout
        if not flags.startswith("Flags:"):
            raise RuntimeError(
                "Could not get CPU features from lscpu. "
                "Make sure lscpu is installed and available in PATH."
            )
        return flags.split()[1:]

    @staticmethod
    def _get_feature_list_darwin() -> list[str]:
        """Get features from sysctl (macOS)."""
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.cpu_features"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()

        # Apple Silicon: enumerate hw.optional.arm.* and hw.optional.armv8_* keys.
        result = subprocess.run(
            ["sysctl", "-a"],
            capture_output=True,
            text=True,
        )
        features = []
        for line in result.stdout.splitlines():
            if not line.startswith("hw.optional."):
                continue
            key, _, value = line.partition(":")
            if value.strip() not in ("1",):
                continue
            # Strip the "hw.optional." prefix.
            feat = key.split(".", 2)[-1]
            features.append(feat)
        if not features:
            raise RuntimeError(
                "Could not get CPU features from sysctl. "
                "Make sure sysctl is installed and available in PATH."
            )
        return features

    def has_features(self, filter: list[str]) -> list[str]:
        """
        Return a list of features that exist on both target and filter.
        """
        compatible = []
        for ext in self.features:
            if ext in filter:
                compatible.append(ext)
        return compatible

    def is_supported(self, hw_extension: str) -> bool:
        """
        Return True if the target supports the given hardware extension
        e.g., AMX or AVX512.
        """
        hw_extension = hw_extension.lower()
        return any(feature.startswith(hw_extension) for feature in self.features)

    @staticmethod
    def _resolve_core_count(core_count: int | None) -> int:
        """Return a host- or env-derived core count, unless explicitly overridden."""
        if core_count is not None:
            return max(1, int(core_count))
        omp_threads = os.environ.get("OMP_NUM_THREADS")
        if omp_threads is not None:
            try:
                return max(1, int(omp_threads))
            except ValueError:
                pass
        return max(1, os.cpu_count() or 1)

    def core_count(self) -> int:
        """Return the target's available core count for sizing heuristics."""
        return self._core_count

    def l2_cache_bytes(self) -> int:
        """Return the per-core L2 cache size, detected on Linux or a default."""
        if self._l2_cache_bytes is None:
            self._l2_cache_bytes = (
                self._detect_l2_cache_bytes() or self.DEFAULT_L2_CACHE_BYTES
            )
        return self._l2_cache_bytes

    @staticmethod
    def _detect_l2_cache_bytes() -> int | None:
        cache_dir = "/sys/devices/system/cpu/cpu0/cache"
        if not os.path.isdir(cache_dir):
            return None
        units = {"K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}
        for index in sorted(os.listdir(cache_dir)):
            path = os.path.join(cache_dir, index)
            try:
                with open(os.path.join(path, "level")) as f:
                    if f.read().strip() != "2":
                        continue
                with open(os.path.join(path, "size")) as f:
                    size = f.read().strip()
            except OSError:
                continue
            if size and size[-1] in units and size[:-1].isdigit():
                return int(size[:-1]) * units[size[-1]]
            if size.isdigit():
                return int(size)
        return None

    def vector_register_info(self) -> RegisterInfo | None:
        """Infer SIMD register info from target features."""
        if "avx512f" in self.features:
            return RegisterInfo(width_bits=512, count=32)
        if "avx2" in self.features or "avx" in self.features:
            return RegisterInfo(width_bits=256, count=16)
        if any(feature.startswith("sse") for feature in self.features):
            return RegisterInfo(width_bits=128, count=16)
        return None

    @property
    def vector_register_width_bits(self) -> int | None:
        info = self.vector_register_info()
        if info is None:
            return None
        return info.width_bits
