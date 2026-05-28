"""Fix nrt_get_dmabuf_fd in libnrt.so 2.32.x via runtime hot-patch.

libnrt.so 2.32.31.0 has a bug where nrt_get_dmabuf_fd() returns
NRT_INVALID (2) without calling the kernel ioctl. The kernel driver
(aws-neuronx-dkms >= 2.x.9372.0) supports NEURON_IOCTL_DMABUF_FD and
produces valid dmabuf file descriptors.

This module patches the function in-process so that libfabric's EFA
provider can register Neuron device memory for RDMA (P2P) transfers.

Usage:
    from nkipy.vllm_plugin.nrt_dmabuf_fix import patch_nrt_dmabuf
    patch_nrt_dmabuf()  # call before NIXL agent creation
"""

import ctypes
import struct
import logging

logger = logging.getLogger(__name__)

_patched = False
_shim_lib = None


def _build_shim():
    """Build the dmabuf shim .so if not already present."""
    import os
    import subprocess
    import tempfile

    shim_path = os.path.join(
        os.path.dirname(__file__), "libnrt_dmabuf_shim.so"
    )
    if os.path.exists(shim_path):
        return shim_path

    src = r"""
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define NEURON_IOCTL_DMABUF_FD \
    ((2u << 30) | (0x4e << 8) | 107 | (8u << 16))

struct neuron_ioctl_dmabuf_fd { uint64_t va; uint64_t size; int32_t *fd; };

int nrt_get_dmabuf_fd(uint64_t va, uint64_t size, int *fd_out) {
    if (!fd_out || size == 0) return 2;
    *fd_out = -1;
    for (int i = 0; i < 16; i++) {
        char devpath[32];
        snprintf(devpath, sizeof(devpath), "/dev/neuron%d", i);
        int devfd = open(devpath, O_RDWR);
        if (devfd < 0) continue;
        int32_t result_fd = -1;
        struct neuron_ioctl_dmabuf_fd arg = { .va = va, .size = size, .fd = &result_fd };
        int rc = ioctl(devfd, NEURON_IOCTL_DMABUF_FD, &arg);
        close(devfd);
        if (rc == 0 && result_fd >= 0) { *fd_out = result_fd; return 0; }
    }
    return 1;
}
"""
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(src)
        src_path = f.name

    try:
        subprocess.check_call(
            ["gcc", "-shared", "-fPIC", "-O2", "-o", shim_path, src_path],
            stderr=subprocess.DEVNULL,
        )
    finally:
        os.unlink(src_path)

    return shim_path


def patch_nrt_dmabuf():
    """Patch nrt_get_dmabuf_fd to call the kernel ioctl directly.

    Safe to call multiple times (idempotent). Must be called before any
    NIXL agent is created (before libfabric's HMEM init resolves the symbol).
    """
    global _patched, _shim_lib
    if _patched:
        return

    try:
        libnrt = ctypes.CDLL("/opt/aws/neuron/lib/libnrt.so.1")
    except OSError:
        logger.debug("libnrt.so.1 not found, skipping dmabuf patch")
        return

    real_addr = ctypes.cast(libnrt.nrt_get_dmabuf_fd, ctypes.c_void_p).value
    if real_addr is None:
        logger.debug("nrt_get_dmabuf_fd symbol not found")
        return

    shim_path = _build_shim()
    _shim_lib = ctypes.CDLL(shim_path)
    shim_addr = ctypes.cast(
        _shim_lib.nrt_get_dmabuf_fd, ctypes.c_void_p
    ).value

    # x86_64 detour: movabs rax, <addr>; jmp rax
    jmp_code = b"\x48\xB8" + struct.pack("<Q", shim_addr) + b"\xFF\xE0"

    libc = ctypes.CDLL("libc.so.6")
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mprotect.restype = ctypes.c_int

    page_size = 4096
    page_addr = real_addr & ~(page_size - 1)
    PROT_RWX = 0x7  # READ | WRITE | EXEC

    rc = libc.mprotect(page_addr, page_size * 2, PROT_RWX)
    if rc != 0:
        logger.warning("mprotect failed for nrt_get_dmabuf_fd patch")
        return

    ctypes.memmove(real_addr, jmp_code, len(jmp_code))
    _patched = True
    logger.info(
        "Patched nrt_get_dmabuf_fd (0x%x -> 0x%x) for dmabuf ioctl bypass",
        real_addr,
        shim_addr,
    )


def is_patched():
    """Return True if the patch has been applied."""
    return _patched
