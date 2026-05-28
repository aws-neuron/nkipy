/*
 * nrt_dmabuf_shim.c — Interposes nrt_get_dmabuf_fd in libfabric's EFA provider.
 *
 * libnrt.so 2.32.31.0 has a bug where nrt_get_dmabuf_fd() returns
 * NRT_INVALID (2) without issuing the kernel ioctl. The kernel driver
 * (aws-neuronx-dkms >= 2.x.9372.0) supports NEURON_IOCTL_DMABUF_FD
 * correctly. This shim intercepts the call and issues the ioctl directly.
 *
 * Build:
 *   gcc -shared -fPIC -o libnrt_dmabuf_shim.so nrt_dmabuf_shim.c -ldl
 *
 * Use (must be loaded BEFORE libfabric resolves libnrt):
 *   LD_PRELOAD=./libnrt_dmabuf_shim.so python ...
 *
 * Or patch libfabric's symbol table at runtime (see patch_libfabric_neuron_hmem).
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define NEURON_IOCTL_BASE 0x4e
#define MAX_NEURON_DEVICES 16

struct neuron_ioctl_dmabuf_fd {
    uint64_t va;
    uint64_t size;
    int32_t *fd;
};

/* _IOR(0x4e, 107, 8) */
#define NEURON_IOCTL_DMABUF_FD \
    ((2u << 30) | (NEURON_IOCTL_BASE << 8) | 107 | (8u << 16))

#define NRT_SUCCESS 0
#define NRT_FAILURE 1
#define NRT_INVALID 2

/*
 * Fixed implementation of nrt_get_dmabuf_fd that calls the kernel directly.
 * The real libnrt function has a bug where it rejects valid VAs.
 */
int nrt_get_dmabuf_fd(uint64_t va, uint64_t size, int *fd_out) {
    if (!fd_out || size == 0) {
        return NRT_INVALID;
    }

    *fd_out = -1;

    for (int i = 0; i < MAX_NEURON_DEVICES; i++) {
        char devpath[32];
        snprintf(devpath, sizeof(devpath), "/dev/neuron%d", i);

        int devfd = open(devpath, O_RDWR);
        if (devfd < 0)
            continue;

        int32_t result_fd = -1;
        struct neuron_ioctl_dmabuf_fd arg = {
            .va = va,
            .size = size,
            .fd = &result_fd,
        };

        int rc = ioctl(devfd, NEURON_IOCTL_DMABUF_FD, &arg);
        close(devfd);

        if (rc == 0 && result_fd >= 0) {
            *fd_out = result_fd;
            return NRT_SUCCESS;
        }
    }

    return NRT_FAILURE;
}
