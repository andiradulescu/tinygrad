# Qualcomm OpenCL KGSL capture

Date: 2026-07-27

This records what was captured on `comma-12462a9b` while investigating the
meaning of `A6XX_SP_CS_INSTR_SIZE`. The capture used Qualcomm's proprietary
OpenCL userspace runtime. It did not capture tinygrad's direct QCOM backend.

## Device and driver

```text
Linux comma-12462a9b 4.9.103 #1 SMP PREEMPT Wed Jul 22 21:29:31 UTC 2026 aarch64 aarch64 aarch64 GNU/Linux
DEVICE_NAME=QUALCOMM Adreno(TM)
DRIVER_VERSION=OpenCL 2.0 QUALCOMM build: commit #f437276 changeid # Date: 12/06/18 Thu Local Branch:  Remote Branch:  Compiler E031.36.03.00
```

After creating a tinygrad `CLDevice` and compiling an OpenCL kernel, these
relevant mappings were present in the same process:

```text
/dev/kgsl-3d0
/usr/lib/aarch64-linux-gnu/libOpenCL.so
/usr/lib/aarch64-linux-gnu/libgsl.so
/usr/lib/aarch64-linux-gnu/libllvm-qcom.so
```

The captured libraries were:

| Library | SHA-256 | ELF build ID |
| --- | --- | --- |
| `libOpenCL.so` | `3ee7a796512d1d63bd2d54d5864a954852aab58d2c2958d62e4a9c660799b64d` | `5988c4e78d706c4a86f5c24202e2d47f9fbdf363` |
| `libgsl.so` | `1c4a28ea9d6f9e50c0cc0fb3154f734dd43d2489d343eca8c0c628218759ebc6` | `0cd444415cc0e6a5e227bf188b60f049806063d0` |
| `libllvm-qcom.so` | `fb7e6390cc25700d6935b2eef3acad85a43f247206bdf84cb4d37bd48a60b093` | `5673844ee98a5a8747a4db663700879ce6483bb6` |

## What was sniffed

`extra/qcom_gpu_driver/opencl_ioctl.py` replaced `libc.ioctl` inside the Python
process that loaded the proprietary OpenCL runtime. The hook observed the
runtime's `IOCTL_KGSL_GPU_COMMAND` calls to `/dev/kgsl-3d0`, read their command
buffers, and decoded the submitted Adreno PM4 type 4 and type 7 packets.

The test kernels were compiled and launched through
`tinygrad.runtime.ops_cl.CLDevice` and `CLProgram`. The OpenCL binary's shader
image offset and size were read from offsets `0xc0` and `0x100`. The capture
then compared that image size with:

- the value written to `REG_A6XX_SP_CS_INSTR_SIZE`
- the `num_unit` field in the shader `CP_LOAD_STATE6_FRAG` packet

This was a per-process userspace ioctl hook. It was not a system-wide kernel
trace, a USB capture, or an observation of another process. It shows what the
proprietary Qualcomm OpenCL stack submitted through KGSL for these launches.

## Raw captures

### Small kernel

OpenCL source:

```c
__kernel void add(__global int *out) { out[0] = 7; }
```

Relevant capture:

```text
 E0 -- typ 4: size=  1, REG_A6XX_SP_CS_INSTR_SIZE (1)
CAPTURE image_size=48 image_div4=12 ceil_div128=1 instr_reg=1 shader_loads=[(13, 0, 1, 0)]
```

The kernel result was `7`.

### 64-store kernel

The second kernel contained 64 literal stores from `out[0] = 0` through
`out[63] = 63`.

Relevant capture:

```text
 E0 -- typ 4: size=  1, REG_A6XX_SP_CS_INSTR_SIZE (25)
CAPTURE image_size=3096 image_div4=774 ceil_div128=25 instr_reg=25 shader_loads=[(13, 0, 25, 0)]
```

The checked boundary results were `0` and `63`.

## Result

| Shader image size | `size // 4` | `size // 128` | `ceil(size / 128)` | Vendor instruction register | Vendor shader load units |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 12 | 0 | 1 | 1 | 1 |
| 3096 | 774 | 24 | 25 | 25 | 25 |

For both captured kernels, the proprietary OpenCL runtime programmed:

```text
A6XX_SP_CS_INSTR_SIZE = round_up(shader_image_size, 128) / 128
```

The register also matched the 128-byte shader load unit count. These captures
rule out both `image_size // 4` and floor division by 128 for the proprietary
OpenCL binary path. The two non-aligned sizes are important: 48 bytes requires
one unit instead of zero, and 3096 bytes requires 25 units instead of 24.

The corresponding implementation rule is:

- IR3 binaries should use the compiler-provided `instrlen`.
- Qualcomm proprietary OpenCL binaries should use
  `round_up(image_size, 128) // 128`.

No new KGSL, GPU, IOMMU, fault, or hang messages appeared in `dmesg` during
these captures.

## Limitations

This is direct evidence from two compute kernels on one Adreno 630 device and
one Qualcomm OpenCL build. It distinguishes the candidate formulas for these
inputs, but it is not a specification for every Adreno generation or binary
format. The evidence-collection changes live only on `tools/qcom-sniffer`;
they do not by themselves validate the MSM DRM implementation.
