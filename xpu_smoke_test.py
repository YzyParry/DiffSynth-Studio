"""DiffSynth-Studio Intel GPU (XPU) smoke test -- no model download needed.

Usage, from the repository root:
    python xpu_smoke_test.py

See XPU_TESTING.md for the full test procedure (inference and training).

Every check runs independently: a failure is reported and the remaining checks
still run. FAIL = something DiffSynth needs is broken, WARN = optional feature.
Set SMOKE_DEVICE=cpu to dry-run the script itself on a machine without a GPU.
"""
import copy
import inspect
import os
import sys
import time

import torch

DEV = os.environ.get("SMOKE_DEVICE", "xpu")
results = []


def check(name, optional=False, skip_on_cpu=False):
    def run(fn):
        if skip_on_cpu and DEV == "cpu":
            results.append(("SKIP", name, "not applicable to the cpu dry run"))
            return fn
        t0 = time.time()
        try:
            detail = fn() or ""
            results.append(("PASS", name, f"{detail}  [{time.time() - t0:.1f}s]"))
        except Exception as e:  # noqa: BLE001 -- report every failure, keep going
            msg = (str(e).strip().splitlines() or [""])[0][:300]
            results.append(("WARN" if optional else "FAIL", name, f"{type(e).__name__}: {msg}"))
        return fn
    return run


def close(out, ref, tol):
    out, ref = out.detach().float().cpu(), ref.detach().float().cpu()
    err = ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
    assert err < tol, f"relative error {err:.2e} exceeds {tol:.0e}"
    return f"rel err {err:.1e}"


# ---------------------------------------------------------------- environment
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, torch.version.xpu={getattr(torch.version, 'xpu', None)}")
if DEV == "xpu":
    if not torch.xpu.is_available():
        sys.exit("torch.xpu.is_available() is False: install the Intel GPU driver and the XPU build of torch "
                 "(pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu).")
    p = torch.xpu.get_device_properties(0)
    print(f"device: {p.name} x{torch.xpu.device_count()}, {p.total_memory / 2**30:.1f} GiB, "
          f"driver {getattr(p, 'driver_version', '?')}, native fp64: {getattr(p, 'has_fp64', 'unknown')}")
print()


@check("diffsynth with XPU support is imported")
def _():
    import diffsynth
    import diffsynth.core.device as d
    assert hasattr(d, "IS_XPU_AVAILABLE"), (
        f"{diffsynth.__file__} has no XPU support: check out branch claude/awesome-galileo-ibp4tf and `pip install -e .`")
    return os.path.dirname(diffsynth.__file__)


@check("device auto-detection (get_device_type, pipeline defaults, comm backend)")
def _():
    from diffsynth.core.device.npu_compatible_device import get_device_type, get_nccl_backend
    from diffsynth.pipelines.wan_video import WanVideoPipeline
    assert get_device_type() == DEV, f"get_device_type() returned {get_device_type()!r}"
    default = inspect.signature(WanVideoPipeline.__init__).parameters["device"].default
    assert default == DEV, f"pipeline default device is {default!r}"
    backend = get_nccl_backend() if DEV != "cpu" else "-"
    assert DEV == "cpu" or backend == "xccl", f"comm backend {backend!r}"
    return f"device={get_device_type()}, backend={backend}"


@check("memory query (vram_limit / VRAM management)", skip_on_cpu=True)
def _():
    free, total = torch.xpu.mem_get_info("xpu")
    return f"free {free / 2**30:.1f} / total {total / 2**30:.1f} GiB"


@check("bf16 matmul")
def _():
    a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
    return close(a.to(DEV, torch.bfloat16) @ b.to(DEV, torch.bfloat16), a @ b, 2e-2)


@check("attention (diffsynth attention_forward)")
def _():
    from diffsynth.core.attention import attention_forward
    from diffsynth.core.attention.attention import ATTENTION_IMPLEMENTATION
    q, k, v = (torch.randn(1, 8, 512, 64) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = attention_forward(*(t.to(DEV, torch.bfloat16) for t in (q, k, v)))
    return f"backend={ATTENTION_IMPLEMENTATION}, " + close(out, ref, 3e-2)


@check("float64 basic ops")
def _():
    x = torch.linspace(0, 1, 4096, dtype=torch.float64)
    return close(torch.sin(x.to(DEV)) * 3.0, torch.sin(x) * 3.0, 1e-12)


@check("float64 / complex128 RoPE (Wan, used by many models)")
def _():
    from diffsynth.models.wan_video_dit import precompute_freqs_cis, rope_apply
    x = torch.randn(1, 256, 4 * 64)
    freqs = precompute_freqs_cis(64, 256).reshape(256, 1, 32)
    return close(rope_apply(x.to(DEV), freqs.to(DEV), 4), rope_apply(x, freqs, 4), 1e-5)


@check("autocast-disabled fp32 regions")
def _():
    from diffsynth.core.device.npu_compatible_device import get_device_type
    lin = torch.nn.Linear(64, 64).to(DEV)
    x = torch.randn(4, 64, device=DEV)
    with torch.autocast(DEV, dtype=torch.bfloat16):
        assert lin(x).dtype == torch.bfloat16, "autocast itself is not active"
        with torch.autocast(get_device_type(), enabled=False):
            y = lin(x)
    assert y.dtype == torch.float32, f"region ran in {y.dtype}"
    return "fp32 inside bf16 autocast"


@check("seeded random generator on device")
def _():
    a = torch.randn(16, generator=torch.Generator(DEV).manual_seed(0), device=DEV)
    b = torch.randn(16, generator=torch.Generator(DEV).manual_seed(0), device=DEV)
    assert torch.equal(a, b), "same seed gave different noise"


@check("VRAM management: CPU-offloaded Linear loaded to device on demand")
def _():
    from diffsynth.core.vram.layers import AutoWrappedLinear, enable_vram_management
    torch.manual_seed(0)
    ref_model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 256))
    x = torch.randn(8, 256)
    ref = ref_model(x)
    cfg = {k: torch.bfloat16 for k in ("offload_dtype", "onload_dtype", "preparing_dtype", "computation_dtype")}
    cfg.update(offload_device="cpu", onload_device="cpu", preparing_device=DEV, computation_device=DEV)
    limit = torch.xpu.mem_get_info("xpu")[1] / 2**30 - 2 if DEV == "xpu" else None
    model = enable_vram_management(copy.deepcopy(ref_model), {torch.nn.Linear: AutoWrappedLinear}, cfg, vram_limit=limit)
    for m in model.modules():
        if isinstance(m, AutoWrappedLinear):
            m.onload()
    out = model(x.to(DEV, torch.bfloat16))
    return f"weights now on {model[0].weight.device}, " + close(out, ref, 3e-2)


@check("pinned memory + non-blocking copy + synchronize (offload training)", skip_on_cpu=True)
def _():
    from diffsynth.core.device import parse_device_type
    t = torch.randn(1024, 1024).pin_memory()
    assert t.is_pinned(), "pin_memory() did not pin"
    d = t.to(DEV, non_blocking=True)
    getattr(torch, parse_device_type(d.device)).synchronize()
    back = d.to("cpu")
    assert torch.equal(back, t)


@check("torch.compile", optional=True)
def _():
    fn = torch.compile(lambda a: torch.nn.functional.gelu(a) * 2 + 1)
    x = torch.randn(4096, device=DEV)
    return close(fn(x), torch.nn.functional.gelu(x) * 2 + 1, 1e-5)


@check("XCCL backend available (multi-GPU training)", optional=True, skip_on_cpu=True)
def _():
    import torch.distributed as dist
    assert dist.is_available() and dist.is_xccl_available(), "torch.distributed.is_xccl_available() is False"


# ---------------------------------------------------------------- summary
print(f"{'':6}{'check':78}detail")
for status, name, detail in results:
    print(f"{status:6}{name:78}{detail}")
fails = [r for r in results if r[0] == "FAIL"]
print(f"\n{sum(r[0] == 'PASS' for r in results)} passed, {len(fails)} failed, "
      f"{sum(r[0] == 'WARN' for r in results)} warnings, {sum(r[0] == 'SKIP' for r in results)} skipped")
if any("float64" in r[1] for r in fails):
    print("note: float64 failures mean this GPU lacks FP64 support; models that compute RoPE in float64 (e.g. Wan) will fail.")
sys.exit(1 if fails else 0)
