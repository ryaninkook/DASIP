#!/usr/bin/env python3
"""Measure the SDE sampler's cost per action chunk, before and after the
single-evaluation drift, with the real ConditionalUnet1D_Flow at the
Robomimic low-dim Square sizes (action 10, obs 23 x 2 steps, horizon 16).

"two-call" re-creates the original drift, drift(x,t) + D * score(x,t), which
runs the identical forward pass twice. "fused" is Sampler.sample_sde as it is
now. Both use the same seed, so the samples must match bit for bit.

    python scripts/bench_sde_sampler.py [--device cpu|cuda] [--repeats 5]

Weights are random: sampling cost does not depend on what the network learned.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SIP"))
from SiT.transport import create_transport, Sampler, integrators  # noqa: E402
from SI_policy.model.diffusion.conditional_unet1d_flow import ConditionalUnet1D_Flow  # noqa: E402

LAST_STEP_SIZE = 0.01  # what scripts/evaluate.py sets for SDE runs


class Counting:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *a, **k):
        self.calls += 1
        return self.fn(*a, **k)


def two_call_sampler(sampler, solver, num_steps, diffusion_form):
    """The original upstream-SiT drift: two forward passes per drift evaluation."""
    _, diffusion = sampler._Sampler__get_sde_diffusion_and_drift(diffusion_form=diffusion_form)

    def drift(x, t, model, **kw):
        return sampler.drift(x, t, model, **kw) + diffusion(x, t) * sampler.score(x, t, model, **kw)

    t0, t1 = sampler.transport.check_interval(
        sampler.transport.train_eps, sampler.transport.sample_eps, diffusion_form=diffusion_form,
        sde=True, eval=True, last_step_size=LAST_STEP_SIZE)
    sde = integrators.sde(drift, diffusion, t0=t0, t1=t1, num_steps=num_steps, sampler_type=solver)

    def sample(x, model, **kw):
        xs = sde.sample(x, model, **kw)
        ts = torch.ones(x.size(0), device=x.device) * t1
        xs.append(xs[-1] + drift(xs[-1], ts, model, **kw) * LAST_STEP_SIZE)  # "Mean" last step
        return xs

    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)

    torch.manual_seed(0)
    net = ConditionalUnet1D_Flow(
        input_dim=10, global_cond_dim=46, diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024], kernel_size=5, n_groups=8, cond_predict_scale=True,
        T_CHANGED=True, t_minus_one="t_minus_one", T=100,
    ).to(dev).eval()
    sampler = Sampler(create_transport("Linear", "velocity", None, 0.0, 0.0), "uniform", [1.5, 1.0], 0.999)
    x_init = torch.randn(args.batch, 16, 10, device=dev)
    cond = torch.randn(args.batch, 46, device=dev)

    # The SDE rows of DASIP's README table: Stochastic (50, Euler, SDE), Continuous (100, Heun, SDE).
    configs = [("Euler", 50), ("Heun", 100)]
    print(f"torch {torch.__version__}, device {dev}, threads {args.threads}, batch {args.batch}, "
          f"repeats {args.repeats}")
    print(f"{'config':<18}{'NFE two-call':>13}{'NFE fused':>11}{'ms two-call':>13}{'ms fused':>10}"
          f"{'speedup':>9}{'identical':>11}")
    for solver, steps in configs:
        fused_fn = sampler.sample_sde(sampling_method=solver, diffusion_form="sigma", last_step="Mean",
                                      last_step_size=LAST_STEP_SIZE, num_steps=steps)
        ref_fn = two_call_sampler(sampler, solver, steps, "sigma")
        times, nfe, outs = {}, {}, {}
        for name, fn in (("ref", ref_fn), ("fused", fused_fn)):
            samples = []
            for r in range(args.repeats + 1):  # first run is warm-up
                model = Counting(net.forward)
                torch.manual_seed(1000 + r)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.no_grad():
                    out = fn(x_init, model, global_cond=cond)[-1]
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if r:
                    samples.append((time.perf_counter() - t) * 1e3)
                nfe[name] = model.calls
                outs.setdefault(name, []).append(out)
            times[name] = statistics.median(samples)
        same = all(torch.equal(a, b) for a, b in zip(outs["ref"], outs["fused"]))
        print(f"{solver + ' SDE, ' + str(steps):<18}{nfe['ref']:>13}{nfe['fused']:>11}"
              f"{times['ref']:>13.1f}{times['fused']:>10.1f}{times['ref'] / times['fused']:>8.2f}x"
              f"{str(same):>11}")


if __name__ == "__main__":
    main()
