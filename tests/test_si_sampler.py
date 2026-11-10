"""CPU tests for the stochastic-interpolant transport used by DASIP.

These pin down the pieces DASIP's adaptive inference depends on: what
`num_steps` means, how many network evaluations (NFE) each
<steps, solver, ODE/SDE> configuration really costs, and that the samplers
integrate a known transport correctly. No simulator, dataset or GPU needed.

Run from the repository root:  python -m pytest tests -q
"""
import math

import pytest
import torch

from SiT.transport import create_transport, Sampler
from SiT.transport import integrators, path

MU, S = 2.0, 0.5  # target N(MU, S^2); prior N(0, 1)
PATHS = ["Linear", "GVP", "GVP2", "VP"]


def make_sampler(path_type="Linear"):
    transport = create_transport(path_type, "velocity", None, 0.0, 0.0)
    return Sampler(transport, "uniform", [1.5, 1.0], 0.999)


class CountingModel:
    """Wraps a callable model and counts forward passes (NFE)."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, x, t, **kwargs):
        self.calls += 1
        return self.fn(x, t, **kwargs)


def analytic_velocity(plan):
    """Exact velocity E[d_alpha*x1 + d_sigma*x0 | x_t] for x0~N(0,1), x1~N(MU,S^2).

    For Gaussian endpoints the conditional expectation is linear in x_t, so the
    probability-flow ODE with this field maps N(0,1) exactly onto N(MU, S^2).
    """

    def v(x, t, **_):
        t = path.expand_t_like_x(t, x)
        a, da = plan.compute_alpha_t(t)
        s, ds = plan.compute_sigma_t(t)
        a, da, s, ds = (torch.as_tensor(z, dtype=x.dtype) for z in (a, da, s, ds))
        var = a**2 * S**2 + s**2
        r = x - a * MU
        e_x1 = MU + a * S**2 / var * r
        e_x0 = s / var * r
        return da * e_x1 + ds * e_x0

    return v


def small_unet():
    from SI_policy.model.diffusion.conditional_unet1d_flow import ConditionalUnet1D_Flow

    torch.manual_seed(0)
    net = ConditionalUnet1D_Flow(
        input_dim=10, global_cond_dim=46, diffusion_step_embed_dim=32,
        down_dims=[32, 64], kernel_size=5, n_groups=8, cond_predict_scale=True,
        T_CHANGED=True, t_minus_one="t_minus_one", T=100,
    )
    return net.eval()


# --------------------------------------------------------------------------- paths


@pytest.mark.parametrize("path_type", PATHS)
def test_path_derivatives_match_autograd(path_type):
    plan = make_sampler(path_type).transport.path_sampler
    t = torch.linspace(0.05, 0.95, 19, dtype=torch.float64, requires_grad=True)
    for fn in (plan.compute_alpha_t, plan.compute_sigma_t):
        value, deriv = fn(t)
        value = torch.as_tensor(value, dtype=t.dtype) * torch.ones_like(t)
        (auto,) = torch.autograd.grad(value.sum(), t, allow_unused=True)
        auto = torch.zeros_like(t) if auto is None else auto
        deriv = torch.as_tensor(deriv, dtype=t.dtype) * torch.ones_like(t)
        torch.testing.assert_close(deriv, auto, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("path_type", ["Linear", "GVP"])
def test_path_endpoints(path_type):
    plan = make_sampler(path_type).transport.path_sampler
    t = torch.tensor([0.0, 1.0], dtype=torch.float64)
    alpha, _ = plan.compute_alpha_t(t)
    sigma, _ = plan.compute_sigma_t(t)
    torch.testing.assert_close(torch.as_tensor(alpha, dtype=t.dtype), torch.tensor([0.0, 1.0], dtype=t.dtype))
    torch.testing.assert_close(torch.as_tensor(sigma, dtype=t.dtype), torch.tensor([1.0, 0.0], dtype=t.dtype), atol=1e-12, rtol=0)


@pytest.mark.parametrize("path_type", PATHS)
def test_plan_target_is_time_derivative_of_xt(path_type):
    plan = make_sampler(path_type).transport.path_sampler
    torch.manual_seed(0)
    x0 = torch.randn(8, 16, 10, dtype=torch.float64)
    x1 = torch.randn(8, 16, 10, dtype=torch.float64)
    t = torch.rand(8, dtype=torch.float64) * 0.9 + 0.05
    _, _, ut = plan.plan(t, x0, x1)
    xt_of_t = lambda tt: plan.compute_xt(tt, x0, x1)
    _, dxt_dt = torch.autograd.functional.jvp(xt_of_t, (t,), (torch.ones_like(t),))
    torch.testing.assert_close(ut, dxt_dt)


# ----------------------------------------------------------------- ODE sampling


@pytest.mark.parametrize("path_type", ["Linear", "GVP"])
def test_ode_euler_converges_to_analytic_target(path_type):
    sampler = make_sampler(path_type)
    v = analytic_velocity(sampler.transport.path_sampler)
    torch.manual_seed(0)
    x0 = torch.randn(20000, 1, 1, dtype=torch.float64)
    errors = []
    for num_steps in (3, 5, 11, 101):
        out = sampler.sample_ode(sampling_method="euler", num_steps=num_steps)(x0, v)[-1]
        # Compare against the exact pushforward of the same noise, not sample stats.
        exact = MU + S * x0
        errors.append((out - exact).abs().max().item())
    assert all(a > b for a, b in zip(errors, errors[1:])), errors
    # Euler is first order: 10x the steps (10 -> 100) gives ~10x less error.
    assert errors[2] / errors[3] > 8, errors
    assert errors[-1] < 0.05, errors


@pytest.mark.parametrize("num_steps", [2, 3, 5, 10, 100])
def test_ode_euler_nfe_is_num_steps_minus_one(num_steps):
    sampler = make_sampler()
    model = CountingModel(analytic_velocity(sampler.transport.path_sampler))
    sampler.sample_ode(sampling_method="euler", num_steps=num_steps)(torch.randn(4, 16, 10), model)
    assert model.calls == num_steps - 1


def test_ode_heun_is_accepted_and_beats_euler():
    """The policies pass DDPM.lower(), i.e. "heun"; torchdiffeq calls it "heun2"."""
    sampler = make_sampler()
    v = analytic_velocity(sampler.transport.path_sampler)
    torch.manual_seed(0)
    x0 = torch.randn(20000, 1, 1, dtype=torch.float64)
    exact = MU + S * x0
    model = CountingModel(v)
    heun = sampler.sample_ode(sampling_method="heun", num_steps=6)(x0, model)[-1]
    assert model.calls == 2 * 5
    euler = sampler.sample_ode(sampling_method="euler", num_steps=11)(x0, v)[-1]  # same NFE
    assert (heun - exact).abs().max() < (euler - exact).abs().max()


@pytest.mark.parametrize("mode", ["ode", "sde"])
def test_single_point_grid_is_rejected(mode):
    """num_steps=1 used to make the ODE return the prior noise untouched (0 NFE)."""
    sampler = make_sampler()
    with pytest.raises(ValueError, match="num_steps=1"):
        if mode == "ode":
            sampler.sample_ode(sampling_method="euler", num_steps=1)
        else:
            sampler.sample_sde(sampling_method="Euler", diffusion_form="sigma",
                               last_step="Mean", last_step_size=0.01, num_steps=1)


# ----------------------------------------------------------------- SDE sampling


@pytest.mark.parametrize("solver,per_step", [("Euler", 1), ("Heun", 2)])
@pytest.mark.parametrize("num_steps", [3, 10, 50])
def test_sde_nfe_one_network_call_per_drift(solver, per_step, num_steps):
    sampler = make_sampler()
    model = CountingModel(analytic_velocity(sampler.transport.path_sampler))
    fn = sampler.sample_sde(sampling_method=solver, diffusion_form="sigma",
                            last_step="Mean", last_step_size=0.01, num_steps=num_steps)
    xs = fn(torch.randn(4, 16, 10), model)
    assert len(xs) == num_steps
    # (num_steps - 1) integrator steps plus the "Mean" last step.
    assert model.calls == per_step * (num_steps - 1) + 1


@pytest.mark.parametrize("path_type", PATHS)
@pytest.mark.parametrize("diffusion_form", ["sigma", "SBDM", "linear", "increasing-decreasing"])
def test_sde_drift_bit_identical_to_two_call_reference(path_type, diffusion_form):
    """The fused drift must equal drift(x,t) + D(x,t) * score(x,t) exactly."""
    sampler = make_sampler(path_type)
    net = small_unet()
    torch.manual_seed(1)
    x = torch.randn(3, 16, 10)
    cond = torch.randn(3, 46)
    t = torch.full((3,), 0.37)
    fused, diffusion = sampler._Sampler__get_sde_diffusion_and_drift(diffusion_form=diffusion_form)
    with torch.no_grad():
        got = fused(x, t, net.forward, global_cond=cond)
        ref = (sampler.drift(x, t, net.forward, global_cond=cond)
               + diffusion(x, t) * sampler.score(x, t, net.forward, global_cond=cond))
    assert torch.equal(got, ref)


@pytest.mark.parametrize("solver", ["Euler", "Heun"])
def test_sde_trajectory_bit_identical_to_two_call_reference(solver):
    """Same seed, same samples: only the number of forward passes changes."""
    sampler = make_sampler("Linear")
    net = small_unet()
    cond = torch.randn(2, 46)
    num_steps = 12
    _, diffusion = sampler._Sampler__get_sde_diffusion_and_drift(diffusion_form="sigma")

    def reference_drift(x, t, model, **kw):  # the original (upstream SiT) formula
        return sampler.drift(x, t, model, **kw) + diffusion(x, t) * sampler.score(x, t, model, **kw)

    t0, t1 = sampler.transport.check_interval(0.0, 0.0, diffusion_form="sigma", sde=True,
                                              eval=True, last_step_size=0.01)
    ref_sde = integrators.sde(reference_drift, diffusion, t0=t0, t1=t1,
                              num_steps=num_steps, sampler_type=solver)
    x_init = torch.randn(2, 16, 10)

    ref_model, new_model = CountingModel(net.forward), CountingModel(net.forward)
    torch.manual_seed(123)
    ref = ref_sde.sample(x_init, ref_model, global_cond=cond)[-1]
    torch.manual_seed(123)
    fn = sampler.sample_sde(sampling_method=solver, diffusion_form="sigma",
                            last_step="Mean", last_step_size=0.01, num_steps=num_steps)
    new = fn(x_init, new_model, global_cond=cond)[-2]  # [-1] is the extra last step
    assert torch.equal(new, ref)
    per_step = 2 if solver == "Heun" else 1
    assert ref_model.calls == 2 * per_step * (num_steps - 1)
    assert new_model.calls == per_step * (num_steps - 1) + 1


def test_sde_euler_matches_target_statistics():
    """Euler-Maruyama with the analytic field lands on N(MU, S^2) in distribution."""
    sampler = make_sampler("Linear")
    v = analytic_velocity(sampler.transport.path_sampler)
    torch.manual_seed(0)
    x0 = torch.randn(40000, 1, 1, dtype=torch.float64)
    out = sampler.sample_sde(sampling_method="Euler", diffusion_form="sigma", last_step="Mean",
                             last_step_size=0.01, num_steps=200)(x0, v)[-1]
    assert abs(out.mean().item() - MU) < 0.03
    assert abs(out.std().item() - S) < 0.03
