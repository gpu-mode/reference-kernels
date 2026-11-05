import math

import pytest
import torch
import yaml
from conftest import PROBLEMS, ROOT


@pytest.mark.parametrize("problem", PROBLEMS)
def test_reference_result_passes(problem, load_problem, examples):
    ref = load_problem(problem)
    data = examples[problem]
    output = ref.ref_kernel(data).clone()
    passed, message = ref.check_implementation(data, output)
    assert passed, message


@pytest.mark.parametrize("problem", PROBLEMS)
def test_tensor_subclasses_are_rejected_without_dispatch(
    problem, load_problem, examples
):
    class InterceptedOutput(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            raise AssertionError("checker invoked an untrusted output hook")

    ref = load_problem(problem)
    data = examples[problem]
    base = ref.ref_kernel(data).clone()
    output = torch.Tensor._make_subclass(InterceptedOutput, base, False)
    passed, message = ref.check_implementation(data, output)
    assert not passed
    assert "TYPE MISMATCH" in message


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("bad_result", ("shape", "dtype", "type", "device"))
def test_malformed_results_are_rejected(problem, bad_result, load_problem, examples):
    ref = load_problem(problem)
    data = examples[problem]
    output = ref.ref_kernel(data).clone()
    if bad_result == "shape":
        output = output.unsqueeze(0)
    elif bad_result == "dtype":
        output = output.to(
            torch.float64 if output.dtype != torch.float64 else torch.float32
        )
    elif bad_result == "type":
        output = output.tolist()
    else:
        output = torch.empty_like(output, device="meta")
    passed, message = ref.check_implementation(data, output)
    assert not passed
    assert "MISMATCH" in message


@pytest.mark.parametrize("problem", tuple(p for p in PROBLEMS if p != "histogram"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_results_are_rejected(problem, value, load_problem, examples):
    ref = load_problem(problem)
    data = examples[problem]
    output = torch.full_like(ref.ref_kernel(data), value)
    assert not ref.check_implementation(data, output)[0]


def test_histogram_does_not_accept_broadcast_scalar(load_problem, examples):
    # Every bin is four. The old comparator broadcast this scalar and accepted it.
    ref = load_problem("histogram")
    assert not ref.check_implementation(examples["histogram"], torch.tensor(4))[0]


@pytest.mark.parametrize("problem", PROBLEMS)
def test_input_independent_constant_is_rejected(problem, load_problem, examples):
    ref = load_problem(problem)
    data = examples[problem]
    output = torch.full_like(ref.ref_kernel(data), 17)
    assert not ref.check_implementation(data, output)[0]


def test_sort_rejects_large_seed_analytic_sequence(load_problem):
    # Old seed-as-mean generation collapses float32 input to one value. Its rtol
    # then accepts a monotone sequence of invented values millions apart.
    ref = load_problem("sort")
    seed, size = 10**12, 4096
    original = torch.full((size,), seed, dtype=torch.float32)
    fabricated = torch.linspace(seed - 4_000_000, seed + 4_000_000, size)
    assert not torch.equal(fabricated, original)
    assert not ref.check_implementation(
        (original, torch.empty_like(original)), fabricated
    )[0]


def test_sort_rejects_one_ulp_changes_and_lost_duplicates(load_problem):
    ref = load_problem("sort")
    values = torch.tensor([-1.0, 0.0, 1.0, 1.0, 2.0])
    expected = values.sort().values
    changed = torch.nextafter(expected, torch.full_like(expected, math.inf))
    assert not ref.check_implementation((values, torch.empty_like(values)), changed)[0]
    lost_duplicate = torch.tensor([-1.0, 0.0, 1.0, 2.0, 2.0])
    assert not ref.check_implementation(
        (values, torch.empty_like(values)), lost_duplicate
    )[0]


@pytest.mark.parametrize(
    "distribution",
    ("normal", "uniform", "duplicates", "constant", "sorted", "reverse", "outliers"),
)
@pytest.mark.parametrize("seed", (42, 10**12, 2**62 + 7123))
def test_sort_generators_keep_entropy_and_sort_correctly(
    distribution, seed, load_problem
):
    ref = load_problem("sort")
    data = ref.generate_input(1025, seed, distribution, device="cpu")
    repeated = ref.generate_input(1025, seed, distribution, device="cpu")
    assert torch.equal(data[0], repeated[0])
    assert data[0].dtype == torch.float32
    assert torch.isfinite(data[0]).all()
    assert data[0].abs().max() < 10_000
    if distribution != "constant":
        assert data[0].unique().numel() > 1
    expected = torch.sort(data[0]).values
    assert ref.check_implementation(data, expected)[0]


@pytest.mark.parametrize(
    "distribution", ("normal", "positive", "negative", "alternating", "sparse")
)
@pytest.mark.parametrize("size", (1, 1023, 4097, 262144))
def test_parallel_float32_prefix_sum_passes(distribution, size, load_problem):
    ref = load_problem("prefixsum")
    data = ref.generate_input(size, 7821, distribution, device="cpu")
    assert ref.check_implementation(data, data[0].cumsum(0))[0]


def test_prefix_sum_rejects_percentage_error(load_problem):
    ref = load_problem("prefixsum")
    values = torch.ones(262144)
    forged = values.cumsum(0) * 1.001
    assert not ref.check_implementation((values, torch.empty_like(values)), forged)[0]


def test_prefix_sum_tolerance_tracks_input_energy(load_problem):
    ref = load_problem("prefixsum")
    values = torch.zeros(262144)
    forged = torch.full_like(values, 0.001)
    assert not ref.check_implementation((values, torch.empty_like(values)), forged)[0]


@pytest.mark.parametrize("problem", ("sort", "prefixsum"))
def test_all_configured_cases_generate_and_reference_passes(problem, load_problem):
    ref = load_problem(problem)
    task = yaml.safe_load((ROOT / f"{problem}_py" / "task.yml").read_text())
    for case in task["tests"]:
        data = ref.generate_input(**case, device="cpu")
        expected = ref.ref_kernel(data).clone()
        assert ref.check_implementation(data, expected)[0], case


def test_determinism_settings_are_restored_on_exception(load_problem):
    ref = load_problem("sort")
    before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )
    with pytest.raises(RuntimeError), ref.DeterministicContext():
        raise RuntimeError("deliberate failure")
    after = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )
    assert after == before
