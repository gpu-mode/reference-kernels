import os
import sys
from pathlib import Path

if Path().resolve().name == "scripts":
    os.chdir("..")

sys.path.append("src/discord-cluster-manager")

from leaderboard_eval import cu_eval
from run_eval import run_cuda_script

ref = Path("examples/identity_cuda/reference.cuh")


def test_does_not_compile():
    # input_tt is a typo, so this won't compile
    sub = """
    output_t custom_kernel(input_tt data) {   }
    """

    comp, run = run_cuda_script(cu_eval, ref.read_text(), sub, arch=None)
    assert comp.success is False
    assert run.success is False
    assert comp.nvcc_found is True
    assert comp.stdout == ""
    assert 'train.cuh(2): error: identifier "input_tt" is undefined' in comp.stderr
    assert '1 error detected in the compilation of "eval.cu".' in comp.stderr
    assert comp.command.startswith("/usr/local/cuda/bin/nvcc")
    assert "nvcc: NVIDIA (R) Cuda compiler driver" in comp.nvcc_version


def test_cuda_runtime_error():
    # deliberately causing illegal memory access
    sub = """
#include <array>
#include <vector>
#include "reference.cuh"

__global__ void copy_kernel(float* a) {
    a[-100] = 10.0;
}

output_t custom_kernel(input_t data)
{
    int blockSize = 1;
    int numBlocks = 1;
    copy_kernel<<<numBlocks, blockSize>>>(data[0].data());
    return (output_t) data;
}

    """
    comp, run = run_cuda_script(cu_eval, ref.read_text(), sub, arch=None)
    assert comp.success is True
    assert run.success is False
    assert run.command == "./eval.out"
    assert "warming up..." in run.stdout
    assert "cudaDeviceSynchronize() at eval.cu(63) in `measure_runtime`" in run.stderr
    assert "an illegal memory access was encountered" in run.stderr
    assert run.exit_code == 110
    assert len(run.result) == 0


def test_cuda_validation_fail():
    # no-op, runs fine but isn't correct
    sub = """
    #include "reference.cuh"

    output_t custom_kernel(input_t data)
    {
        output_t result;
        for (int i = 0; i < N_SIZES; ++i)
        {
            int N = Ns[i];
            result[i].resize(N);
        }
        return result;
    }

        """
    comp, run = run_cuda_script(cu_eval, ref.read_text(), sub, arch=None)
    assert comp.success is True
    assert run.success is False
    assert run.command == "./eval.out"
    # we never reach the benchmark part, because the test fails
    assert "warming up..." not in run.stdout
    assert "ERROR AT 0, 0" in run.stderr
    assert run.exit_code == 112
    assert run.result["check"] == "fail"


def test_cuda_correct():
    sub = Path("examples/identity_cuda/submission.cuh").read_text()

    comp, run = run_cuda_script(cu_eval, ref.read_text(), sub, arch=None)
    assert comp.success is True
    assert run.success is True
    assert "warming up..." in run.stdout
    assert run.exit_code == 0
    assert run.result["check"] == "pass"
