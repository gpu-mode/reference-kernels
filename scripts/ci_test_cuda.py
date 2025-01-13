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

    cout, score = run_cuda_script(cu_eval, ref.read_text(), sub, arch=None)
    assert score == 0
    assert "CUDA compilation failed" in cout


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
    cout, score = run_cuda_script(cu_eval, ref.read_text(), sub, arch=None)
    assert score == 0
    assert "Command '['./eval.out']' returned non-zero exit status 3." in cout
    assert "cudaDeviceSynchronize() at eval.cu(64) in `measure_runtime`" in cout
    assert "an illegal memory access was encountered" in cout


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
    cout, score = run_cuda_script(cu_eval, ref.read_text(), sub, arch=None)
    assert score == 0
    assert "Command '['./eval.out']' returned non-zero exit status 1." in cout
    assert "ERROR AT 0, 0" in cout


def test_cuda_correct():
    sub = Path("examples/identity_cuda/submission.cuh")

    cout, score = run_cuda_script(cu_eval, ref.read_text(), sub.read_text(), arch=None)
    assert score > 0
