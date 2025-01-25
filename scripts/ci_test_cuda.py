import os
import sys
import tempfile
from pathlib import Path

import pytest

if Path().resolve().name == "scripts":
    os.chdir("..")

sys.path.append("src/discord-cluster-manager")

from consts import ExitCode
from leaderboard_eval import cu_eval
from run_eval import compile_cuda_script, run_cuda_script

ref = Path("examples/identity_cuda/reference.cuh")


def test_does_not_compile():
    # input_tt is a typo, so this won't compile
    sub = """
    output_t custom_kernel(input_tt data) {   }
    """

    comp, run = run_cuda_script(
        {"eval.cu": cu_eval}, {"reference.cuh": ref.read_text(), "submission.cuh": sub}, arch=None
    )
    assert comp.success is False
    assert run.success is False
    assert comp.nvcc_found is True
    assert comp.exit_code != ExitCode.SUCCESS
    assert comp.stdout == ""
    assert 'submission.cuh(2): error: identifier "input_tt" is undefined' in comp.stderr
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
    comp, run = run_cuda_script(
        {"eval.cu": cu_eval}, {"reference.cuh": ref.read_text(), "submission.cuh": sub}, arch=None
    )
    assert comp.success is True
    assert run.success is False
    assert run.command == "./eval.out"
    assert "warming up..." in run.stdout
    assert "cudaDeviceSynchronize() at eval.cu(63) in `measure_runtime`" in run.stderr
    assert "an illegal memory access was encountered" in run.stderr
    assert run.exit_code == ExitCode.CUDA_FAIL
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
    comp, run = run_cuda_script(
        {"eval.cu": cu_eval}, {"reference.cuh": ref.read_text(), "submission.cuh": sub}, arch=None
    )
    assert comp.success is True
    assert run.success is True
    assert run.passed is False
    assert run.command == "./eval.out"
    # we never reach the benchmark part, because the test fails
    assert "warming up..." not in run.stdout
    assert "ERROR AT 0, 0" in run.stderr
    assert run.exit_code == ExitCode.VALIDATE_FAIL
    assert run.result["check"] == "fail"


def test_cuda_correct():
    sub = Path("examples/identity_cuda/submission.cuh").read_text()

    comp, run = run_cuda_script(
        {"eval.cu": cu_eval}, {"reference.cuh": ref.read_text(), "submission.cuh": sub}, arch=None
    )
    assert comp.success is True
    assert run.success is True
    assert "warming up..." in run.stdout
    assert run.exit_code == ExitCode.SUCCESS
    assert run.result["check"] == "pass"


def test_cuda_compile_validation():
    # doesn't work because test.cpp is missing, but does not raise an exception either
    assert compile_cuda_script(["test.cpp"]).success is False

    with pytest.raises(ValueError):
        compile_cuda_script(["test.cpp"], flags=["does_not_start_with_dash"])

    with pytest.raises(ValueError):
        compile_cuda_script(["test.cpp"], defines={"-not-an-identifier": "value"})

    with pytest.raises(FileNotFoundError):
        compile_cuda_script(["test.cpp"], include_dirs=["this_directory_does_not_exist"])

    with tempfile.NamedTemporaryFile() as file:
        with pytest.raises(NotADirectoryError):
            compile_cuda_script(["test.cpp"], include_dirs=[file.name])


def test_cuda_define():
    sub = """
#include "reference.cuh"
#include <iostream>

output_t custom_kernel(input_t data)
{
    TEST_FROM_DEFINE;
    return data;
}
            """
    # doesn't compile without define
    comp, run = run_cuda_script(
        {"eval.cu": cu_eval},
        {"reference.cuh": ref.read_text(), "submission.cuh": sub},
    )
    assert comp.success is False

    # compiles with define
    comp, run = run_cuda_script(
        {"eval.cu": cu_eval},
        {"reference.cuh": ref.read_text(), "submission.cuh": sub},
        defines={"TEST_FROM_DEFINE": 'std::cout << "TEST TEXT" << std::endl;'},
    )
    assert comp.success is True
    assert run.success is True
    assert run.passed is True
    assert run.command == "./eval.out"
    # check that we inserted the code
    assert "TEST TEXT" in run.stdout


def test_include_dirs(tmp_path: Path):
    (tmp_path / "include_from_path.h").write_text(
        Path("examples/identity_cuda/submission.cuh").read_text()
    )
    sub = """
#include "include_from_path.h"
"""

    # verify that naive does not work:
    comp, run = run_cuda_script(
        {"eval.cu": cu_eval}, {"reference.cuh": ref.read_text(), "submission.cuh": sub}
    )
    assert comp.success is False

    # but with include dirs, it works
    comp, run = run_cuda_script(
        {"eval.cu": cu_eval},
        {"reference.cuh": ref.read_text(), "submission.cuh": sub},
        include_dirs=[".", tmp_path],
    )

    assert comp.success is True
    assert run.success is True
    assert "warming up..." in run.stdout
    assert run.exit_code == ExitCode.SUCCESS
    assert run.result["check"] == "pass"

    # can also use generic flags argument
    comp, run = run_cuda_script(
        {"eval.cu": cu_eval},
        {"reference.cuh": ref.read_text(), "submission.cuh": sub},
        flags=["-I.", f"-I{tmp_path}"],
    )

    assert comp.success is True


def test_link_libs(tmp_path: Path):
    sub = Path("examples/identity_cuda/submission.cuh").read_text()
    sub += """
#include <cuda.h>
void function_that_uses_cuda_lib() {
    int sms;
    cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
    std::cout << sms << "\\n";
}
"""
    # without extra link libs, this doesn't compile
    comp, run = run_cuda_script(
        {"eval.cu": cu_eval}, {"reference.cuh": ref.read_text(), "submission.cuh": sub}
    )
    assert comp.success is False

    comp, run = run_cuda_script(
        {"eval.cu": cu_eval},
        {"reference.cuh": ref.read_text(), "submission.cuh": sub},
        libraries=["cuda"],
    )

    assert comp.success is True
    assert run.success is True
    assert "warming up..." in run.stdout
    assert run.exit_code == ExitCode.SUCCESS
    assert run.result["check"] == "pass"


def test_huge_output():
    sub = """
#include "reference.cuh"
#include <iostream>
output_t custom_kernel(input_t data)
{
    for(int i = 0; i < 10000; ++i) {
        std::cout << "blah blah\\n";
    }
    return data;
}
    """

    comp, run = run_cuda_script(
        {"eval.cu": cu_eval}, {"reference.cuh": ref.read_text(), "submission.cuh": sub}, arch=None
    )
    assert run.success
    assert len(run.stdout) < 16384
    assert "[...]" in run.stdout

    sub = sub.replace("std::cout", "std::cerr")

    comp, run = run_cuda_script(
        {"eval.cu": cu_eval}, {"reference.cuh": ref.read_text(), "submission.cuh": sub}, arch=None
    )
    assert run.success
    assert len(run.stderr) < 16384
    assert "[...]" in run.stderr
