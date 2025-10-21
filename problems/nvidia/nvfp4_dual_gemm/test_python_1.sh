# # BUILD_DIR=/scratch/dynamic-kernel-generator/dynamic-kernel-generator/build
BUILD_DIR=/home/scratch.vickiw_gpu/dynamic-kernel-generator/dynamic-kernel-generator/build_python
LLVM_DIR=$BUILD_DIR/llvm-prebuilt
# # BUILD_DIR=/home/scratch.ftse_gpu/workspace/dkg/build
# # BUILD_DIR=/scratch/dynamic-kernel-generator/dynamic-kernel-generator/build 
# #BUILD_DIR=/home/yanchengz/scratch_1/dynamic-kernel-generator/build_debug2
# # sudo /home/scratch.computelab/utils/driver/install_driver.py --installer=/home/builds/daily/display/x86_64/rel/gpu_drv/r580/r580_00/20250527_36037303/NVIDIA-Linux-x86_64-rel_gpu_drv_r580_r580_00-20250527_36037303-internal.run --reason="Change to tot driver"


# # BUILD_DIR=/home/scratch.nbommi_gpu/warp-phase-trace/dynamic-kernel-generator/build_main

export PYTHONPATH=$BUILD_DIR/cutlass_ir/python_packages
#export PYTHONPATH=/scratch/dynamic-kernel-generator/dynamic-kernel-generator/scripts
export CUDA_TOOLKIT_PATH=$BUILD_DIR/compiler_next
MLIR_CUDA_RUNTIME="$LLVM_DIR/lib/libmlir_cuda_runtime.so"
MLIR_C_RUNNER_UTILS="$LLVM_DIR/lib/libmlir_c_runner_utils.so"
MLIR_RUNNER_UTILS="$LLVM_DIR/lib/libmlir_runner_utils.so"
CUDA_DIALECT_RUNTIME="$BUILD_DIR/lib/libcuda_dialect_runtime.so"
export CUTE_DSL_LIBS="$MLIR_CUDA_RUNTIME:$MLIR_C_RUNNER_UTILS:$MLIR_RUNNER_UTILS:$CUDA_DIALECT_RUNTIME"


#export CUTE_DSL_PREPROCESSOR=True

# export CUTE_DSL_PRINT_IR=1
# just compile the IR but not execute it
# export CUTE_DSL_DRYRUN=1
# export CUTE_DSL_JIT_TIME_PROFILING=ON
# export CUTE_DSL_KEEP_IR=True
# export CUTE_DSL_PRINT_IR=1
# export CUTE_DSL_KEEP_CUBIN=1
# export CUTE_DSL_LINEINFO=True
# export CUTE_DSL_LOG_TO_CONSOLE=1
# export PYTHONUNBUFFERED=1
# export CUTE_DSL_KEEP_SASS=1
# whether to show detailed log in preprocessing
# export CUTE_DSL_FILTER_STACKTRACE=10
export CUTE_DSL_ARCH=sm_100a

# 
# /home/scratch.vickiw_gpu/env/bin/python3 /home/scratch.vickiw_gpu/dynamic-kernel-generator/dynamic-kernel-generator/cutlass_ir/compiler/python/examples/internal/blackwell/nvfp4_gemv_cute_layout.py
/home/scratch.vickiw_gpu/env/bin/python3 /home/scratch.vickiw_gpu/reference-kernels/problems/nvidia/nvfp4_dual_gemm/submission.py
/home/scratch.vickiw_gpu/env/bin/python3 eval.py test task.yml
/home/scratch.vickiw_gpu/env/bin/python3 eval.py benchmark task.yml
# /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/bin/cuda-gdb --args 

# /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/bin/compute-sanitizer  --tool racecheck \
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/internal/blackwell/nvfp4_gemv_cute_layout.py
# # /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/bin/compute-sanitizer  --tool racecheck \
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/internal/blackwell/nvfp4_group_gemm.py
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/internal/blackwell/nvfp4_gated_dual_gemm.py
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/internal/blackwell/nvfp4_gecccccbkvnjtrvtfreufijlfglnudnvuggvdfucidbnhk
# mm.py
# /home/scratch.vickiw_gpu/env/bin/python3 /home/scratch.vickiw_gpu/dsl-gpu-mode/gemm/nvfp4_gemm.py
# /home/scratch.vickiw_gpu/env/bin/python3 /home/scratch.vickiw_gpu/dsl-gpu-mode/gemv/nvfp4_gemv.py
# /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/bin/compute-sanitizer  --tool memcheck \
# /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/ncu --metrics gpu__time_duration \
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/blackwell/tutorial_gemm/fp16_gemm_0.py --mnk 7168,128,16384 #135us
# /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/ncu --metrics gpu__time_duration \
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/blackwell/tutorial_gemm/fp16_gemm_0.py --mnk 4096,128,7168 #62

# /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/ncu --metrics gpu__time_duration \
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/blackwell/tutorial_gemm/fp16_gemm_0.py --mnk 7168,128,2048 #26


# /home/scratch.vickiw_gpu/env/bin/python3  cutlass_ir/compiler/python/examples/internal/blackwell/nvfp4_group_gemm.py
# /home/scratch.vickiw_gpu/env/bin/python3 /home/scratch.vickiw_gpu/dsl-gpu-mode/gated_dual_gemm/nvfp4_gated_dual_gemm.py
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/internal/blackwell/nvfp4_gemv_naive.py



# print out ncu time 
# /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/ncu --metrics gpu__time_duration \
# python3 vicki/tutorial_fp16_gemm_0__.py --mnk 7168,8,512

# use sanitizer to check race contention and memref error
#  /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/bin/compute-sanitizer --tool racecheck|memcheck
#  cutlass_ir/compiler/test/python/examples/sm_100a/test_nvfp4_gemv.py

# capture ncu report 
# /home/scratch.svc_compute_arch/release/cuda_toolkit/internal/latest/ncu --check-exit-code 0 -f --set full --import-source yes --target-processes all --clock-control base --cache-control none -o gemv_4.1 \
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/internal/blackwell/nvfp4_gemv.py --m 128 --k 128 --l 2

# regular run python example
# /home/scratch.vickiw_gpu/env/bin/python3 cutlass_ir/compiler/python/examples/internal/blackwell/min_latency_hmma.py --mnkl 7168,8,512,1

# run pytest
# pytest cutlass_ir/compiler/test/python/examples/sm_80/test_sgemm.py
