#include <torch/extension.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

#define CHECK_CUTLASS(status) \\
    if (status != cutlass::Status::kSuccess) { \\
        std::cerr << "CUTLASS Error: " << cutlass::cutlassGetStatusString(status) << std::endl; \\
        return \\
    }

struct NVFP4GemmConfig {
    // 1. Data Types
    using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutA     = cutlass::layout::RowMajor;
    using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutB     = cutlass::layout::RowMajor;

    // Output is FP16. 
    // In CUTLASS, 'ElementC' is the source accumulator (for beta) 
    // and 'ElementD' is the destination.
    // Since we are doing C = A*B, we just ignore the source input.
    using ElementC    = cutlass::half_t; // Unused source
    using LayoutC     = cutlass::layout::RowMajor;
    using ElementD    = cutlass::half_t; // Actual Output
    using LayoutD     = cutlass::layout::RowMajor;

    // 2. Architecture
    using ArchTag        = cutlass::arch::Sm100;
    using OperatorClass  = cutlass::arch::OpClassBlockScaledTensorOp;
    using ElementAccum   = float;
    using ElementCompute = float;

    // 3. Tile Shapes
    using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
    using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

    // 4. Epilogue
    using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
        ElementD,       // Output type
        ElementCompute, // Compute type (alpha/beta)
        ElementC,       // Source C type
        ElementAccum    // Accumulator type
    >;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementAccum,
        ElementC, LayoutC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementD, LayoutD, 128 / cutlass::sizeof_bits<ElementD>::value,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA, 32,
        ElementB, LayoutB, 32,
        ElementAccum,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename CollectiveEpilogue::SharedStorage)>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

void gemm_nvfp4(
    torch::Tensor a, 
    torch::Tensor b,
    torch::Tensor sfa,
    torch::Tensor sfb,
    torch::Tensor c
) {
    const int M = a.size(0);
    const int K = a.size(1) * 2; // a is in fp4, so each uint8 contains 2 elements
    const int N = b.size(0);

    using Config = NVFP4GemmConfig;
    using Gemm = Config::Gemm;

    // 1. Define Strides
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    // Stride for Output C (Row Major)
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    // 2. Define Scale Factor Layouts
    using Sm100Config = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = Sm100Config::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm100Config::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    // 3. Construct Arguments
    // Note: We pass d_C as the 'Destination' (args.D).
    // We pass nullptr for 'Source' (args.C) because beta is 0.
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1}, // Problem Shape
        { // Mainloop
            reinterpret_cast<const Config::ElementA*>(a.data_ptr()), stride_A,
            reinterpret_cast<const Config::ElementB*>(b.data_ptr()), stride_B,
            reinterpret_cast<const Config::ElementA::ScaleFactorType*>(sfa.data_ptr()), layout_SFA,
            reinterpret_cast<const Config::ElementB::ScaleFactorType*>(sfb.data_ptr()), layout_SFB
        },
        { // Epilogue
            { 1.0f, 0.0f }, // alpha = 1, beta = 0
            nullptr, stride_C, // Source C (unused)
            reinterpret_cast<Config::ElementD*>(c.data_ptr()), stride_C // Destination D (Our Output C)
        }
    };

    // 4. Initialize and Run
    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    void* workspace = nullptr;
    
    if(workspace_size > 0) {
        if (cudaMallocAsync(&workspace, workspace_size) != cudaSuccess) return;
    }

    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
        if(workspace) cudaFreeAsync(workspace);
        return;
    }

    CHECK_CUTLASS(gemm.initialize(arguments, workspace));
    CHECK_CUTLASS(gemm.run());

    if(workspace) cudaFreeAsync(workspace);
}