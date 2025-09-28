import torch
from torch.utils.cpp_extension import load_inline
import os

# this is a minimal example saying how to compile and link when you're using rocshmem
# more examples, pls see https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/ and https://github.com/ROCm/rocSHMEM/tree/develop
def test_rocshmem_compilation():
    """Test ROCshmem compilation using PyTorch's load_inline"""
    
    print("=== ROCshmem PyTorch Inline Test ===")
    
    # C++ source code for ROCshmem test
    cpp_source = """
    #include <rocshmem.hpp>
    #include <iostream>
    #include <torch/extension.h>
    
    void test_rocshmem() {
        std::cout << "Testing ROCshmem compilation..." << std::endl;
        
        // Just test that we can compile and link with rocshmem
        // Don't actually initialize since we may not have proper MPI setup
        std::cout << "ROCshmem headers included successfully!" << std::endl;
        std::cout << "Compilation test passed!" << std::endl;
    }
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("test_rocshmem", &test_rocshmem, "Test ROCshmem compilation");
    }
    """
    
    # Set up include paths and libraries
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    rocshmem_path = os.environ.get('ROCSHMEM_INSTALL_DIR', '/home/runner/rocshmem')
    ompi_path = os.environ.get('OMPI_INSTALL_DIR', '/opt/openmpi')
    # dirs that must be included
    include_dirs = [
        f"{rocm_path}/include",
        f"{rocshmem_path}/include/rocshmem",
        f"{ompi_path}/include"
    ]
    # libs that must be linked
    library_dirs = [
        f"{rocm_path}/lib",
        f"{rocshmem_path}/lib",
        f"{ompi_path}/lib"
    ]
    libraries = [
        "rocshmem",
        "mpi", 
        "amdhip64",
        "hsa-runtime64"
    ]

    ldflags = []
    for lib_dir in library_dirs:
        ldflags.append(f"-L{lib_dir}")

    for lib in libraries:
        ldflags.append(f"-l{lib}")

    extra_cflags = [f"-I{include_dir}" for include_dir in include_dirs]

    extra_ldflags = [
        "--hip-link"
    ] + ldflags
    
    try:
        # Use torch.utils.cpp_extension.load_inline to compile
        rocshmem_module = load_inline(
            name="rocshmem_test",
            cpp_sources=cpp_source,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            verbose=True
        )
        
        print("Compilation successful!")
        print("Linking successful!")
        
        # Run the test
        rocshmem_module.test_rocshmem()
        
        print("ROCshmem test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ROCshmem test failed: {e}")
        return False

if __name__ == "__main__":
    test_rocshmem_compilation()