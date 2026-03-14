from task import input_t, output_t
import torch
import helion
import helion.language as hl


# Per-shape configs: map input shape tuples to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
# Include all test and benchmark shapes from task.yml.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # (shape_dim_1, shape_dim_2, ...): helion.Config(...),  # TODO: replace with your config
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(...) -> ...:
        # Your Helion kernel implementation
        ...

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    # Extract shape key from input tensors to select the right kernel
    # shape_key = (...)
    # kernel = _KERNELS[shape_key]
    pass
