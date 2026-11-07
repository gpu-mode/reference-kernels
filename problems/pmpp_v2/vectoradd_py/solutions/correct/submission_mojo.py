#!POPCORN leaderboard vectoradd_v2
import importlib
from pathlib import Path

import mojo.importer  # noqa: F401
import torch
from task import input_t, output_t

MOJO_SOURCE = """
from std.math import ceildiv
from std.memory import Pointer
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import RowMajorLayout, TileTensor, row_major

comptime BLOCK = 256
comptime Layout1D = RowMajorLayout[Int]


def add_kernel(
    a: TileTensor[.float16, Layout1D, ImmUntrackedOrigin],
    b: TileTensor[.float16, Layout1D, ImmUntrackedOrigin],
    c: TileTensor[.float16, Layout1D, MutUntrackedOrigin],
    n: Int32,
):
    var idx = Int32(global_idx.x)
    if idx < n:
        c[idx] = a[idx] + b[idx]


@fieldwise_init
struct Runner(Movable, Writable):
    var ctx: DeviceContext
    var func: type_of(DeviceContext().compile_function[add_kernel]())

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Runner()")

    @staticmethod
    def py_init(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        var ctx = DeviceContext()
        var func = ctx.compile_function[add_kernel]()
        self = Self(ctx^, func^)

    @staticmethod
    def launch(
        self_ptr: Pointer[Self, MutAnyOrigin],
        a_py: PythonObject,
        b_py: PythonObject,
        c_py: PythonObject,
        n_py: PythonObject,
    ) raises:
        var n = Int(py=n_py)
        var a = TileTensor(
            Pointer[Float16, ImmUntrackedOrigin](unsafe_from_address=Int(py=a_py)),
            row_major(n),
        )
        var b = TileTensor(
            Pointer[Float16, ImmUntrackedOrigin](unsafe_from_address=Int(py=b_py)),
            row_major(n),
        )
        var c = TileTensor(
            Pointer[Float16, MutUntrackedOrigin](unsafe_from_address=Int(py=c_py)),
            row_major(n),
        )
        self_ptr[].ctx.enqueue_function(
            self_ptr[].func,
            a,
            b,
            c,
            Int32(n),
            grid_dim=ceildiv(n, BLOCK),
            block_dim=BLOCK,
        )


@export
def PyInit_mojo_vecadd_v2() abi("C") -> PythonObject:
    try:
        var mb = PythonModuleBuilder("mojo_vecadd_v2")
        _ = (
            mb.add_type[Runner]("Runner")
            .def_py_init[Runner.py_init]()
            .def_method[Runner.launch]("launch")
        )
        return mb.finalize()
    except e:
        abort(t"Failed to initialize mojo_vecadd_v2: {e}")
"""

(Path(__file__).resolve().parent / "mojo_vecadd_v2.mojo").write_text(MOJO_SOURCE)
mojo_vecadd_v2 = importlib.import_module("mojo_vecadd_v2")

_LAUNCH = mojo_vecadd_v2.Runner().launch


def custom_kernel(data: input_t) -> output_t:
    A, B, C = data
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Input/output tensors must be on GPU"
    assert A.shape == B.shape, "Input tensors must have the same shape"
    assert C.shape == A.shape, "Output tensor and input tensors must have the same shape"
    assert A.dtype == torch.float16 and B.dtype == torch.float16 and C.dtype == torch.float16, "Input/output tensors must be float16"
    _LAUNCH(A.data_ptr(), B.data_ptr(), C.data_ptr(), A.numel())
    return C
