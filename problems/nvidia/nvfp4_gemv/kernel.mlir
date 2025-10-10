!memref_gmem_f32 = !cute.memref<f32, gmem, "((1,1024,1),(?,?,?)):((0,?{i64},0),(?{i64},?{i64 div=1024},1))">
!memref_gmem_f32_1 = !cute.memref<f32, gmem, "((1,1024,1)):((0,?{i64},0))">
!memref_gmem_f32_2 = !cute.memref<f32, gmem, "(128,8):(?{i64 div=8},?{i64})">
!memref_gmem_f32_3 = !cute.memref<f32, gmem, "(8):(?{i64})">
!memref_gmem_f32_4 = !cute.memref<f32, gmem, "(?,?,?):(?{i64},?{i64},1)">
!memref_gmem_f4E2M1FN = !cute.memref<f4E2M1FN, gmem, align<16>, "(?,?,?):(?{i64},1,?{i64})">
!memref_gmem_i8 = !cute.memref<i8, gmem, align<16>, "((1,512,1),(?,?,?)):((0,1,0),(?{i64},512,?{i64}))">
!memref_gmem_i8_1 = !cute.memref<i8, gmem, "((1,512,1)):((0,1,0))">
!memref_gmem_i8_2 = !cute.memref<i8, gmem, "(128,4):(4,1)">
!memref_gmem_i8_3 = !cute.memref<i8, gmem, "(4):(1)">
!memref_gmem_i8_4 = !cute.memref<i8, gmem, align<16>, "(?,?,?):(?{i64},1,?{i64})">
!memref_rmem_f32 = !cute.memref<f32, rmem, align<32>, "8:1">
!memref_rmem_i8 = !cute.memref<i8, rmem, align<32>, "4:1">
module attributes {gpu.container_module} {
  gpu.module @kernels {
    func.func public @kernel_cutlass__convert_kernel_tensorptrf32gmemo11024100div10241_tensorptri8gmemalign16o15121010512_tensor000o1102410110101024112____Float32_Float4E2M1FN_0(%arg0: !memref_gmem_f32, %arg1: !memref_gmem_i8, %arg2: !cute.coord_tensor<"(0,0,0)", "((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">, %arg3: !cute.layout<"(128,8):(8,1)">, %arg4: !cute.layout<"(128,4):(4,1)">, %arg5: i32, %arg6: i32, %arg7: i32) attributes {cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 128, 1, 1>} {
      %iter = cute.get_iter(%arg0) : !memref_gmem_f32
      %iter_0 = cute.get_iter(%arg1) : !memref_gmem_i8
      %iter_1 = cute.get_iter(%arg2) : !cute.coord_tensor<"(0,0,0)", "((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">
      %tup = cute.deref_arith_tuple_iter(%iter_1) : !cute.arith_tuple_iter<"(0,0,0)">
      %e0, %e1, %e2 = cute.get_leaves(%tup) : !cute.int_tuple<"(0,0,0)">
      %iter_2 = cute.get_iter(%arg0) : !memref_gmem_f32
      %iter_3 = cute.get_iter(%arg1) : !memref_gmem_i8
      %iter_4 = cute.get_iter(%arg2) : !cute.coord_tensor<"(0,0,0)", "((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">
      %tup_5 = cute.deref_arith_tuple_iter(%iter_4) : !cute.arith_tuple_iter<"(0,0,0)">
      %e0_6, %e1_7, %e2_8 = cute.get_leaves(%tup_5) : !cute.int_tuple<"(0,0,0)">
      %lay = cute.get_layout(%arg0) : !memref_gmem_f32
      %0 = cute.get_shape(%lay) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,?{i64},0),(?{i64},?{i64 div=1024},1))">) -> !cute.shape<"((1,1024,1),(?,?,?))">
      %e0_9, %e1_10, %e2_11, %e3, %e4, %e5 = cute.get_leaves(%0) : !cute.shape<"((1,1024,1),(?,?,?))">
      %itup = cute.to_int_tuple(%e3) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %1 = cute.get_scalars(%itup) : !cute.int_tuple<"?">
      %itup_12 = cute.to_int_tuple(%e4) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %2 = cute.get_scalars(%itup_12) : !cute.int_tuple<"?">
      %itup_13 = cute.to_int_tuple(%e5) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %3 = cute.get_scalars(%itup_13) : !cute.int_tuple<"?">
      %4 = cute.get_stride(%lay) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,?{i64},0),(?{i64},?{i64 div=1024},1))">) -> !cute.stride<"((0,?{i64},0),(?{i64},?{i64 div=1024},1))">
      %e0_14, %e1_15, %e2_16, %e3_17, %e4_18, %e5_19 = cute.get_leaves(%4) : !cute.stride<"((0,?{i64},0),(?{i64},?{i64 div=1024},1))">
      %itup_20 = cute.to_int_tuple(%e1_15) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
      %5 = cute.get_scalars(%itup_20) : !cute.int_tuple<"?{i64}">
      %itup_21 = cute.to_int_tuple(%e3_17) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
      %6 = cute.get_scalars(%itup_21) : !cute.int_tuple<"?{i64}">
      %itup_22 = cute.to_int_tuple(%e4_18) : !cute.stride<"?{i64 div=1024}"> to !cute.int_tuple<"?{i64 div=1024}">
      %7 = cute.get_scalars(%itup_22) : !cute.int_tuple<"?{i64 div=1024}">
      %lay_23 = cute.get_layout(%arg1) : !memref_gmem_i8
      %8 = cute.get_shape(%lay_23) : (!cute.layout<"((1,512,1),(?,?,?)):((0,1,0),(?{i64},512,?{i64}))">) -> !cute.shape<"((1,512,1),(?,?,?))">
      %e0_24, %e1_25, %e2_26, %e3_27, %e4_28, %e5_29 = cute.get_leaves(%8) : !cute.shape<"((1,512,1),(?,?,?))">
      %itup_30 = cute.to_int_tuple(%e3_27) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %9 = cute.get_scalars(%itup_30) : !cute.int_tuple<"?">
      %itup_31 = cute.to_int_tuple(%e4_28) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %10 = cute.get_scalars(%itup_31) : !cute.int_tuple<"?">
      %itup_32 = cute.to_int_tuple(%e5_29) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %11 = cute.get_scalars(%itup_32) : !cute.int_tuple<"?">
      %12 = cute.get_stride(%lay_23) : (!cute.layout<"((1,512,1),(?,?,?)):((0,1,0),(?{i64},512,?{i64}))">) -> !cute.stride<"((0,1,0),(?{i64},512,?{i64}))">
      %e0_33, %e1_34, %e2_35, %e3_36, %e4_37, %e5_38 = cute.get_leaves(%12) : !cute.stride<"((0,1,0),(?{i64},512,?{i64}))">
      %itup_39 = cute.to_int_tuple(%e3_36) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
      %13 = cute.get_scalars(%itup_39) : !cute.int_tuple<"?{i64}">
      %itup_40 = cute.to_int_tuple(%e5_38) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
      %14 = cute.get_scalars(%itup_40) : !cute.int_tuple<"?{i64}">
      %lay_41 = cute.get_layout(%arg2) : !cute.coord_tensor<"(0,0,0)", "((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">
      %15 = cute.get_shape(%lay_41) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">) -> !cute.shape<"((1,1024,1),(?,?,?))">
      %e0_42, %e1_43, %e2_44, %e3_45, %e4_46, %e5_47 = cute.get_leaves(%15) : !cute.shape<"((1,1024,1),(?,?,?))">
      %itup_48 = cute.to_int_tuple(%e3_45) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %16 = cute.get_scalars(%itup_48) : !cute.int_tuple<"?">
      %itup_49 = cute.to_int_tuple(%e4_46) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %17 = cute.get_scalars(%itup_49) : !cute.int_tuple<"?">
      %itup_50 = cute.to_int_tuple(%e5_47) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %18 = cute.get_scalars(%itup_50) : !cute.int_tuple<"?">
      %19 = cute.get_stride(%lay_41) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">) -> !cute.stride<"((0,1@1,0),(1@0,1024@1,1@2))">
      %e0_51, %e1_52, %e2_53, %e3_54, %e4_55, %e5_56 = cute.get_leaves(%19) : !cute.stride<"((0,1@1,0),(1@0,1024@1,1@2))">
      %20 = cute.get_shape(%arg3) : (!cute.layout<"(128,8):(8,1)">) -> !cute.shape<"(128,8)">
      %e0_57, %e1_58 = cute.get_leaves(%20) : !cute.shape<"(128,8)">
      %21 = cute.get_stride(%arg3) : (!cute.layout<"(128,8):(8,1)">) -> !cute.stride<"(8,1)">
      %e0_59, %e1_60 = cute.get_leaves(%21) : !cute.stride<"(8,1)">
      %22 = cute.get_shape(%arg4) : (!cute.layout<"(128,4):(4,1)">) -> !cute.shape<"(128,4)">
      %e0_61, %e1_62 = cute.get_leaves(%22) : !cute.shape<"(128,4)">
      %23 = cute.get_stride(%arg4) : (!cute.layout<"(128,4):(4,1)">) -> !cute.stride<"(4,1)">
      %e0_63, %e1_64 = cute.get_leaves(%23) : !cute.stride<"(4,1)">
      %24 = cute.get_shape(%lay) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,?{i64},0),(?{i64},?{i64 div=1024},1))">) -> !cute.shape<"((1,1024,1),(?,?,?))">
      %e0_65, %e1_66, %e2_67, %e3_68, %e4_69, %e5_70 = cute.get_leaves(%24) : !cute.shape<"((1,1024,1),(?,?,?))">
      %itup_71 = cute.to_int_tuple(%e3_68) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %25 = cute.get_scalars(%itup_71) : !cute.int_tuple<"?">
      %itup_72 = cute.to_int_tuple(%e4_69) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %26 = cute.get_scalars(%itup_72) : !cute.int_tuple<"?">
      %itup_73 = cute.to_int_tuple(%e5_70) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %27 = cute.get_scalars(%itup_73) : !cute.int_tuple<"?">
      %28 = cute.get_stride(%lay) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,?{i64},0),(?{i64},?{i64 div=1024},1))">) -> !cute.stride<"((0,?{i64},0),(?{i64},?{i64 div=1024},1))">
      %e0_74, %e1_75, %e2_76, %e3_77, %e4_78, %e5_79 = cute.get_leaves(%28) : !cute.stride<"((0,?{i64},0),(?{i64},?{i64 div=1024},1))">
      %itup_80 = cute.to_int_tuple(%e1_75) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
      %29 = cute.get_scalars(%itup_80) : !cute.int_tuple<"?{i64}">
      %itup_81 = cute.to_int_tuple(%e3_77) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
      %30 = cute.get_scalars(%itup_81) : !cute.int_tuple<"?{i64}">
      %itup_82 = cute.to_int_tuple(%e4_78) : !cute.stride<"?{i64 div=1024}"> to !cute.int_tuple<"?{i64 div=1024}">
      %31 = cute.get_scalars(%itup_82) : !cute.int_tuple<"?{i64 div=1024}">
      %32 = cute.get_shape(%lay_23) : (!cute.layout<"((1,512,1),(?,?,?)):((0,1,0),(?{i64},512,?{i64}))">) -> !cute.shape<"((1,512,1),(?,?,?))">
      %e0_83, %e1_84, %e2_85, %e3_86, %e4_87, %e5_88 = cute.get_leaves(%32) : !cute.shape<"((1,512,1),(?,?,?))">
      %itup_89 = cute.to_int_tuple(%e3_86) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %33 = cute.get_scalars(%itup_89) : !cute.int_tuple<"?">
      %itup_90 = cute.to_int_tuple(%e4_87) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %34 = cute.get_scalars(%itup_90) : !cute.int_tuple<"?">
      %itup_91 = cute.to_int_tuple(%e5_88) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %35 = cute.get_scalars(%itup_91) : !cute.int_tuple<"?">
      %36 = cute.get_stride(%lay_23) : (!cute.layout<"((1,512,1),(?,?,?)):((0,1,0),(?{i64},512,?{i64}))">) -> !cute.stride<"((0,1,0),(?{i64},512,?{i64}))">
      %e0_92, %e1_93, %e2_94, %e3_95, %e4_96, %e5_97 = cute.get_leaves(%36) : !cute.stride<"((0,1,0),(?{i64},512,?{i64}))">
      %itup_98 = cute.to_int_tuple(%e3_95) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
      %37 = cute.get_scalars(%itup_98) : !cute.int_tuple<"?{i64}">
      %itup_99 = cute.to_int_tuple(%e5_97) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
      %38 = cute.get_scalars(%itup_99) : !cute.int_tuple<"?{i64}">
      %39 = cute.get_shape(%lay_41) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">) -> !cute.shape<"((1,1024,1),(?,?,?))">
      %e0_100, %e1_101, %e2_102, %e3_103, %e4_104, %e5_105 = cute.get_leaves(%39) : !cute.shape<"((1,1024,1),(?,?,?))">
      %itup_106 = cute.to_int_tuple(%e3_103) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %40 = cute.get_scalars(%itup_106) : !cute.int_tuple<"?">
      %itup_107 = cute.to_int_tuple(%e4_104) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %41 = cute.get_scalars(%itup_107) : !cute.int_tuple<"?">
      %itup_108 = cute.to_int_tuple(%e5_105) : !cute.shape<"?"> to !cute.int_tuple<"?">
      %42 = cute.get_scalars(%itup_108) : !cute.int_tuple<"?">
      %43 = cute.get_stride(%lay_41) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">) -> !cute.stride<"((0,1@1,0),(1@0,1024@1,1@2))">
      %e0_109, %e1_110, %e2_111, %e3_112, %e4_113, %e5_114 = cute.get_leaves(%43) : !cute.stride<"((0,1@1,0),(1@0,1024@1,1@2))">
      %44 = cute.get_shape(%arg3) : (!cute.layout<"(128,8):(8,1)">) -> !cute.shape<"(128,8)">
      %e0_115, %e1_116 = cute.get_leaves(%44) : !cute.shape<"(128,8)">
      %45 = cute.get_stride(%arg3) : (!cute.layout<"(128,8):(8,1)">) -> !cute.stride<"(8,1)">
      %e0_117, %e1_118 = cute.get_leaves(%45) : !cute.stride<"(8,1)">
      %46 = cute.get_shape(%arg4) : (!cute.layout<"(128,4):(4,1)">) -> !cute.shape<"(128,4)">
      %e0_119, %e1_120 = cute.get_leaves(%46) : !cute.shape<"(128,4)">
      %47 = cute.get_stride(%arg4) : (!cute.layout<"(128,4):(4,1)">) -> !cute.stride<"(4,1)">
      %e0_121, %e1_122 = cute.get_leaves(%47) : !cute.stride<"(4,1)">
      %48 = nvvm.read.ptx.sreg.tid.x : i32
      %49 = nvvm.read.ptx.sreg.ctaid.x : i32
      %coord = cute.make_coord(%49) : (i32) -> !cute.coord<"(_,?)">
      %slice = cute.slice(%arg0, %coord) : !memref_gmem_f32, !cute.coord<"(_,?)">
      %iter_123 = cute.get_iter(%slice) : !memref_gmem_f32_1
      %iter_124 = cute.get_iter(%slice) : !memref_gmem_f32_1
      %coord_125 = cute.make_coord(%49) : (i32) -> !cute.coord<"(_,?)">
      %slice_126 = cute.slice(%arg1, %coord_125) : !memref_gmem_i8, !cute.coord<"(_,?)">
      %iter_127 = cute.get_iter(%slice_126) : !memref_gmem_i8_1
      %iter_128 = cute.get_iter(%slice_126) : !memref_gmem_i8_1
      %coord_129 = cute.make_coord(%49) : (i32) -> !cute.coord<"(_,?)">
      %slice_130 = cute.slice(%arg2, %coord_129) : !cute.coord_tensor<"(0,0,0)", "((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">, !cute.coord<"(_,?)">
      %iter_131 = cute.get_iter(%slice_130) : !cute.coord_tensor<"(?,?{div=1024},?)", "((1,1024,1)):((0,1@1,0))">
      %tup_132 = cute.deref_arith_tuple_iter(%iter_131) : !cute.arith_tuple_iter<"(?,?{div=1024},?)">
      %e0_133, %e1_134, %e2_135 = cute.get_leaves(%tup_132) : !cute.int_tuple<"(?,?{div=1024},?)">
      %50 = cute.get_scalars(%e0_133) : !cute.int_tuple<"?">
      %51 = cute.get_scalars(%e1_134) : !cute.int_tuple<"?{div=1024}">
      %52 = cute.get_scalars(%e2_135) : !cute.int_tuple<"?">
      %iter_136 = cute.get_iter(%slice_130) : !cute.coord_tensor<"(?,?{div=1024},?)", "((1,1024,1)):((0,1@1,0))">
      %tup_137 = cute.deref_arith_tuple_iter(%iter_136) : !cute.arith_tuple_iter<"(?,?{div=1024},?)">
      %e0_138, %e1_139, %e2_140 = cute.get_leaves(%tup_137) : !cute.int_tuple<"(?,?{div=1024},?)">
      %53 = cute.get_scalars(%e0_138) : !cute.int_tuple<"?">
      %54 = cute.get_scalars(%e1_139) : !cute.int_tuple<"?{div=1024}">
      %55 = cute.get_scalars(%e2_140) : !cute.int_tuple<"?">
      %56 = cute.composition(%slice, %arg3) : (!memref_gmem_f32_1, !cute.layout<"(128,8):(8,1)">) -> !memref_gmem_f32_2
      %iter_141 = cute.get_iter(%56) : !memref_gmem_f32_2
      %57 = cute.composition(%slice_126, %arg4) : (!memref_gmem_i8_1, !cute.layout<"(128,4):(4,1)">) -> !memref_gmem_i8_2
      %iter_142 = cute.get_iter(%57) : !memref_gmem_i8_2
      %58 = cute.composition(%slice_130, %arg3) : (!cute.coord_tensor<"(?,?{div=1024},?)", "((1,1024,1)):((0,1@1,0))">, !cute.layout<"(128,8):(8,1)">) -> !cute.coord_tensor<"(?,?{div=1024},?)", "(128,8):(8@1,1@1)">
      %iter_143 = cute.get_iter(%58) : !cute.coord_tensor<"(?,?{div=1024},?)", "(128,8):(8@1,1@1)">
      %tup_144 = cute.deref_arith_tuple_iter(%iter_143) : !cute.arith_tuple_iter<"(?,?{div=1024},?)">
      %e0_145, %e1_146, %e2_147 = cute.get_leaves(%tup_144) : !cute.int_tuple<"(?,?{div=1024},?)">
      %59 = cute.get_scalars(%e0_145) : !cute.int_tuple<"?">
      %60 = cute.get_scalars(%e1_146) : !cute.int_tuple<"?{div=1024}">
      %61 = cute.get_scalars(%e2_147) : !cute.int_tuple<"?">
      %coord_148 = cute.make_coord(%48) : (i32) -> !cute.coord<"(?,_)">
      %slice_149 = cute.slice(%56, %coord_148) : !memref_gmem_f32_2, !cute.coord<"(?,_)">
      %iter_150 = cute.get_iter(%slice_149) : !memref_gmem_f32_3
      %iter_151 = cute.get_iter(%slice_149) : !memref_gmem_f32_3
      %coord_152 = cute.make_coord(%48) : (i32) -> !cute.coord<"(?,_)">
      %slice_153 = cute.slice(%57, %coord_152) : !memref_gmem_i8_2, !cute.coord<"(?,_)">
      %iter_154 = cute.get_iter(%slice_153) : !memref_gmem_i8_3
      %iter_155 = cute.get_iter(%slice_153) : !memref_gmem_i8_3
      %coord_156 = cute.make_coord(%48) : (i32) -> !cute.coord<"(?,_)">
      %slice_157 = cute.slice(%58, %coord_156) : !cute.coord_tensor<"(?,?{div=1024},?)", "(128,8):(8@1,1@1)">, !cute.coord<"(?,_)">
      %iter_158 = cute.get_iter(%slice_157) : !cute.coord_tensor<"(?,?{div=8},?)", "(8):(1@1)">
      %tup_159 = cute.deref_arith_tuple_iter(%iter_158) : !cute.arith_tuple_iter<"(?,?{div=8},?)">
      %e0_160, %e1_161, %e2_162 = cute.get_leaves(%tup_159) : !cute.int_tuple<"(?,?{div=8},?)">
      %62 = cute.get_scalars(%e0_160) : !cute.int_tuple<"?">
      %63 = cute.get_scalars(%e1_161) : !cute.int_tuple<"?{div=8}">
      %64 = cute.get_scalars(%e2_162) : !cute.int_tuple<"?">
      %iter_163 = cute.get_iter(%slice_157) : !cute.coord_tensor<"(?,?{div=8},?)", "(8):(1@1)">
      %tup_164 = cute.deref_arith_tuple_iter(%iter_163) : !cute.arith_tuple_iter<"(?,?{div=8},?)">
      %e0_165, %e1_166, %e2_167 = cute.get_leaves(%tup_164) : !cute.int_tuple<"(?,?{div=8},?)">
      %65 = cute.get_scalars(%e0_165) : !cute.int_tuple<"?">
      %66 = cute.get_scalars(%e1_166) : !cute.int_tuple<"?{div=8}">
      %67 = cute.get_scalars(%e2_167) : !cute.int_tuple<"?">
      %coord_168 = cute.make_coord() : () -> !cute.coord<"0">
      %slice_169 = cute.slice(%slice_157, %coord_168) : !cute.coord_tensor<"(?,?{div=8},?)", "(8):(1@1)">, !cute.coord<"0">
      %iter_170 = cute.get_iter(%slice_169) : !cute.coord_tensor<"(?,?{div=8},?)", "():()">
      %tup_171 = cute.deref_arith_tuple_iter(%iter_170) : !cute.arith_tuple_iter<"(?,?{div=8},?)">
      %e0_172, %e1_173, %e2_174 = cute.get_leaves(%tup_171) : !cute.int_tuple<"(?,?{div=8},?)">
      %68 = cute.get_scalars(%e0_172) : !cute.int_tuple<"?">
      %69 = cute.get_scalars(%e1_173) : !cute.int_tuple<"?{div=8}">
      %70 = cute.get_scalars(%e2_174) : !cute.int_tuple<"?">
      %iter_175 = cute.get_iter(%slice_169) : !cute.coord_tensor<"(?,?{div=8},?)", "():()">
      %tup_176 = cute.deref_arith_tuple_iter(%iter_175) : !cute.arith_tuple_iter<"(?,?{div=8},?)">
      %e0_177, %e1_178, %e2_179 = cute.get_leaves(%tup_176) : !cute.int_tuple<"(?,?{div=8},?)">
      %71 = cute.get_scalars(%e0_177) : !cute.int_tuple<"?">
      %72 = cute.get_scalars(%e1_178) : !cute.int_tuple<"?{div=8}">
      %73 = cute.get_scalars(%e2_179) : !cute.int_tuple<"?">
      %iter_180 = cute.get_iter(%slice_169) : !cute.coord_tensor<"(?,?{div=8},?)", "():()">
      %tup_181 = cute.deref_arith_tuple_iter(%iter_180) : !cute.arith_tuple_iter<"(?,?{div=8},?)">
      %e0_182, %e1_183, %e2_184 = cute.get_leaves(%tup_181) : !cute.int_tuple<"(?,?{div=8},?)">
      %74 = cute.get_scalars(%e0_182) : !cute.int_tuple<"?">
      %75 = cute.get_scalars(%e1_183) : !cute.int_tuple<"?{div=8}">
      %76 = cute.get_scalars(%e2_184) : !cute.int_tuple<"?">
      %coord_185 = cute.make_coord(%e0_182, %e1_183, %e2_184) : (!cute.int_tuple<"?">, !cute.int_tuple<"?{div=8}">, !cute.int_tuple<"?">) -> !cute.coord<"(?,?{div=8},?)">
      %coord_186 = cute.make_coord(%arg5, %arg6, %arg7) : (i32, i32, i32) -> !cute.coord<"(?,?,?)">
      %77 = cute.elem_less(%coord_185, %coord_186) : !cute.coord<"(?,?{div=8},?)">, !cute.coord<"(?,?,?)">
      scf.if %77 {
        %78 = cute.get_shape(%arg3) : (!cute.layout<"(128,8):(8,1)">) -> !cute.shape<"(128,8)">
        %e0_187, %e1_188 = cute.get_leaves(%78) : !cute.shape<"(128,8)">
        %79 = cute.get_shape(%arg3) : (!cute.layout<"(128,8):(8,1)">) -> !cute.shape<"(128,8)">
        %e0_189, %e1_190 = cute.get_leaves(%79) : !cute.shape<"(128,8)">
        %80 = cute.get(%arg3) <{mode = [1]}> : !cute.layout<"(128,8):(8,1)"> -> !cute.layout<"8:1">
        %rmem = cute.memref.alloca(%80) : !memref_rmem_f32
        %iter_191 = cute.get_iter(%rmem) : !memref_rmem_f32
        %iter_192 = cute.get_iter(%rmem) : !memref_rmem_f32
        %81 = cute.get_shape(%arg4) : (!cute.layout<"(128,4):(4,1)">) -> !cute.shape<"(128,4)">
        %e0_193, %e1_194 = cute.get_leaves(%81) : !cute.shape<"(128,4)">
        %82 = cute.get_shape(%arg4) : (!cute.layout<"(128,4):(4,1)">) -> !cute.shape<"(128,4)">
        %e0_195, %e1_196 = cute.get_leaves(%82) : !cute.shape<"(128,4)">
        %83 = cute.get(%arg4) <{mode = [1]}> : !cute.layout<"(128,4):(4,1)"> -> !cute.layout<"4:1">
        %rmem_197 = cute.memref.alloca(%83) : !memref_rmem_i8
        %iter_198 = cute.get_iter(%rmem_197) : !memref_rmem_i8
        %iter_199 = cute.get_iter(%rmem_197) : !memref_rmem_i8
        %atom = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f32>
        cute.copy(%atom, %slice_149, %rmem) : (!cute_nvgpu.atom.universal_copy<f32>, !memref_gmem_f32_3, !memref_rmem_f32)
        %lay_200 = cute.get_layout(%rmem) : !memref_rmem_f32
        %84 = cute.get_shape(%lay_200) : (!cute.layout<"8:1">) -> !cute.shape<"8">
        %e0_201 = cute.get_leaves(%84) : !cute.shape<"8">
        %85 = cute.memref.load_vec %rmem, row_major : !memref_rmem_f32
        %86 = nvgpu.cvt_fptrunc %85 : vector<8xf32> to vector<8xf4E2M1FN>
        %shape = cute.make_shape() : () -> !cute.shape<"8">
        %lay_202 = cute.make_layout(%shape) : !cute.layout<"8:1">
        %87 = cute.recast_layout<8, 4> (%lay_202) : !cute.layout<"8:1"> to !cute.layout<"4:1">
        %88 = cute.get_shape(%87) : (!cute.layout<"4:1">) -> !cute.shape<"4">
        %e0_203 = cute.get_leaves(%88) : !cute.shape<"4">
        %89 = builtin.unrealized_conversion_cast %86 : vector<8xf4E2M1FN> to vector<8xi4>
        %90 = vector.bitcast %89 : vector<8xi4> to vector<4xi8>
        %lay_204 = cute.get_layout(%rmem_197) : !memref_rmem_i8
        %91 = cute.get_shape(%lay_204) : (!cute.layout<"4:1">) -> !cute.shape<"4">
        %e0_205 = cute.get_leaves(%91) : !cute.shape<"4">
        %int_tuple = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
        %sz = cute.size(%int_tuple) : (!cute.int_tuple<"4">) -> !cute.int_tuple<"4">
        %e0_206 = cute.get_leaves(%sz) : !cute.int_tuple<"4">
        %int_tuple_207 = cute.make_int_tuple() : () -> !cute.int_tuple<"4">
        %sz_208 = cute.size(%int_tuple_207) : (!cute.int_tuple<"4">) -> !cute.int_tuple<"4">
        %e0_209 = cute.get_leaves(%sz_208) : !cute.int_tuple<"4">
        cute.memref.store_vec %90, %rmem_197, row_major : !memref_rmem_i8
        %atom_210 = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<i8>
        cute.copy(%atom_210, %rmem_197, %slice_153) : (!cute_nvgpu.atom.universal_copy<i8>, !memref_rmem_i8, !memref_gmem_i8_3)
      }
      return
    }
  }
  func.func @cutlass__convert_Tensorgmemoi64i641_Tensorgmemoi641i64_1_8(%arg0: !memref_gmem_f32_4, %arg1: !memref_gmem_f4E2M1FN) attributes {llvm.emit_c_interface} {
    %iter = cute.get_iter(%arg0) : !memref_gmem_f32_4
    %iter_0 = cute.get_iter(%arg1) : !memref_gmem_f4E2M1FN
    %iter_1 = cute.get_iter(%arg0) : !memref_gmem_f32_4
    %iter_2 = cute.get_iter(%arg1) : !memref_gmem_f4E2M1FN
    %lay = cute.get_layout(%arg0) : !memref_gmem_f32_4
    %0 = cute.get_shape(%lay) : (!cute.layout<"(?,?,?):(?{i64},?{i64},1)">) -> !cute.shape<"(?,?,?)">
    %e0, %e1, %e2 = cute.get_leaves(%0) : !cute.shape<"(?,?,?)">
    %itup = cute.to_int_tuple(%e0) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %1 = cute.get_scalars(%itup) : !cute.int_tuple<"?">
    %itup_3 = cute.to_int_tuple(%e1) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %2 = cute.get_scalars(%itup_3) : !cute.int_tuple<"?">
    %itup_4 = cute.to_int_tuple(%e2) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %3 = cute.get_scalars(%itup_4) : !cute.int_tuple<"?">
    %4 = cute.get_stride(%lay) : (!cute.layout<"(?,?,?):(?{i64},?{i64},1)">) -> !cute.stride<"(?{i64},?{i64},1)">
    %e0_5, %e1_6, %e2_7 = cute.get_leaves(%4) : !cute.stride<"(?{i64},?{i64},1)">
    %itup_8 = cute.to_int_tuple(%e0_5) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
    %5 = cute.get_scalars(%itup_8) : !cute.int_tuple<"?{i64}">
    %itup_9 = cute.to_int_tuple(%e1_6) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
    %6 = cute.get_scalars(%itup_9) : !cute.int_tuple<"?{i64}">
    %lay_10 = cute.get_layout(%arg1) : !memref_gmem_f4E2M1FN
    %7 = cute.get_shape(%lay_10) : (!cute.layout<"(?,?,?):(?{i64},1,?{i64})">) -> !cute.shape<"(?,?,?)">
    %e0_11, %e1_12, %e2_13 = cute.get_leaves(%7) : !cute.shape<"(?,?,?)">
    %itup_14 = cute.to_int_tuple(%e0_11) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %8 = cute.get_scalars(%itup_14) : !cute.int_tuple<"?">
    %itup_15 = cute.to_int_tuple(%e1_12) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %9 = cute.get_scalars(%itup_15) : !cute.int_tuple<"?">
    %itup_16 = cute.to_int_tuple(%e2_13) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %10 = cute.get_scalars(%itup_16) : !cute.int_tuple<"?">
    %11 = cute.get_stride(%lay_10) : (!cute.layout<"(?,?,?):(?{i64},1,?{i64})">) -> !cute.stride<"(?{i64},1,?{i64})">
    %e0_17, %e1_18, %e2_19 = cute.get_leaves(%11) : !cute.stride<"(?{i64},1,?{i64})">
    %itup_20 = cute.to_int_tuple(%e0_17) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
    %12 = cute.get_scalars(%itup_20) : !cute.int_tuple<"?{i64}">
    %itup_21 = cute.to_int_tuple(%e2_19) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
    %13 = cute.get_scalars(%itup_21) : !cute.int_tuple<"?{i64}">
    %shape = cute.make_shape() : () -> !cute.shape<"(128,8)">
    %stride = cute.make_stride() : () -> !cute.stride<"(8,1)">
    %lay_22 = cute.make_layout(%shape, %stride) : !cute.layout<"(128,8):(8,1)">
    %14 = cute.recast_layout<8, 4> (%lay_22) : !cute.layout<"(128,8):(8,1)"> to !cute.layout<"(128,4):(4,1)">
    %iter_23 = cute.recast_iter(%iter_2) : !cute.ptr<f4E2M1FN, gmem, align<16>> to !cute.ptr<i8, gmem, align<16>>
    %15 = cute.recast_layout<8, 4> (%lay_10) : !cute.layout<"(?,?,?):(?{i64},1,?{i64})"> to !cute.layout<"(?,?,?):(?{i64},1,?{i64})">
    %view = cute.make_view(%iter_23, %15) : !memref_gmem_i8_4
    %iter_24 = cute.get_iter(%view) : !memref_gmem_i8_4
    %16 = cute.get_shape(%lay) : (!cute.layout<"(?,?,?):(?{i64},?{i64},1)">) -> !cute.shape<"(?,?,?)">
    %e0_25, %e1_26, %e2_27 = cute.get_leaves(%16) : !cute.shape<"(?,?,?)">
    %itup_28 = cute.to_int_tuple(%e0_25) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %17 = cute.get_scalars(%itup_28) : !cute.int_tuple<"?">
    %itup_29 = cute.to_int_tuple(%e1_26) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %18 = cute.get_scalars(%itup_29) : !cute.int_tuple<"?">
    %itup_30 = cute.to_int_tuple(%e2_27) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %19 = cute.get_scalars(%itup_30) : !cute.int_tuple<"?">
    %shape_31 = cute.make_shape(%itup_28, %itup_29, %itup_30) : (!cute.int_tuple<"?">, !cute.int_tuple<"?">, !cute.int_tuple<"?">) -> !cute.shape<"(?,?,?)">
    %20 = cute.make_identity_tensor(%shape_31) : !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">
    %iter_32 = cute.get_iter(%20) : !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">
    %tup = cute.deref_arith_tuple_iter(%iter_32) : !cute.arith_tuple_iter<"(0,0,0)">
    %e0_33, %e1_34, %e2_35 = cute.get_leaves(%tup) : !cute.int_tuple<"(0,0,0)">
    %21 = cute.get_shape(%lay) : (!cute.layout<"(?,?,?):(?{i64},?{i64},1)">) -> !cute.shape<"(?,?,?)">
    %e0_36, %e1_37, %e2_38 = cute.get_leaves(%21) : !cute.shape<"(?,?,?)">
    %itup_39 = cute.to_int_tuple(%e0_36) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %22 = cute.get_scalars(%itup_39) : !cute.int_tuple<"?">
    %itup_40 = cute.to_int_tuple(%e1_37) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %23 = cute.get_scalars(%itup_40) : !cute.int_tuple<"?">
    %itup_41 = cute.to_int_tuple(%e2_38) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %24 = cute.get_scalars(%itup_41) : !cute.int_tuple<"?">
    %sz = cute.size(%lay_22) : (!cute.layout<"(128,8):(8,1)">) -> !cute.int_tuple<"1024">
    %e0_42 = cute.get_leaves(%sz) : !cute.int_tuple<"1024">
    %lay_43 = cute.get_layout(%view) : !memref_gmem_i8_4
    %25 = cute.get_shape(%lay_43) : (!cute.layout<"(?,?,?):(?{i64},1,?{i64})">) -> !cute.shape<"(?,?,?)">
    %e0_44, %e1_45, %e2_46 = cute.get_leaves(%25) : !cute.shape<"(?,?,?)">
    %itup_47 = cute.to_int_tuple(%e0_44) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %26 = cute.get_scalars(%itup_47) : !cute.int_tuple<"?">
    %itup_48 = cute.to_int_tuple(%e1_45) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %27 = cute.get_scalars(%itup_48) : !cute.int_tuple<"?">
    %itup_49 = cute.to_int_tuple(%e2_46) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %28 = cute.get_scalars(%itup_49) : !cute.int_tuple<"?">
    %sz_50 = cute.size(%14) : (!cute.layout<"(128,4):(4,1)">) -> !cute.int_tuple<"512">
    %e0_51 = cute.get_leaves(%sz_50) : !cute.int_tuple<"512">
    %tile = cute.make_tile() : () -> !cute.tile<"[1:0;1024:1;1:0]">
    %div = cute.zipped_divide(%arg0, %tile) : !memref_gmem_f32_4, !cute.tile<"[1:0;1024:1;1:0]">
    %iter_52 = cute.get_iter(%div) : !memref_gmem_f32
    %iter_53 = cute.get_iter(%div) : !memref_gmem_f32
    %tile_54 = cute.make_tile() : () -> !cute.tile<"[1:0;1024:1;1:0]">
    %div_55 = cute.zipped_divide(%20, %tile_54) : !cute.coord_tensor<"(0,0,0)", "(?,?,?):(1@0,1@1,1@2)">, !cute.tile<"[1:0;1024:1;1:0]">
    %iter_56 = cute.get_iter(%div_55) : !cute.coord_tensor<"(0,0,0)", "((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">
    %tup_57 = cute.deref_arith_tuple_iter(%iter_56) : !cute.arith_tuple_iter<"(0,0,0)">
    %e0_58, %e1_59, %e2_60 = cute.get_leaves(%tup_57) : !cute.int_tuple<"(0,0,0)">
    %iter_61 = cute.get_iter(%div_55) : !cute.coord_tensor<"(0,0,0)", "((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">
    %tup_62 = cute.deref_arith_tuple_iter(%iter_61) : !cute.arith_tuple_iter<"(0,0,0)">
    %e0_63, %e1_64, %e2_65 = cute.get_leaves(%tup_62) : !cute.int_tuple<"(0,0,0)">
    %tile_66 = cute.make_tile() : () -> !cute.tile<"[1:0;512:1;1:0]">
    %div_67 = cute.zipped_divide(%view, %tile_66) : !memref_gmem_i8_4, !cute.tile<"[1:0;512:1;1:0]">
    %iter_68 = cute.get_iter(%div_67) : !memref_gmem_i8
    %iter_69 = cute.get_iter(%div_67) : !memref_gmem_i8
    %sz_70 = cute.size(%div) <{mode = [1]}> : (!memref_gmem_f32) -> !cute.int_tuple<"?">
    %e0_71 = cute.get_leaves(%sz_70) : !cute.int_tuple<"?">
    %29 = cute.get_scalars(%e0_71) : !cute.int_tuple<"?">
    %sz_72 = cute.size(%lay_22) <{mode = [0]}> : (!cute.layout<"(128,8):(8,1)">) -> !cute.int_tuple<"128">
    %e0_73 = cute.get_leaves(%sz_72) : !cute.int_tuple<"128">
    %lay_74 = cute.get_layout(%div) : !memref_gmem_f32
    %30 = cute.get_shape(%lay_74) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,?{i64},0),(?{i64},?{i64 div=1024},1))">) -> !cute.shape<"((1,1024,1),(?,?,?))">
    %e0_75, %e1_76, %e2_77, %e3, %e4, %e5 = cute.get_leaves(%30) : !cute.shape<"((1,1024,1),(?,?,?))">
    %itup_78 = cute.to_int_tuple(%e3) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %31 = cute.get_scalars(%itup_78) : !cute.int_tuple<"?">
    %itup_79 = cute.to_int_tuple(%e4) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %32 = cute.get_scalars(%itup_79) : !cute.int_tuple<"?">
    %itup_80 = cute.to_int_tuple(%e5) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %33 = cute.get_scalars(%itup_80) : !cute.int_tuple<"?">
    %34 = cute.get_stride(%lay_74) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,?{i64},0),(?{i64},?{i64 div=1024},1))">) -> !cute.stride<"((0,?{i64},0),(?{i64},?{i64 div=1024},1))">
    %e0_81, %e1_82, %e2_83, %e3_84, %e4_85, %e5_86 = cute.get_leaves(%34) : !cute.stride<"((0,?{i64},0),(?{i64},?{i64 div=1024},1))">
    %itup_87 = cute.to_int_tuple(%e1_82) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
    %35 = cute.get_scalars(%itup_87) : !cute.int_tuple<"?{i64}">
    %itup_88 = cute.to_int_tuple(%e3_84) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
    %36 = cute.get_scalars(%itup_88) : !cute.int_tuple<"?{i64}">
    %itup_89 = cute.to_int_tuple(%e4_85) : !cute.stride<"?{i64 div=1024}"> to !cute.int_tuple<"?{i64 div=1024}">
    %37 = cute.get_scalars(%itup_89) : !cute.int_tuple<"?{i64 div=1024}">
    %lay_90 = cute.get_layout(%div_67) : !memref_gmem_i8
    %38 = cute.get_shape(%lay_90) : (!cute.layout<"((1,512,1),(?,?,?)):((0,1,0),(?{i64},512,?{i64}))">) -> !cute.shape<"((1,512,1),(?,?,?))">
    %e0_91, %e1_92, %e2_93, %e3_94, %e4_95, %e5_96 = cute.get_leaves(%38) : !cute.shape<"((1,512,1),(?,?,?))">
    %itup_97 = cute.to_int_tuple(%e3_94) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %39 = cute.get_scalars(%itup_97) : !cute.int_tuple<"?">
    %itup_98 = cute.to_int_tuple(%e4_95) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %40 = cute.get_scalars(%itup_98) : !cute.int_tuple<"?">
    %itup_99 = cute.to_int_tuple(%e5_96) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %41 = cute.get_scalars(%itup_99) : !cute.int_tuple<"?">
    %42 = cute.get_stride(%lay_90) : (!cute.layout<"((1,512,1),(?,?,?)):((0,1,0),(?{i64},512,?{i64}))">) -> !cute.stride<"((0,1,0),(?{i64},512,?{i64}))">
    %e0_100, %e1_101, %e2_102, %e3_103, %e4_104, %e5_105 = cute.get_leaves(%42) : !cute.stride<"((0,1,0),(?{i64},512,?{i64}))">
    %itup_106 = cute.to_int_tuple(%e3_103) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
    %43 = cute.get_scalars(%itup_106) : !cute.int_tuple<"?{i64}">
    %itup_107 = cute.to_int_tuple(%e5_105) : !cute.stride<"?{i64}"> to !cute.int_tuple<"?{i64}">
    %44 = cute.get_scalars(%itup_107) : !cute.int_tuple<"?{i64}">
    %lay_108 = cute.get_layout(%div_55) : !cute.coord_tensor<"(0,0,0)", "((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">
    %45 = cute.get_shape(%lay_108) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">) -> !cute.shape<"((1,1024,1),(?,?,?))">
    %e0_109, %e1_110, %e2_111, %e3_112, %e4_113, %e5_114 = cute.get_leaves(%45) : !cute.shape<"((1,1024,1),(?,?,?))">
    %itup_115 = cute.to_int_tuple(%e3_112) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %46 = cute.get_scalars(%itup_115) : !cute.int_tuple<"?">
    %itup_116 = cute.to_int_tuple(%e4_113) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %47 = cute.get_scalars(%itup_116) : !cute.int_tuple<"?">
    %itup_117 = cute.to_int_tuple(%e5_114) : !cute.shape<"?"> to !cute.int_tuple<"?">
    %48 = cute.get_scalars(%itup_117) : !cute.int_tuple<"?">
    %49 = cute.get_stride(%lay_108) : (!cute.layout<"((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">) -> !cute.stride<"((0,1@1,0),(1@0,1024@1,1@2))">
    %e0_118, %e1_119, %e2_120, %e3_121, %e4_122, %e5_123 = cute.get_leaves(%49) : !cute.stride<"((0,1@1,0),(1@0,1024@1,1@2))">
    %50 = cute.get_shape(%lay_22) : (!cute.layout<"(128,8):(8,1)">) -> !cute.shape<"(128,8)">
    %e0_124, %e1_125 = cute.get_leaves(%50) : !cute.shape<"(128,8)">
    %51 = cute.get_stride(%lay_22) : (!cute.layout<"(128,8):(8,1)">) -> !cute.stride<"(8,1)">
    %e0_126, %e1_127 = cute.get_leaves(%51) : !cute.stride<"(8,1)">
    %52 = cute.get_shape(%14) : (!cute.layout<"(128,4):(4,1)">) -> !cute.shape<"(128,4)">
    %e0_128, %e1_129 = cute.get_leaves(%52) : !cute.shape<"(128,4)">
    %53 = cute.get_stride(%14) : (!cute.layout<"(128,4):(4,1)">) -> !cute.stride<"(4,1)">
    %e0_130, %e1_131 = cute.get_leaves(%53) : !cute.stride<"(4,1)">
    %c0_i32 = arith.constant 0 : i32
    %54 = arith.index_cast %29 : i32 to index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    gpu.launch_func  @kernels::@kernel_cutlass__convert_kernel_tensorptrf32gmemo11024100div10241_tensorptri8gmemalign16o15121010512_tensor000o1102410110101024112____Float32_Float4E2M1FN_0 blocks in (%54, %c1, %c1) threads in (%c128, %c1, %c1)  dynamic_shared_memory_size %c0_i32 args(%div : !memref_gmem_f32, %div_67 : !memref_gmem_i8, %div_55 : !cute.coord_tensor<"(0,0,0)", "((1,1024,1),(?,?,?)):((0,1@1,0),(1@0,1024@1,1@2))">, %lay_22 : !cute.layout<"(128,8):(8,1)">, %14 : !cute.layout<"(128,4):(4,1)">, %17 : i32, %18 : i32, %19 : i32) {use_pdl = false}
    return
  }
}
