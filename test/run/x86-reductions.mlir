// REQUIRES: x86
// RUN: lh-run --stage=%TEST/../lighthouse/pipeline/descriptors/x86_64/default.yaml %s --entry-point=softmax --input-shape=256x256xf32xid,256x256xf32x0 --print-tensor=2 | FileCheck %s --check-prefix=SOFTMAX
// RUN: lh-run --stage=%TEST/../lighthouse/pipeline/descriptors/x86_64/default.yaml %s --entry-point=rms_norm --input-shape=256x256xf32xid,256x256xf32x0 --print-tensor=2 | FileCheck %s --check-prefix=RMS

// Row reductions (max, sum) through the x86 default pipeline.
// Identity input: e / (e + 255) on the diagonal and 1 / (e + 255) elsewhere.
// SOFTMAX: Output: {{\[\[}}0.010547{{[0-9]+}} 0.0038802{{[0-9]+}} 0.0038802{{[0-9]+}} ...
// SOFTMAX-NEXT: [0.0038802{{[0-9]+}} 0.010547{{[0-9]+}} 0.0038802{{[0-9]+}} ...
// SOFTMAX: 0.0038802{{[0-9]+}} 0.010547{{[0-9]+}}]]

// RMS norm with an integer power: x / sqrt(mean(x^2)), i.e. 16 on the diagonal.
// RMS: Output: {{\[\[}}16.  0.  0. ...
// RMS-NEXT: [ 0. 16.  0. ...
// RMS: 0. 16.]]

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @softmax(%x: tensor<256x256xf32>, %out: memref<256x256xf32>) {
    %ninf = arith.constant 0xFF800000 : f32
    %zero = arith.constant 0.0 : f32
    %e1 = tensor.empty() : tensor<256xf32>
    %m0 = linalg.fill ins(%ninf : f32) outs(%e1 : tensor<256xf32>) -> tensor<256xf32>
    %max = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%x : tensor<256x256xf32>) outs(%m0 : tensor<256xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %r = arith.maximumf %in, %acc : f32
      linalg.yield %r : f32
    } -> tensor<256xf32>
    %e2 = tensor.empty() : tensor<256x256xf32>
    %exp = linalg.generic {indexing_maps = [#id, #row, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%x, %max : tensor<256x256xf32>, tensor<256xf32>)
        outs(%e2 : tensor<256x256xf32>) {
    ^bb0(%in: f32, %m: f32, %o: f32):
      %s = arith.subf %in, %m : f32
      %r = math.exp %s : f32
      linalg.yield %r : f32
    } -> tensor<256x256xf32>
    %s0 = linalg.fill ins(%zero : f32) outs(%e1 : tensor<256xf32>) -> tensor<256xf32>
    %sum = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%exp : tensor<256x256xf32>) outs(%s0 : tensor<256xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %r = arith.addf %in, %acc : f32
      linalg.yield %r : f32
    } -> tensor<256xf32>
    %div = linalg.generic {indexing_maps = [#id, #row, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%exp, %sum : tensor<256x256xf32>, tensor<256xf32>)
        outs(%e2 : tensor<256x256xf32>) {
    ^bb0(%in: f32, %s: f32, %o: f32):
      %r = arith.divf %in, %s : f32
      linalg.yield %r : f32
    } -> tensor<256x256xf32>
    bufferization.materialize_in_destination %div in restrict writable %out
        : (tensor<256x256xf32>, memref<256x256xf32>) -> ()
    return
  }

  func.func @rms_norm(%x: tensor<256x256xf32>, %out: memref<256x256xf32>) {
    %zero = arith.constant 0.0 : f32
    %n = arith.constant 256.0 : f32
    %two = arith.constant 2 : i32
    %e1 = tensor.empty() : tensor<256xf32>
    %s0 = linalg.fill ins(%zero : f32) outs(%e1 : tensor<256xf32>) -> tensor<256xf32>
    %sq = linalg.generic {indexing_maps = [#id, #row],
        iterator_types = ["parallel", "reduction"]}
        ins(%x : tensor<256x256xf32>) outs(%s0 : tensor<256xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %p = math.fpowi %in, %two : f32, i32
      %r = arith.addf %p, %acc : f32
      linalg.yield %r : f32
    } -> tensor<256xf32>
    %e2 = tensor.empty() : tensor<256x256xf32>
    %norm = linalg.generic {indexing_maps = [#id, #row, #id],
        iterator_types = ["parallel", "parallel"]}
        ins(%x, %sq : tensor<256x256xf32>, tensor<256xf32>)
        outs(%e2 : tensor<256x256xf32>) {
    ^bb0(%in: f32, %s: f32, %o: f32):
      %mean = arith.divf %s, %n : f32
      %rs = math.rsqrt %mean : f32
      %r = arith.mulf %in, %rs : f32
      linalg.yield %r : f32
    } -> tensor<256x256xf32>
    bufferization.materialize_in_destination %norm in restrict writable %out
        : (tensor<256x256xf32>, memref<256x256xf32>) -> ()
    return
  }
}
