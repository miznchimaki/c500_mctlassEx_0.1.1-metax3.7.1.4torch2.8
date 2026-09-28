import torch
from typing import Any, Dict, List
from .mctlassExIn import mctlassExInternalHandleWrapper

class HcPrenormGEMM:

    def __init__(self):
        self._mctlass_op = mctlassExInternalHandleWrapper()

    def _update_config(self, **kwargs):
        pass

    def dtype_to_string(self, matrix: torch.Tensor):
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("Not torch.Tensor")
        return str(matrix.dtype).split('.')[-1]

    def __call__(self, gemm_shapes: List,
                 A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                 sqr_sum: torch.Tensor, num_splits,
                **kwargs):
        self.M = gemm_shapes[0]
        self.N = gemm_shapes[1]
        self.K = gemm_shapes[2]
        self.stream = torch.cuda.current_stream().cuda_stream
        self.A = A
        self.B = B
        self.C = C
        self.sqr_sum = sqr_sum
        self.num_splits = num_splits
        self._update_config(**kwargs)

        self.layout_a = "T" if A.stride()[-1] == 1 else "N"
        self.layout_b = "T" if B.stride()[-1] == 1 else "N"
        self.layout_c = "T" if C.stride()[-1] == 1 else "N"
        self.layout = self.layout_a + self.layout_b + self.layout_c

        self.type_a = self.dtype_to_string(A)
        self.type_b = self.dtype_to_string(B)
        self.type_c = self.dtype_to_string(C)

        self._mctlass_op.mctlass_hc_prenorm_gemm(A, B, C, self.M, self.N, self.K, sqr_sum, num_splits, self.stream)


