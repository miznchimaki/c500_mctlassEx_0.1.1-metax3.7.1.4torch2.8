import torch
from typing import Any, Dict, List
from .mctlassExIn import mctlassExInternalHandleWrapper

class MaskedGroupedGEMM:

    def __init__(self):
        self._mctlass_op = mctlassExInternalHandleWrapper()

    def _update_config(self, **kwargs):
        self.config: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.config[k] = v
        if "A_packed" not in self.config:
            self.config["A_packed"] = False
        if "B_packed" not in self.config:
            self.config["B_packed"] = False
        if "use_fp8" not in self.config:
            self.config["use_fp8"] = False
        if "is_scale_a_1d" not in self.config:
            self.config["is_scale_a_1d"] = False
        if "is_scale_b_1d" not in self.config:
            self.config["is_scale_b_1d"] = False
        if "is_blockwise" not in self.config:
            self.config["is_blockwise"] = False
        if "n_group" not in self.config:
            self.config["n_group"] = 1
        if "avg_mask" not in self.config:
            self.config["avg_mask"] = 0
        if "single_batch_overlap" not in self.config:
            self.config["single_batch_overlap"] = False
        if "num_blocks" not in self.config:
            self.config["num_blocks"] = 0
        if "scale_a_layout" not in self.config:
            self.config["scale_a_layout"] = "k-major"
        if "scale_b_layout" not in self.config:
            self.config["scale_b_layout"] = "k-major"

    def check_shape(self):
        # TODO
        return True

    def check_device(self, matrix: torch.Tensor):
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("Not torch.Tensor")
        return matrix.is_cuda

    def get_gemm_type(self):
        return self.layout

    def dtype_to_string(self, matrix: torch.Tensor):
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("Not torch.Tensor")
        return str(matrix.dtype).split('.')[-1]

    def get_kernel_mn(self, gemm_shapes: List, A, B, C, **kwargs):
        self.M = gemm_shapes[0]
        self.N = gemm_shapes[1]
        self.K = gemm_shapes[2]
        self._update_config(**kwargs)
        return self._mctlass_op.mctlass_masked_group_gemm_get_kernel_mn(A, B, C, self.M, self.N, self.K, self.config["avg_mask"], self.config["single_batch_overlap"])

    def __call__(self,
                 gemm_shapes: List,
                 A: torch.Tensor,
                 B: torch.Tensor,
                 C: torch.Tensor,
                 mask:torch.Tensor,
                 **kwargs):
        self.M = gemm_shapes[0]
        self.N = gemm_shapes[1]
        self.K = gemm_shapes[2]
        self.A = A
        self.B = B
        self.C = C
        self.mask = mask
        self._update_config(**kwargs)

        self.stream = torch.cuda.current_stream().cuda_stream

        self.layout_a = "T" if A.stride()[-1] == 1 else "N"
        self.layout_b = "T" if B.stride()[-1] == 1 else "N"
        self.layout_c = "T" if C.stride()[-1] == 1 else "N"
        self.layout = self.layout_a + self.layout_b + self.layout_c

        self.type_a = self.dtype_to_string(A)
        self.type_b = self.dtype_to_string(B)
        self.type_c = self.dtype_to_string(C)

        self.is_scale_a_k_major = (self.config["scale_a_layout"] == "k-major")
        self.is_scale_b_k_major = (self.config["scale_b_layout"] == "k-major")

        if not self.check_shape():
            raise  ValueError("Error input shape")

        # run pybinde tlass gemm
        if self.config["is_blockwise"]:
            if self.type_c == "bfloat16":
                self.scale_a = self.config["scale_a"]
                self.scale_b = self.config["scale_b"]
                arch = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
                if arch == 80 or arch == 86:  # c500 or c600
                    if self.is_scale_a_k_major == False:
                        self.scale_a = self.config["scale_a"].transpose(-1, -2).contiguous()
                        self.is_scale_a_k_major = True
                    if self.is_scale_b_k_major == False:
                        self.scale_b = self.config["scale_b"].transpose(-1, -2).contiguous()
                        self.is_scale_b_k_major = True
                if self.type_b == "fp8" and self.type_a == "fp8":
                    self._mctlass_op.mctlass_fp8_blockwise_masked_group_gemm(
                        self.A, self.B, self.C, self.scale_a, self.scale_b,
                        self.mask, self.M, self.N, self.K, self.config["n_group"],
                        self.config["is_scale_a_1d"], self.config["is_scale_b_1d"],
                        self.is_scale_a_k_major, self.is_scale_b_k_major)
                else:
                    self._mctlass_op.mctlass_blockwise_masked_group_gemm(
                        self.A, self.B, self.C, self.scale_a, self.scale_b,
                        self.mask, self.M, self.N, self.K, self.config["n_group"],
                        self.config["is_scale_a_1d"], self.config["is_scale_b_1d"],
                        self.is_scale_a_k_major, self.is_scale_b_k_major)
                return
        elif (self.type_b == "int8" or self.type_b == "quint4x2") and self.type_a == "int8" and self.type_c == "bfloat16":
            if self.config["A_packed"] == True:
                self.scale_a = None
            else:
                self.scale_a = self.config["scale_a"]
            if self.config["B_packed"] == True:
                self.scale_b = None
            else:
                self.scale_b = self.config["scale_b"]
            if self.config["single_batch_overlap"]:
                self.signal = self.config["signal"]
            else:
                self.signal = None
            self._mctlass_op.mctlass_masked_group_gemm(self.A,
                                                       self.B,
                                                       self.C,
                                                       self.scale_a,
                                                       self.scale_b,
                                                       self.mask,
                                                       self.stream,
                                                       self.config["avg_mask"],
                                                       self.signal,
                                                       self.config["num_blocks"])
            return
