import torch
from typing import Any, Dict, List
from .mctlassExIn import mctlassExInternalHandleWrapper

class ContinuousGroupedGEMM:

    def __init__(self):
        self._mctlass_op = mctlassExInternalHandleWrapper()

    def _update_config(self, **kwargs):
        for k, v in kwargs.items():
            self.config[k] = v
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

    def get_blocksize_m(self, A, B, C, scale_a, scale_b, **kwargs):
        self.config: Dict[str, Any] = {}
        self._update_config(**kwargs)
        self.is_scale_a_k_major = (self.config["scale_a_layout"] == "k-major")
        self.is_scale_b_k_major = (self.config["scale_b_layout"] == "k-major")

        if self.config["is_blockwise"]:

            arch = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
            if arch == 80 or arch == 86:  # c500 or c600
                if self.is_scale_a_k_major == False:
                    scale_a = scale_a.T.contiguous()
                    self.is_scale_a_k_major = True
                if self.is_scale_b_k_major == False:
                    scale_b = scale_b.transpose(1, 2).contiguous()
                    self.is_scale_b_k_major = True

            return self._mctlass_op.mctlass_get_blockwise_blocksize_m(A,
                                                                      B,
                                                                      C,
                                                                      scale_a,
                                                                      scale_b,
                                                                      self.config["is_scale_a_1d"],
                                                                      self.config["is_scale_b_1d"],
                                                                      self.is_scale_a_k_major,
                                                                      self.is_scale_b_k_major)
        else:
            return self._mctlass_op.mctlass_get_blocksize_m(A, B, C, scale_a, scale_b)

    def compute_m_num_tiles_indptr(self, A, B, C, seg_indptr, m_num_tiles_indptr, blocksizeM, **kwargs):
        self.stream = torch.cuda.current_stream().cuda_stream
        # self.config: Dict[str, Any] = {}
        self._update_config(**kwargs)
        if "m_indices" not in self.config:
            self._mctlass_op.mctlass_compute_m_num_tiles_indptr(A, B, C,
                                                                seg_indptr, m_num_tiles_indptr,
                                                                blocksizeM, self.stream)
        else:
            self._mctlass_op.mctlass_fp8_blockwise_contiguous_group_gemm_helper(A, B, C, seg_indptr, m_num_tiles_indptr,
                                                                                self.config["m_indices"], blocksizeM, self.config["M"], self.config["n_group"])

    def __call__(self, gemm_shapes: List,
                 A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                 scale_a: torch.Tensor, scale_b: torch.Tensor, bias:torch.Tensor,
                 seg_indptr: torch.Tensor, weight_indices: torch.Tensor, m_num_tiles_indptr: torch.Tensor,
                 **kwargs):
        self.M = gemm_shapes[0]
        self.N = gemm_shapes[1]
        self.K = gemm_shapes[2]
        self.A = A
        self.B = B
        self.C = C
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.bias = bias
        self.seg_indptr = seg_indptr
        self.m_num_tiles_indptr = m_num_tiles_indptr
        self.weight_indices = weight_indices
        self._update_config(**kwargs)

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

        if self.config["is_blockwise"]:
            if self.type_c == "bfloat16":
                arch = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
                if arch == 80 or arch == 86:  # c500 or c600
                    if self.is_scale_a_k_major == False:
                        self.scale_a = self.scale_a.T.contiguous()
                        self.is_scale_a_k_major = True
                    if self.is_scale_b_k_major == False:
                        self.scale_b = self.scale_b.transpose(1, 2).contiguous()
                        self.is_scale_b_k_major = True
                self._mctlass_op.mctlass_fp8_blockwise_contiguous_group_gemm(
                    A, B, C, self.scale_a, self.scale_b,
                    seg_indptr, m_num_tiles_indptr, self.config["m_indices"],
                    self.M, self.N, self.K, self.config["n_group"],
                    self.config["is_scale_a_1d"], self.config["is_scale_b_1d"],
                    self.is_scale_a_k_major, self.is_scale_b_k_major)
                return

        elif (self.type_b == "int8" or self.type_b == "quint4x2") and self.type_a == "int8" and self.type_c == "bfloat16":
            self._mctlass_op.mctlass_contiguous_group_gemm(A, B, C, scale_a, scale_b,
                                      seg_indptr, weight_indices, m_num_tiles_indptr, self.stream)
            return
