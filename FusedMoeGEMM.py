import torch
from typing import Any, Dict
from .mctlassExIn import mctlassExInternalHandleWrapper

class FusedMoeGEMM:

    def __init__(self):
        self._mctlass_op = mctlassExInternalHandleWrapper()

    def _update_config(self, **kwargs):
        self.config: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.config[k] = v
        if "mul_weight" not in self.config:
            self.config["mul_weight"] = False
        if "is_blockwise" not in self.config:
            self.config["is_blockwise"] = False
        if "zp_b" not in self.config:
            self.config["zp_b"] = None

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

    def get_kernel_m(self, A, B, C, num_experts, batch_size, N, K, topk, **kwargs):
        self._update_config(**kwargs)
        num_valid_tokens = batch_size * topk
        group_size = None
        if self.config["is_blockwise"]:
            if "group_size" in self.config:
                group_size = self.config['group_size']
            else:
                raise "group_size must be configured in blockwise mode."
                
        return  self._mctlass_op.mctlass_fuse_moe_get_kernel_m_basic(A, B, C, num_experts, batch_size, N, K, num_valid_tokens, topk, group_size, self.config["mul_weight"])

    def __call__(self, batch_size: int, N: int, K: int, num_experts: int,
                 EM: int, topk: int,
                 A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                 scale_a: torch.Tensor, scale_b: torch.Tensor, bias:torch.Tensor,
                 topk_weights: torch.Tensor, token_ids:torch.Tensor, expert_ids: torch.Tensor,
                 num_tokens_post_padded: torch.Tensor, mul_routed_weight: bool, **kwargs):
        if not self.check_shape():
            raise  ValueError("Error input shape")

        self.batch_size = batch_size
        self.N = N
        self.K = K
        self.num_experts = num_experts
        self.EM = EM
        self.topk = topk
        self.num_valid_tokens = batch_size * topk

        self.A = A
        self.B = B
        self.C = C
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.bias = bias

        self.topk_weights = topk_weights
        self.token_ids = token_ids
        self.expert_ids = expert_ids
        self.num_tokens_post_padded = num_tokens_post_padded
        self.mul_routed_weight = mul_routed_weight

        self.stream = torch.cuda.current_stream().cuda_stream
        self._update_config(**kwargs)

        self.layout_a = "T" if A.stride()[-1] == 1 else "N"
        self.layout_b = "T" if B.stride()[-1] == 1 else "N"
        self.layout_c = "T" if C.stride()[-1] == 1 else "N"
        self.layout = self.layout_a + self.layout_b + self.layout_c

        self.type_a = self.dtype_to_string(A)
        self.type_b = self.dtype_to_string(B)
        self.type_c = self.dtype_to_string(C)

        self.filter_expert = self.config.get("filter_expert", False)

        group_size = None
        if self.config["is_blockwise"]:
            if "group_size" in self.config:
                group_size = self.config['group_size']
            else:
                raise "group_size must be configured in blockwise mode."

        # run pybinde tlass gemm
        if ((self.type_b == "int8" or self.type_b == "quint4x2") and self.type_a == "int8" and self.type_c == "bfloat16") or\
           (self.type_a == "bfloat16" and self.type_b == "bfloat16" and self.type_c == "bfloat16") or\
           (self.type_b == "quint4x2" and self.type_a == "bfloat16" and self.type_c == "bfloat16"):
            self._mctlass_op.mctlass_fuse_moe_gemm_basic(A, B, C, scale_a, scale_b, bias,
                                                  topk_weights, token_ids, expert_ids, num_tokens_post_padded,
                                                  num_experts, batch_size, N, K, self.num_valid_tokens,
                                                  EM, topk, mul_routed_weight, group_size, self.filter_expert,
                                                  self.config["zp_b"])