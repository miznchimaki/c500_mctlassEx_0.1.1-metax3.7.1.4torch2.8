import torch
from typing import Any, Dict, List, Tuple
from .mctlassExIn import mctlassExInternalHandleWrapper

class ScaledGEMM:
    """
    gemm = ScaledGEMM()

    1.w8a8_scaled_mm_azp:
        call: gemm([M, N, K], A, B, C, scale_a, scale_b, bias, azp_adj=azp_adj, azp=azp)
        Perform a w8a8 fused GEMM with int8 input, int8 weight and bf16 output.
            1. c = a_scales * b_scales * (a * b)
            2. c = a_scales * b_scales * (a * b) + bias
            3. c = a_scales * b_scales * (a * b - azp_adj) + bias
            4. c = a_scales * b_scales * (a * b - azp_adj * azp) + bias
        Azp and azpPertoken are not supported in bmm.

        Arguements:
            a:
                1.tensor(shape(M, K), stride(K, 1), int8, device)
                2.tensor(shape(group, M, K), stride(M*K, K, 1), int8, device)
            b:
                1.tensor(shape(K, N), stride(1, K), int8, device)
                2.tensor(shape(N, K), stride(K, 1), int8, device)
                3.tensor(shape(group, K, N), stride(N*K, 1, K), int8, device)
                4.tensor(shape(group, N, K), stride(N*K, K, 1), int8, device)
            c:
                1.tensor(shape(M, N), stride(N, 1), bfloat16, device)
                2.tensor(shape(group, M, N), stride(M*N, N, 1), bfloat16, device)
            a_scales:
                1.tensor(shape(M, 1), float32, device) -- per_token(column-vector)
                2.tensor(shape(group, M, 1), float32, device) -- per_token(column-vector)
            b_scales: the shape is according to b
                1.tensor(shape(1, N), float32, device) -- per_channel(row-vector)
                2.tensor(shape(N, 1), float32, device) -- per_channel(row-vector)
                3.tensor(shape(group, 1, N), float32, device) -- per_channel(row-vector)
                4.tensor(shape(group, N, 1), float32, device) -- per_channel(row-vector)
            bias: the shape is according to b
                1.tensor(shape(1, N), bfloat16, device) -- per_channel(row-vector)
                2.tensor(shape(N, 1), bfloat16, device) -- per_channel(row-vector)
                3.tensor(shape(group, 1, N), bfloat16, device) -- per_channel(row-vector)
                4.tensor(shape(group, N, 1), bfloat16, device) -- per_channel(row-vector)
                5.None -- if only scaled
            azp_adj: the shape is according to b
                1.tensor(shape(1, N), int32, device) -- per_channel(row-vector)
                2.tensor(shape(N, 1), int32, device) -- per_channel(row-vector)
                3.None -- if not azp or azpPerToken
            azp:
                1.tensor(shape(M, 1), int32, device) -- per_token(column-vector)
                2.None -- if not azpPerToken
    2.w8a16_scaled_mm:
        call: gemm([M, N, K], A, B, C, None, scale_b, bias/None, group_size = group_size)
        Perform a w8a16 fused GEMM with bf16 input, int8 weight and bf16 output.
            1. c = a * (b_scales * b)
            2. c = a * (b_scales * b) + bias

        Arguements:
            a:
                1.tensor((M, K), bfloat16, device)
                2.tensor((batchsize, M, K), bfloat16, device)
            b:
                1.tensor((N, K), int8, device)
                2.tensor((batchsize, N, K), int8, device)
            c:
                1.tensor((M, N), bfloat16, device)
                2.tensor((batchsize, M, N), bfloat16, device)
            b_scales:
                1.tensor(N, bfloat16, device) -- per_channel(row-vector)
                2.tensor((N, K / group_size), bfloat16, device) -- group_wise
            bias:
                1.tensor(N, bfloat16, device) -- per_channel(row-vector)
                2.None -- if only scaled
            group_size:
                1.128 -- group_wise
                2.None -- per_channel
    3.w4a16_scaled_mm:
        call: gemm([M, N, K], A, B, C, None, scale_b, bias/None, group_size = group_size)
        Perform a w4a16 fused GEMM with bf16 input, int8 weight and bf16 output.
            1. c = a * (b_scales * b)
            2. c = a * (b_scales * b) + bias

        Arguements:
            a:
                1.tensor((M, K), bfloat16, device)
                2.tensor((batchsize, M, K), bfloat16, device)
            b:
                1.tensor((N, K / 2), quint4x2, device)
                2.tensor((batchsize, N, K / 2), quint4x2, device)
            c:
                1.tensor((M, N), bfloat16, device)
                2.tensor((batchsize, M, N), bfloat16, device)
            b_scales:
                1.tensor(N, bfloat16, device) -- per_channel(row-vector)
                2.tensor((N, K / group_size), bfloat16, device) -- group_wise
            bias:
                1.tensor(N, bfloat16, device) -- per_channel(row-vector)
                2.None -- if only scaled
            group_size:
                1.128 -- group_wise
                2.None -- per_channel
    4.fp8_blockwise_gemm:
        call: gemm([M, N, K], A, B, C, scale_a, scale_b, None, is_blockwise = True, use_fp8 = True, is_scale_a_1d = True, is_scale_b_1d = False)
        Perform a fp8 blockwise GEMM with fp8 input, fp8 weight and bf16 output.
        Now support 1d2d and 1d1d.

        Arguements:
            a:
                size=m*k, k-major
            b:
                size=n*k, k-major
            c:
                size=m*n, n-major
            a_scales:
                1.1D--shape(m, k/128)
            b_scales:
                1.1D--shape(n, k/128)
                2.2D--shape(n/128, k/128)
            m:
                the row_num of a and c
            n:
                the col_num of a and c
            k:
                the col_num of a, and row_num of b
            batch_size:
                the group size of a, b, c, a_scales, and b_scales.
                Only support 1 now.
            is_scale_a_1d:
                Only support 1D now.
            is_scale_b_1d:
                True:  1D1D
                False: 1D2D
            scale_a_layout:
                1. k-major
                2. m-major 
            scale_b_layout:
                1. k-major
                2. n-major
    """

    def __init__(self):
        self._mctlass_op = mctlassExInternalHandleWrapper()

    def _update_config(self, **kwargs):
        for k, v in kwargs.items():
            self.config[k] = v
        if "azp" not in self.config:
            self.config["azp"] = None
        if "azp_adj" not in self.config:
            self.config["azp_adj"] = None
        if "group_size" not in self.config:
            self.config["group_size"] = None
        if "use_fp8" not in self.config:
            self.config["use_fp8"] = False
        if "is_scale_a_1d" not in self.config:
            self.config["is_scale_a_1d"] = False
        if "is_scale_b_1d" not in self.config:
            self.config["is_scale_b_1d"] = False
        if "is_blockwise" not in self.config:
            self.config["is_blockwise"] = False
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

    def __call__(self, gemm_shapes: List,
                 A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                 scale_a: torch.Tensor, scale_b: torch.Tensor, bias:torch.Tensor,
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
        self.stream = torch.cuda.current_stream().cuda_stream
        self.config: Dict[str, Any] = {}
        self._update_config(**kwargs)

        self.layout_a = "T" if A.stride()[-1] == 1 else "N"
        self.layout_b = "T" if B.stride()[-1] == 1 else "N"
        self.layout_c = "T" if C.stride()[-1] == 1 else "N"
        self.layout = self.layout_a + self.layout_b + self.layout_c
        self.batchsize = 1 if B.dim() == 2 else B.shape[0]

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
                        self.scale_a = self.scale_a.transpose(-1, -2).contiguous()
                        self.is_scale_a_k_major = True
                    if self.is_scale_b_k_major == False:
                        self.scale_b = self.scale_b.transpose(-1, -2).contiguous()
                        self.is_scale_b_k_major = True
                self._mctlass_op.mctlass_fp8_blockwise_gemm(self.A, self.B, self.C, self.scale_a, self.scale_b,
                                                            self.M, self.N, self.K, self.batchsize, self.config["is_scale_a_1d"],
                                                            self.config["is_scale_b_1d"], self.is_scale_a_k_major,
                                                            self.is_scale_b_k_major)
                return
        elif (self.type_b == "int8" and self.type_a == "int8" and self.type_c == "bfloat16") or\
            (self.type_a == "float8_e4m3fn" and self.type_b == "float8_e4m3fn" and self.type_c == "bfloat16"):
            self._mctlass_op.mctlass_w8a8_scaled_mm_azp(self.A, self.B, self.C, self.scale_a, self.scale_b, self.bias,
                                                       self.config["azp_adj"], self.config["azp"], self.stream)
            return

        elif self.type_b == "int8" and self.type_a == "bfloat16" and self.type_c == "bfloat16":
            self._mctlass_op.mctlass_w8a16_scaled_mm(self.A, self.B, self.C, self.scale_b, self.bias,
                                                     self.config["group_size"], self.stream)
            return

        elif self.type_b == "quint4x2" and self.type_a == "bfloat16" and self.type_c == "bfloat16":
            self._mctlass_op.mctlass_w4a16_scaled_mm(self.A, self.B, self.C, self.scale_b, self.bias,
                                                     self.config["group_size"], self.stream)
            return
