import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re
import time
from tqdm import tqdm

llama3_paradict_permute={
    "part1":[
        "self_attn.v_proj.weight"
    ],
    "part2":[
        "self_attn.o_proj.weight"
    ],
    "part3":[
        "mlp.gate_proj.weight"
    ],
    "part4":[
        "mlp.up_proj.weight"
    ],
    "part5":[
        "mlp.down_proj.weight"
    ],
}

llama3_paradict_invert_mat={
    "part1":[
        "self_attn.q_proj.weight"
    ],
    "part2":[
        "self_attn.k_proj.weight"
    ]
}

llama_paradict_scale={
    "part1":[
        "self_attn.q_proj.weight"
    ],
    "part2":[
        "self_attn.k_proj.weight"
    ],
    "part3":[
        "self_attn.v_proj.weight"
    ],
    "part4":[
        "mlp.gate_proj.weight"
    ],
    "part5":[
        "mlp.up_proj.weight"
    ],
    "part6":[
        "input_layernorm.weight",
    ],
    "part7":[
        "post_attention_layernorm.weight"
    ]
}

def get_layer_number(param_name):
    """
    从参数名称中提取层数信息
    :param param_name: 参数名称字符串，例如 "model.layers.0.self_attn.o_proj.weight"
    :return: 层数（整数），如果无法提取则返回 None
    """
    match = re.search(r'\.(\d+)\.', param_name)
    if match:
        return int(match.group(1))
    return None

def WM_llama3_with_permute(model, permlist1, permlist2, paradict, NGROUP, NHEADS, DHEAD):
    with torch.no_grad():
        for name, param in model.named_parameters():
            layer = get_layer_number(name)
            if any(sub in name for sub in paradict["part1"]):
                for i in range(NGROUP):
                    start = i * (DHEAD)
                    end = (i + 1) * (DHEAD)
                    param.data[start:end] = param.data[start:end][permlist1[layer]]
            if any(sub in name for sub in paradict["part2"]):
                for i in range(NHEADS):
                    start = i * (DHEAD)
                    end = (i + 1) * (DHEAD)
                    param.data[:,start:end] = param.data[:,start:end][:,permlist1[layer]]
            if any(sub in name for sub in paradict["part3"]):
                param.data = param.data[permlist2[layer]]
            if any(sub in name for sub in paradict["part4"]):
                param.data = param.data[permlist2[layer]]
            if any(sub in name for sub in paradict["part5"]):
                param.data = param.data[:,permlist2[layer]]

def WM_llama3_with_invert_mat(model, Plist, P_inv_Tlist, paradict):
    with torch.no_grad():
        for name, param in model.named_parameters():
            layer = get_layer_number(name)
            if any(sub in name for sub in paradict["part1"]):
                param.data = param.data @ Plist[layer].cuda()
            if any(sub in name for sub in paradict["part2"]):
                param.data = param.data @ P_inv_Tlist[layer].cuda()

def WM_llama3_slice_with_invert_mat(model, Plist, P_inv_Tlist, paradict, NLAYER):
    with torch.no_grad():
        for i in range(NLAYER):
            model.model.layers[i].self_attn.q_proj.weight.data = model.model.layers[i].self_attn.q_proj.weight.data @ Plist[i].cuda()
            model.model.layers[i].self_attn.k_proj.weight.data = model.model.layers[i].self_attn.k_proj.weight.data @ P_inv_Tlist[i].cuda()

def WM_llama3_with_scale(model, scalelist1, scalelist2, paradict):
    with torch.no_grad():
        for name, param in model.named_parameters():
            layer = get_layer_number(name)
            if any(sub in name for sub in paradict["part1"]):
                param.data = param.data / scalelist1[layer].cuda()
            if any(sub in name for sub in paradict["part2"]):
                param.data = param.data / scalelist1[layer].cuda()
            if any(sub in name for sub in paradict["part3"]):
                param.data = param.data / scalelist1[layer].cuda()
            if any(sub in name for sub in paradict["part4"]):
                param.data = param.data / scalelist2[layer].cuda()
            if any(sub in name for sub in paradict["part5"]):
                param.data = param.data / scalelist2[layer].cuda()
            if any(sub in name for sub in paradict["part6"]):
                param.data = param.data * scalelist1[layer].cuda()
            if any(sub in name for sub in paradict["part7"]):
                param.data = param.data * scalelist2[layer].cuda()

def frobenius_norm_difference(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    return torch.norm(tensor1 - tensor2, p='fro')**2 / tensor1.numel()



def extract_WM_for_permute_byte1_layeri(modelori, modelWM, permlist, layer, NGROUP, NHEADS, DHEAD):
    mselist=[]
    T1 = modelWM.model.layers[layer].self_attn.v_proj.weight.data
    for i in range(len(permlist)):
        T2 = modelori.model.layers[layer].self_attn.v_proj.weight.data
        for j in range(NGROUP):
            start = j * (DHEAD)
            end = (j + 1) * (DHEAD)
            T2[start:end] = T2[start:end][permlist[i]]
        mselist.append(frobenius_norm_difference(T1, T2).item())
    return mselist.index(min(mselist))        

def extract_WM_for_permute_byte2_layeri(modelori, modelWM, permlist, layer):
    mselist=[]
    T1 = modelWM.model.layers[layer].mlp.gate_proj.weight.data
    for i in range(len(permlist)):
        T2 = modelori.model.layers[layer].mlp.gate_proj.weight.data[permlist[i]]
        mselist.append(frobenius_norm_difference(T1, T2).item())
    return mselist.index(min(mselist))

def extract_WM_for_permute(modelori, modelWM, permspace1, permspace2, NLAYER, NGROUP, NHEADS, DHEAD):
    perm_byte=[]
    for i in range(NLAYER):
        byte1 = extract_WM_for_permute_byte1_layeri(modelori, modelWM, permspace1[i], i, NGROUP, NHEADS, DHEAD)
        byte2 = extract_WM_for_permute_byte2_layeri(modelori, modelWM, permspace2[i], i)
        perm_byte.append([byte1, byte2])
    return perm_byte

def extract_WM_for_scaling_byte1_layeri(modelori, modelWM, scalelist, layer):
    mselist=[]
    T1 = modelWM.model.layers[layer].input_layernorm.weight.data
    for i in range(len(scalelist)):
        T2 = modelori.model.layers[layer].input_layernorm.weight.data * scalelist[i].cuda()
        mselist.append(frobenius_norm_difference(T1, T2).item())
    return mselist.index(min(mselist))

def extract_WM_for_scaling_byte2_layeri(modelori, modelWM, scalelist, layer):
    mselist=[]
    T1 = modelWM.model.layers[layer].post_attention_layernorm.weight.data
    for i in range(len(scalelist)):
        T2 = modelori.model.layers[layer].post_attention_layernorm.weight.data * scalelist[i].cuda()
        mselist.append(frobenius_norm_difference(T1, T2).item())
    return mselist.index(min(mselist))

def extract_WM_for_scaling(modelori, modelWM, scalespace1, scalespace2):
    scaling_byte=[]
    for i in range(NLAYER):
        byte1 = extract_WM_for_scaling_byte1_layeri(modelori, modelWM, scalespace1[i], i)
        byte2 = extract_WM_for_scaling_byte2_layeri(modelori, modelWM, scalespace2[i], i)
        scaling_byte.append([byte1, byte2])
    return scaling_byte

def extract_WM_for_invmat_byte_layeri(modelori, modelWM, Plist, layer):
    mselist=[]
    T1 = modelWM.model.layers[layer].self_attn.q_proj.weight.data
    for i in range(len(Plist)):
        T2 = modelori.model.layers[layer].self_attn.q_proj.weight.data @ Plist[i].cuda()
        mselist.append(frobenius_norm_difference(T1, T2).item())
    return mselist.index(min(mselist))
    
def extract_WM_for_invmat(modelori, modelWM, Pspace, NLAYER):
    invmat_byte=[]
    for i in range(NLAYER):
        byte = extract_WM_for_invmat_byte_layeri(modelori, modelWM, Pspace[i], i)
        invmat_byte.append(byte)
    return invmat_byte

def generate_scaling_vector(n: int) -> torch.Tensor:
  """
  生成满足 log10(a) 在 [-1, 1] 上均匀分布的缩放向量 a。

  Args:
      n: 缩放向量的维度。

  Returns:
      形状为 (n,) 的 tensor，满足 log10(a) 在 [-1, 1] 上均匀分布。
  """
  # 在 [-1, 1] 区间上生成均匀分布的随机数
  log10_a = torch.rand(n) * 2 - 1

  # 计算 a
  a = 10 ** log10_a

  return a

def build_matrix_P(dim: int, phi: float = 0.5, lambda_factor: float = 1.0):
  """
  构建可逆矩阵 P。

  Args:
      dim: 模型的维度，需要是偶数。
      phi: 旋转角度。
      lambda_factor: 缩放因子。

  Returns:
      P: 可逆矩阵，形状为 (dim, dim)。
  """
  assert dim % 2 == 0, "Dimension must be even"
  P = torch.zeros(dim, dim)
  for i in range(0, dim, 2):
    rotation_matrix = torch.tensor([
        [lambda_factor * torch.cos(torch.tensor(phi)), -lambda_factor * torch.sin(torch.tensor(phi))],
        [lambda_factor * torch.sin(torch.tensor(phi)), lambda_factor * torch.cos(torch.tensor(phi))]
    ])
    P[i:i+2, i:i+2] = rotation_matrix
  return P

def invert_matrix_P(P: torch.Tensor, lambda_factor: float = 1.0):
  """
  计算可逆矩阵 P 的 (P^T)^-1。

  Args:
      P: 可逆矩阵，形状为 (dim, dim)。
      lambda_factor: 缩放因子。

  Returns:
      P_inv_T: (P^T)^-1，形状为 (dim, dim)。
  """
  dim = P.shape[0]
  P_inv_T = torch.zeros_like(P)
  for i in range(0, dim, 2):
      rotation_matrix = P[i:i + 2, i:i + 2]
      determinant = lambda_factor**2
      
      inverse_rotation_matrix = torch.tensor([
            [rotation_matrix[1,1], -rotation_matrix[1,0]],
            [-rotation_matrix[0,1], rotation_matrix[0,0]]
        ]) / determinant

      P_inv_T[i:i+2, i:i+2] = inverse_rotation_matrix.T
  return P_inv_T

NLAYER = 32
NLAYER_QK=4
NHEADS = 32
DHEAD = 128
DFF  = 14336
NGROUP = 8
DMODEL=4096
benchtimes=100

k=8
model_path = "/path/Meta-Llama-3-8B-Instruct"
print("start Key generation ...")
permspace1 = [[torch.randperm(DHEAD) for _ in range (2**k)]  for _ in range(NLAYER)]
permspace2 = [[torch.randperm(DFF) for _ in range (2**k)]  for _ in range(NLAYER)]
scalespace1 = [[generate_scaling_vector(DMODEL).half() for _ in range (2**k)]  for _ in range(NLAYER)]
scalespace2 = [[generate_scaling_vector(DMODEL).half() for _ in range (2**k)]  for _ in range(NLAYER)]
Pspace = [[build_matrix_P(DMODEL, torch.rand(1)[0].item(), torch.rand(1)[0].item()).half() for _ in range (2**k)] for _ in range(NLAYER_QK)]
invPspace = [[invert_matrix_P(P).half() for P in Pspace[i]] for i in range(NLAYER_QK)]

WMkey = torch.randint(0, 256, (32, 5))

print("Key generation finish...")


# permute time test
permlist1 = [permspace1[i][WMkey[i][0]]  for i in range(NLAYER)]
permlist2 = [permspace2[i][WMkey[i][1]]  for i in range(NLAYER)]

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    temperature=0,
    do_sample=False,
    device_map="cuda:0",
    )

print("start permute WM insertion ...")
start_time = time.time()
for _ in tqdm(range(benchtimes)):
    WM_llama3_with_permute(model, permlist1, permlist2, llama3_paradict_permute, NGROUP, NHEADS, DHEAD)
end_time = time.time()
print(f"Watermark insertion time for permute: {(end_time - start_time)/benchtimes} seconds")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    temperature=0,
    do_sample=False,
    device_map="cuda:0",
    )
modelori = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    temperature=0,
    do_sample=False,
    device_map="cuda:0",
    )

WM_llama3_with_permute(model, permlist1, permlist2, llama3_paradict_permute, NGROUP, NHEADS, DHEAD)
print("start permute WM extraction ...")
start_time = time.time()
for _ in tqdm(range(benchtimes)):
    perm_byte = extract_WM_for_permute(modelori, model, permspace1, permspace2, NLAYER, NGROUP, NHEADS, DHEAD)
end_time = time.time()
print(f"Watermark extraction time for permute: {(end_time - start_time)/benchtimes} seconds")

# scaling time test
scalelist1 = [scalespace1[i][WMkey[i][2]]  for i in range(NLAYER)]
scalelist2 = [scalespace2[i][WMkey[i][3]]  for i in range(NLAYER)]

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    temperature=0,
    do_sample=False,
    device_map="cuda:0",
    )

print("start scaling WM insertion ...")
start_time = time.time()
for _ in tqdm(range(benchtimes)):
    WM_llama3_with_scale(model, scalelist1, scalelist2, llama_paradict_scale)
end_time = time.time()
print(f"Watermark insertion time for scaling: {(end_time - start_time)/benchtimes} seconds")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    temperature=0,
    do_sample=False,
    device_map="cuda:0",
    )
modelori = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    temperature=0,
    do_sample=False,
    device_map="cuda:0",
    )
WM_llama3_with_scale(model, scalelist1, scalelist2, llama_paradict_scale)

print("start scaling WM extraction ...")
start_time = time.time()
for _ in tqdm(range(benchtimes)):
    scaling_code = extract_WM_for_scaling(modelori, model, scalespace1, scalespace2)
end_time = time.time()
print(f"Watermark extraction time for scaling: {(end_time - start_time)/benchtimes} seconds")

# QK_inv time test

Plist = [Pspace[i][WMkey[i][4]]  for i in range(NLAYER_QK)]
P_inv_Tlist = [invPspace[i][WMkey[i][4]]  for i in range(NLAYER_QK)]

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    temperature=0,
    do_sample=False,
    device_map="cuda:0",
    )

print("start QK_inv WM insertion ...")
start_time = time.time()
for _ in tqdm(range(benchtimes)):
    WM_llama3_slice_with_invert_mat(model, Plist, P_inv_Tlist, llama3_paradict_invert_mat, NLAYER_QK)
end_time = time.time()
print(f"Watermark insertion time for QK_inv: {(end_time - start_time)*(NLAYER/NLAYER_QK)/benchtimes} seconds")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    temperature=0,
    do_sample=False,
    device_map="cuda:0",
    )
modelori = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    temperature=0,
    do_sample=False,
    device_map="cuda:0",
    )
WM_llama3_slice_with_invert_mat(model, Plist, P_inv_Tlist, llama3_paradict_invert_mat, NLAYER_QK)

print("start QK_inv WM extraction ...")
start_time = time.time()
for _ in tqdm(range(benchtimes)):
    QKinv_code = extract_WM_for_invmat(modelori, model, Pspace, NLAYER_QK)
end_time = time.time()
print(f"Watermark extraction time for QK_inv: {(end_time - start_time)*(NLAYER/NLAYER_QK)/benchtimes} seconds")