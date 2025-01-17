import os
import sys
import time
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse



class WatermarkV1Base:
    d = None
    t = None
    s = None
    scale_wm = None
    wm_num = None
    mListbase = None
    U_perm_list = None
    wm_index_list = None
    Am_perm_list = None
    numit = None
    beta = None
    rho = None
    model_type = None
    output_dir = None

    def __init__(self, model_dir: str, model_type: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="float16", trust_remote_code=True, device_map="auto")
        self.model_type = model_type
    
    def load_ori_model(self, ori_model_dir: str):
        self.model_ori = AutoModelForCausalLM.from_pretrained(ori_model_dir, torch_dtype="float16", trust_remote_code=True, device_map="auto")

    def compute_Am(self,Wq, Wk, Embedding, m):
        return
    
    def generate_Am_perm_list(self, d, t, Am_ori):
        return

    def get_final_Am(self,t, Am_ori, Am_perm_list):
        for i in range(t):
            Am_ori[i] = Am_ori[i][Am_perm_list[i]]

    def compute_Am_condition_number(self, d, t, Am):
        A2 = Am[:, d-t:].to(torch.float32)
        # Calculate singular values
        U, S, V = torch.svd(A2)

        # S contains singular values, condition number is the ratio of the largest singular value to the smallest singular value
        condition_number = S.max() / S.min()
        # print(f"Condition Number of Am: {condition_number.item()}")
        return condition_number.item()

    def compute_Am_combine(self, Wq, Wk, Embedding, mList):
        Am_List=[]
        for i in range(len(mList)):
            Am = self.compute_Am(Wq, Wk, Embedding, mList[i])
            Am_List.append(Am[i])

        Am_combine = torch.stack(Am_List)
        return Am_combine

    def compute_watermark(self, Embedding, Am, p, d, t, li):
        Eli = Embedding[li]
        Elp = Eli[p].unsqueeze(1)

        x1 = Elp[:d-t,:]*1
        b = torch.zeros(t, 1)
        A = Am[:t,:]*1
        A1 = A[:, :d-t]
        A2 = A[:, d-t:]

        A1_32 = A1.to(torch.float32)
        A2_32 = A2.to(torch.float32)
        x1_32 = x1.to(torch.float32)
        rhs_32 = -A1_32 @ x1_32

        x2_32 = torch.linalg.solve(A2_32, rhs_32)
        x2 = x2_32.to(torch.float16)
        return x2
    
    def generate_watermark_list(self, Embedding, Am, d, t, U_perm_list, wm_index_list):
        wm_list = []
        for i in range(len(U_perm_list)):
            wm = self.compute_watermark(Embedding, Am, U_perm_list[i], d, t, wm_index_list[i])
            wm_list.append(wm)
        return wm_list

    def insert_watermark(self, Embedding, d, t, U_perm_list, wm_index_list, wm_list, scale_wm):
        for i in range(len(U_perm_list)):
            li = wm_index_list[i]
            p = U_perm_list[i]
            invp = torch.argsort(p)
            wm = wm_list[i]

            # difference = Embedding[li][p][d-t:] - wm.view(-1)/scale_wm
            # print("difference:", difference.data)
            
            self.model.model.embed_tokens.weight.data[li] = torch.cat((Embedding[li][p][:d-t], wm.view(-1)/scale_wm), dim=0)[invp].data

    def generate_wm_index_list(self, wm_num, index_size, mList):
        while True:
            wm_index_list = torch.randperm(index_size)[:wm_num]
            if not torch.any(torch.isin(wm_index_list, mList)):
                break
        return wm_index_list

    def watermark_model(self):
        return
    
    def save_key(self):
        file1 = f"{self.output_dir}/key/{self.model_type}_wm_index_list_{self.t}_{self.wm_num}_V1.pt"
        torch.save(self.wm_index_list, file1)
        file2 = f"{self.output_dir}/key/{self.model_type}_SK_U_perm_list_{self.t}_{self.wm_num}_V1.pt"
        torch.save(self.U_perm_list, file2)
        file3 = f"{self.output_dir}/key/{self.model_type}_Am_perm_list_{self.t}_{self.wm_num}_V1.pt"
        torch.save(self.Am_perm_list, file3)
    
    def load_key_default(self):
        file1 = f"{self.output_dir}/key/{self.model_type}_wm_index_list_{self.t}_{self.wm_num}_V1.pt"
        file2 = f"{self.output_dir}/key/{self.model_type}_SK_U_perm_list_{self.t}_{self.wm_num}_V1.pt"
        file3 = f"{self.output_dir}/key/{self.model_type}_Am_perm_list_{self.t}_{self.wm_num}_V1.pt"
        self.wm_index_list = torch.load(file1)
        self.U_perm_list = torch.load(file2)
        self.Am_perm_list = torch.load(file3)
    
    def load_key(self, key_dir):
        file1 = f"{key_dir}/{self.model_type}_wm_index_list_{self.t}_{self.wm_num}_V1.pt"
        self.wm_index_list = torch.load(file1)
        file2 = f"{key_dir}/{self.model_type}_SK_U_perm_list_{self.t}_{self.wm_num}_V1.pt"
        self.U_perm_list = torch.load(file2)
        file3 = f"{key_dir}/{self.model_type}_Ak_perm_list_{self.t}_{self.wm_num}_V1.pt"
        self.Am_perm_list = torch.load(file3)
    
    def save_model(self):
        model_dir = f"{self.output_dir}/model/{self.model_type}_WM_{self.t}_{self.wm_num}_V1"
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
    
    def wm_order_compute(self, Embedding, Am, p, d, t, scale_wm, li, numit):
 
        # Preallocate the shape of wm_detect
        wm_detect_tmp = Embedding[li][p].unsqueeze(1)
        wm_detect = torch.empty_like(wm_detect_tmp)
 
        # Only need to do one slice and scale operation
        wm_detect[:d-t, :] = wm_detect_tmp[:d-t, :]
        wm_detect[d-t:, :] = wm_detect_tmp[d-t:, :] * scale_wm
 
        # Calculate the benchmark value sum_wm of the watermark
        sum_wm = torch.abs(torch.sum(torch.matmul(Am, wm_detect))).item()
        print("watermark place sum:", sum_wm)
 
        order = 0
 
        # Batch generate numit random permutations and combine them into a batch tensor
        permutations = torch.stack([torch.randperm(d) for _ in range(numit)])
 
        # Predefine the batch storage for wm_detecti
        wm_detecti_batch = torch.empty((numit, d, 1), dtype=Embedding.dtype, device=Embedding.device)
 
        # Replace cat operation with slice in batch processing
        for i in range(numit):
            pi = permutations[i]
            wm_detect_tmp = Embedding[li][pi].unsqueeze(1)
            wm_detecti_batch[i, :d-t, :] = wm_detect_tmp[:d-t, :]
            wm_detecti_batch[i, d-t:, :] = wm_detect_tmp[d-t:, :] * scale_wm
 
        # Perform batch matrix multiplication and calculate absolute value
        sum_batch = torch.abs(torch.sum(torch.matmul(Am, wm_detecti_batch), dim=1)).unsqueeze(1)
 
        # Count the number of times sumi < sum_wm
        order = (sum_batch < sum_wm).sum().item()
 
        return order

    def embedding_quick_recover(self):
        Embedding_ori = self.model_ori.model.embed_tokens.weight.data
        Embedding_wm = self.model.model.embed_tokens.weight.data
        norm1 = Embedding_ori.norm(p=2, dim=0, keepdim=True)
        norm2 = Embedding_wm.norm(p=2, dim=0, keepdim=True)
        Embedding_ori_norm = Embedding_ori / norm1
        Embedding_wm_norm = Embedding_wm / norm2
        p=[]
        for i in tqdm(range(self.d)):
            max_value, max_index = torch.max(torch.matmul(Embedding_wm_norm.transpose(-2,-1),Embedding_ori_norm[:,i]), dim=0)
            p.append(max_index.item())
        self.model.model.embed_tokens.weight.data = self.model.model.embed_tokens.weight.data[:,p]

    def watermark_detect(self):
        return

    

class WatermarkV1Llama3(WatermarkV1Base):
    d = 4096
    s = 128256
    mListbase = torch.tensor([49666, 45408, 90958, 28075, 57922, 98504, 910, 32202, 81619, 93282, 27432, 104376, 6915, 97862, 10774, 43762, 84976, 46152, 81618, 34108])
    def __init__(self, model_name: str, model_type: str, t, wm_num, numit, beta, rho, scale_wm, output_dir):
        super().__init__(model_name, model_type)
        self.t = t
        self.wm_num = wm_num
        self.numit = numit
        self.beta = beta
        self.rho = rho
        self.scale_wm = scale_wm
        self.output_dir = output_dir
        
    def compute_Am(self,Wq, Wk, Embedding, m):
        Em = Embedding[m]
        Q1 = Wq(Em)[:1024].unsqueeze(1)
        Q2 = Wq(Em)[1024:2048].unsqueeze(1)
        Q3 = Wq(Em)[2048:3072].unsqueeze(1)
        Q4 = Wq(Em)[3072:].unsqueeze(1)
        K = Wk(Em).unsqueeze(1)
        Am1 = torch.matmul(Q1,K.transpose(-2,-1))
        Am2 = torch.matmul(Q2,K.transpose(-2,-1))
        Am3 = torch.matmul(Q3,K.transpose(-2,-1))
        Am4 = torch.matmul(Q4,K.transpose(-2,-1))
        Am = torch.cat((Am1, Am2, Am3, Am4), dim=1)
        return Am
    
    def generate_Am_perm_list(self, d, t, Am_ori):
        Am_tmp = torch.zeros(t,d)
        tau = t+5
        for i in range(1000000):
            Am_tmp.zero_()
            Am_perm_list=[]
            for j in range(t):
                p = torch.randperm(d)
                Am_tmp[j] = Am_ori[j][p]
                Am_perm_list.append(p)
            cd_num = self.compute_Am_condition_number(d, t, Am_tmp)
            if cd_num < tau:
                print("condition number:", cd_num)
                break
            if i % 2000 == 0:
                tau = tau+5
        return Am_perm_list
    
    def watermark_model(self):
        Embedding = self.model.model.embed_tokens.weight.data
        Wq=self.model.model.layers[0].self_attn.q_proj
        Wk=self.model.model.layers[0].self_attn.k_proj
        mList = self.mListbase[:self.t]
        Am = self.compute_Am_combine(Wq,Wk, Embedding, mList)
        self.Am_perm_list = self.generate_Am_perm_list(self.d, self.t, Am)
        self.get_final_Am(self.t, Am, self.Am_perm_list)
        self.wm_index_list = self.generate_wm_index_list(self.wm_num, self.s, mList)
        self.U_perm_list = []
        for _ in range(self.wm_num):
            p = torch.randperm(self.d)
            self.U_perm_list.append(p)
        wm_list = self.generate_watermark_list(Embedding, Am, self.d, self.t, self.U_perm_list, self.wm_index_list)
        self.insert_watermark(Embedding, self.d, self.t, self.U_perm_list, self.wm_index_list, wm_list, self.scale_wm)
    
    def watermark_detect(self):
        Embedding = self.model.model.embed_tokens.weight.data
        Wq=self.model.model.layers[0].self_attn.q_proj
        Wk=self.model.model.layers[0].self_attn.k_proj
        mList = self.mListbase[:self.t]
        Am = self.compute_Am_combine(Wq,Wk, Embedding, mList)
        self.get_final_Am(self.t, Am, self.Am_perm_list)

        detect = 0
        order_list = []
        start_time = time.time()
        self.embedding_quick_recover()
        end_time = time.time()
        print(f"Recover embedding time: {end_time - start_time} seconds")
        for i in range(len(self.U_perm_list)):
            li = self.wm_index_list[i]
            p = self.U_perm_list[i]
            orderi = self.wm_order_compute(Embedding, Am, p, self.d, self.t, self.scale_wm, li, self.numit)
            print("order:", orderi)
            order_list.append(orderi)
            if orderi < self.numit*self.rho:
                detect+=1
        print("Detect watermark number:", detect)
        if detect >= self.beta:
            print("Detect watermark success")
        else:
            print("Detect watermark failed")
        return order_list
    
class WatermarkV1Gemma(WatermarkV1Base):
    d = 2048
    s = 256000
    mListbase = torch.tensor([49666, 45408, 90958, 28075, 57922, 98504, 910, 32202, 81619, 93282, 27432, 104376, 6915, 97862, 10774, 43762, 84976, 46152, 81618, 34108])
    def __init__(self, model_name: str, model_type: str, t, wm_num, numit, beta, rho, scale_wm, output_dir):
        super().__init__(model_name, model_type)
        self.t = t
        self.wm_num = wm_num
        self.numit = numit
        self.beta = beta
        self.rho = rho
        self.scale_wm = scale_wm
        self.output_dir = output_dir
        
    def compute_Am(self,Wq, Wk, Embedding, m):
        Em = Embedding[m]
        Q1 = Wq(Em)[:256].unsqueeze(1)
        Q2 = Wq(Em)[256:512].unsqueeze(1)
        Q3 = Wq(Em)[512:768].unsqueeze(1)
        Q4 = Wq(Em)[768:1024].unsqueeze(1)
        Q5 = Wq(Em)[1024:1280].unsqueeze(1)
        Q6 = Wq(Em)[1280:1536].unsqueeze(1)
        Q7 = Wq(Em)[1536:1792].unsqueeze(1)
        Q8 = Wq(Em)[1792:].unsqueeze(1)
        K = Wk(Em).unsqueeze(1)
        Am1 = torch.matmul(Q1,K.transpose(-2,-1))
        Am2 = torch.matmul(Q2,K.transpose(-2,-1))
        Am3 = torch.matmul(Q3,K.transpose(-2,-1))
        Am4 = torch.matmul(Q4,K.transpose(-2,-1))
        Am5 = torch.matmul(Q5,K.transpose(-2,-1))
        Am6 = torch.matmul(Q6,K.transpose(-2,-1))
        Am7 = torch.matmul(Q7,K.transpose(-2,-1))
        Am8 = torch.matmul(Q8,K.transpose(-2,-1))
        Am = torch.cat((Am1, Am2, Am3, Am4, Am5, Am6, Am7, Am8), dim=1)
        return Am
    
    def generate_Am_perm_list(self, d, t, Am_ori):
        Am_tmp = torch.zeros(t,d)
        tau = 2*t
        for i in range(1000000):
            Am_tmp.zero_()
            Am_perm_list=[]
            for j in range(t):
                p = torch.randperm(d)
                Am_tmp[j] = Am_ori[j][p]
                Am_perm_list.append(p)
            cd_num = self.compute_Am_condition_number(d, t, Am_tmp)
            if cd_num < tau:
                print("condition number:", cd_num)
                break
            if i % 2000 == 0:
                tau = tau+5
        return Am_perm_list
    
    def watermark_model(self):
        Embedding = self.model.model.embed_tokens.weight.data
        Wq=self.model.model.layers[0].self_attn.q_proj
        Wk=self.model.model.layers[0].self_attn.k_proj
        mList = self.mListbase[:self.t]
        Am = self.compute_Am_combine(Wq,Wk, Embedding, mList)
        self.Am_perm_list = self.generate_Am_perm_list(self.d, self.t, Am)
        self.get_final_Am(self.t, Am, self.Am_perm_list)
        self.wm_index_list = self.generate_wm_index_list(self.wm_num, self.s, mList)
        self.U_perm_list = []
        for _ in range(self.wm_num):
            p = torch.randperm(self.d)
            self.U_perm_list.append(p)
        wm_list = self.generate_watermark_list(Embedding, Am, self.d, self.t, self.U_perm_list, self.wm_index_list)
        self.insert_watermark(Embedding, self.d, self.t, self.U_perm_list, self.wm_index_list, wm_list, self.scale_wm)
    
    def watermark_detect(self):
        Embedding = self.model.model.embed_tokens.weight.data
        Wq=self.model.model.layers[0].self_attn.q_proj
        Wk=self.model.model.layers[0].self_attn.k_proj
        mList = self.mListbase[:self.t]
        Am = self.compute_Am_combine(Wq,Wk, Embedding, mList)
        self.get_final_Am(self.t, Am, self.Am_perm_list)

        detect = 0
        order_list = []
        start_time = time.time()
        self.embedding_quick_recover()
        end_time = time.time()
        print(f"Recover embedding time: {end_time - start_time} seconds")
        for i in range(len(self.U_perm_list)):
            li = self.wm_index_list[i]
            p = self.U_perm_list[i]
            orderi = self.wm_order_compute(Embedding, Am, p, self.d, self.t, self.scale_wm, li, self.numit)
            print("order:", orderi)
            order_list.append(orderi)
            if orderi < self.numit*self.rho:
                detect+=1
        print("Detect watermark number:", detect)
        if detect >= self.beta:
            print("Detect watermark success")
        else:
            print("Detect watermark failed")
        return order_list
    
class WatermarkV1Phi3(WatermarkV1Base):
    d = 3072
    s = 32064
    mListbase = torch.tensor([1522, 5866, 8412, 10537, 13668, 17920,   22559, 27541, 29663, 30744, 12264, 2258, 8411, 28965, 23312, 18620,  6882,  7845, 20914,  1863])
    def __init__(self, model_name: str, model_type: str, t, wm_num, numit, beta, rho, scale_wm, output_dir):
        super().__init__(model_name, model_type)
        self.t = t
        self.wm_num = wm_num
        self.numit = numit
        self.beta = beta
        self.rho = rho
        self.scale_wm = scale_wm
        self.output_dir = output_dir
     
    def compute_Am(self, Wqkv, Embedding, m):
        Em = Embedding[m]
        Q = Wqkv(Em)[:3072].unsqueeze(1)
        K = Wqkv(Em)[3072:6144].unsqueeze(1)
        Am = torch.matmul(Q,K.transpose(-2,-1))

        return Am
    
    def compute_Am_combine(self,Wqkv, Embedding, mList):
        Am_List=[]
        for i in range(len(mList)):
            Am = self.compute_Am(Wqkv, Embedding, mList[i])
            Am_List.append(Am[i])

        Am_combine = torch.stack(Am_List)
        return Am_combine
    
    # generate Am perm list
    def generate_Am_perm_list(self, d, t, Am_ori):
        Am_tmp = torch.zeros(t,d)
        tau = 4*t
        for i in range(1000000):
            Am_tmp.zero_()
            Am_perm_list=[]
            for j in range(t):
                p = torch.randperm(d)
                Am_tmp[j] = Am_ori[j][p]
                Am_perm_list.append(p)
            cd_num = self.compute_Am_condition_number(d, t, Am_tmp)
            if cd_num < tau:
                print("condition number:", cd_num)
                break
            if i % 2000 == 0:
                tau = tau+5
        return Am_perm_list
    
    # insert watermark
    def watermark_model(self):
        Embedding = self.model.model.embed_tokens.weight.data
        Wqkv=self.model.model.layers[0].self_attn.qkv_proj
        mList = self.mListbase[:self.t]
        Am = self.compute_Am_combine(Wqkv, Embedding, mList)
        self.Am_perm_list = self.generate_Am_perm_list(self.d, self.t, Am)
        self.get_final_Am(self.t, Am, self.Am_perm_list)
        self.wm_index_list = self.generate_wm_index_list(self.wm_num, self.s, mList)
        self.U_perm_list = []
        for _ in range(self.wm_num):
            p = torch.randperm(self.d)
            self.U_perm_list.append(p)
        wm_list = self.generate_watermark_list(Embedding, Am, self.d, self.t, self.U_perm_list, self.wm_index_list)
        self.insert_watermark(Embedding, self.d, self.t, self.U_perm_list, self.wm_index_list, wm_list, self.scale_wm)
    
    # detect watermark
    def watermark_detect(self):
        Embedding = self.model.model.embed_tokens.weight.data
        Wqkv=self.model.model.layers[0].self_attn.qkv_proj
        mList = self.mListbase[:self.t]
        Am = self.compute_Am_combine(Wqkv, Embedding, mList)
        self.get_final_Am(self.t, Am, self.Am_perm_list)

        detect = 0
        order_list = []
        start_time = time.time()
        self.embedding_quick_recover()
        end_time = time.time()
        print(f"Recover embedding time: {end_time - start_time} seconds")
        for i in range(len(self.U_perm_list)):
            li = self.wm_index_list[i]
            p = self.U_perm_list[i]
            orderi = self.wm_order_compute(Embedding, Am, p, self.d, self.t, self.scale_wm, li, self.numit)
            print("order:", orderi)
            order_list.append(orderi)
            if orderi < self.numit*self.rho:
                detect+=1
        print("Detect watermark number:", detect)
        if detect >= self.beta:
            print("Detect watermark success")
        else:
            print("Detect watermark failed")
        return order_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Watermark")
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--model_type', type=str, required=True, help='Model type')
    parser.add_argument('--ori_model_dir', type=str, help='Original model directory')
    parser.add_argument('--t', type=int, default=10, help='watermark length')
    parser.add_argument('--wm_num', type=int, default=50, help='Watermark number (default: 50)')
    parser.add_argument('--numit', type=int, default=10000, help='Number of iterations (default: 10000)')
    parser.add_argument('--beta', type=int, default=35, help='Threshold for watermark detection (default: 35)')
    parser.add_argument('--rho', type=int, default=0.3, help='Threshold for watermark detection (default: 0.3)')
    parser.add_argument('--scale_wm', type=int, default=1000, help='scale parameter for watermark (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory (default: ./output)')
    parser.add_argument('--key_dir', type=str, help='custom key directory')
    parser.add_argument('--command', type=str, choices=['insert', 'extract'], required=True, help='Command to execute: insert or extract watermark')

    args = parser.parse_args()
    
    # Initialize watermark model
    if args.model_type == "Llama-3-8B":
        watermark_model = WatermarkV1Llama3(args.model_dir, args.model_type, args.t, args.wm_num, args.numit, args.beta, args.rho, args.scale_wm, args.output_dir)
    elif args.model_type == "Gemma-2B":
        watermark_model = WatermarkV1Gemma(args.model_dir, args.model_type, args.t, args.wm_num, args.numit, args.beta, args.rho, args.scale_wm, args.output_dir)
    elif args.model_type == "Phi3-4B":
        watermark_model = WatermarkV1Phi3(args.model_dir, args.model_type, args.t, args.wm_num, args.numit, args.beta, args.rho, args.scale_wm, args.output_dir)
   
    # Execute command
    if args.command == 'insert':
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            os.makedirs(f"{args.output_dir}/key")
            os.makedirs(f"{args.output_dir}/model")
        print("start insert watermark for model:", args.model_type)
        start_time = time.time()
        watermark_model.watermark_model()
        end_time = time.time()
        print(f"Inserting watermark time: {end_time - start_time} seconds")
        watermark_model.save_key()
        watermark_model.save_model()
        print("insert watermark success")
    elif args.command == 'extract':
        print("start extract watermark for model:", args.model_type)
        print(f"start loading original model for embedding recovery")
        if not args.ori_model_dir:
            print("missing original model directory")
            sys.exit(1)
        else:
            watermark_model.load_ori_model(args.ori_model_dir)
        if args.key_dir:
            watermark_model.load_key(args.key_dir)
        else:
            print("no key directory given, use default key")
            watermark_model.load_key_default()
        start_time = time.time()
        order_list = watermark_model.watermark_detect()
        end_time = time.time()
        print(f"Extracting watermark time: {end_time - start_time} seconds")
        print("Extracted watermark order list:", order_list)
        print("order < 10 number:", sum(1 for x in order_list if x < 10))
        print("order < 100 number:", sum(1 for x in order_list if x < 100))
        print("order < 1000 number:", sum(1 for x in order_list if x < 1000))
        print("order < 2000 number:", sum(1 for x in order_list if x < 2000))
        print("order < 3000 number:", sum(1 for x in order_list if x < 3000))
        print("order < 5000 number:", sum(1 for x in order_list if x < 5000))