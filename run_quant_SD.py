import numpy

import sys

import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from src.build_model import OffloadConfig, QuantConfig, build_model
from src.dp import get_cache_size
from transformers import GenerationConfig

from transformers import TextStreamer
import time
import argparse
import math
import datasets
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    path = "/clk/model"
    model_name = path
    quantized_model_name = path
    state_path = path
    assistant_checkpoint = "/clk/model/Mistral-7B-Instruct-v0.1-GPTQ"
    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_checkpoint,device_map="cuda:0",  
        # generation_config = GenerationConfig(
        #     um_assistant_tokens_schedule="heuristic", 
        #     num_assistant_tokens=5),
    )

    config = AutoConfig.from_pretrained(model_name)

    device = torch.device("cuda:0")

    main_size = args.size
    cache_strategy = get_cache_size(main_size,args.adapgate)
    print(cache_strategy)

    num_experts = config.num_local_experts

    offload_config = OffloadConfig(
        main_size=main_size,
        cache_strategy=cache_strategy,
        offload_size=config.num_hidden_layers * 8,
        buffer_size=8,
    )


    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256

    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )
    # quant
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)


    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )
    if args.adapgate:
        weight = [46.69189453125, 17.303466796875, 13.0157470703125, 7.640838623046875, 4.169464111328125, 2.2296905517578125, 1.2559890747070312, 0.8444786071777344, 0.6837844848632812, 0.5602836608886719, 0.5125999450683594, 0.4780292510986328, 0.44536590576171875, 0.4355907440185547, 0.38361549377441406, 0.30994415283203125, 0.23305416107177734, 0.1760721206665039, 0.13840198516845703, 0.1137852668762207, 0.10472536087036133, 0.09542703628540039, 0.08624792098999023, 0.07712841033935547, 0.06937980651855469, 0.06109476089477539, 0.0502467155456543, 0.042557716369628906, 0.03349781036376953, 0.025272369384765625, 0.020682811737060547, 0.02294778823852539]
        for idx, layer in enumerate(model.model.layers):
            layer.block_sparse_moe.threshold = math.sqrt(0.005/weight[idx])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    past_key_values = None
    sequence = None
    
    device = torch.device("cuda:0")
    
    dataset_name = "/clk/MoE-Infinity/dataset/bigbench"
    all_data = datasets.load_dataset(dataset_name+'/abstract_narrative_understanding')
    texts = all_data["validation"]

    # model_name = "/clk/model/Mixtral-8x7B-v0.1"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    output_token = 5
    batch_size = 1

    idx_text = 0

    # if past_key_values is None:
    #     attention_mask = torch.ones_like(input_ids)
    # else:
    #     seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
    #     attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

    for i in range(10):
        streamer = TimerStreamer()
        time_sum = 0
        num_tokens = 0
        batch = []
        for _ in range(batch_size):
            text = texts[idx_text]["inputs"]
            idx_text += 1
            batch.append(text)
        input_ids = tokenizer(
            batch, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True,
            padding=True,
        )["input_ids"]
        input_ids = input_ids.to(device)

        if past_key_values is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
            attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

        start_time = time.time()
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
        with profile(activities=activities) as prof:            result = model.generate(
                input_ids=input_ids,
                assistant_model=assistant_model,
                max_new_tokens=output_token,
                min_new_tokens=output_token,
                streamer=streamer,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                # mixtral-offload-wenz 
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
        # prof.export_chrome_trace("trace_{}.json".format(i+1))
        end_time = time.time()
        time_sum += end_time - start_time
        num_tokens += result["sequences"].shape[1]
                
        sequence = result["sequences"]
        past_key_values = result["past_key_values"]
        # print(f'input text: {batch}')
        # for i in range(result["sequences"].shape[0]):
        #     print(f'{i} output text: {tokenizer.decode(result["sequences"][i])}')
        print(f"Inputs {i+1}")
        print(f"Batch Size: {input_ids.shape[0]}")
        print(f"Prefilling time: {streamer.prefilling_time} seconds")
        print(f"Decoding time: {streamer.decoding_time} seconds")
        print(f"Decoding iterations: {streamer.decoding_iterations}")
        print(
            f"Decoding time per iteration: {streamer.decoding_time / streamer.decoding_iterations} seconds"
        )
        print(
            f"Time per output token(TPOT): {streamer.decoding_time / streamer.decoding_iterations / input_ids.shape[0]} seconds"
        )
        print("-------------------------------")


    # seq_len = 0
    # total_time = 0
    # total_tokens = 0
    # while True:
    #     print("User: ", end="")
    #     user_input = input()
    #     print("\n")

    #     user_entry = dict(role="user", content=user_input)
    #     input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

    #     if past_key_values is None:
    #         attention_mask = torch.ones_like(input_ids)
    #     else:
    #         seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
    #         attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

    #     print("Mixtral: ", end="")
    #     start_time = time.time()
    #     result = model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         past_key_values=past_key_values,
    #         streamer=streamer,
    #         do_sample=True,
    #         top_k=1,
    #         max_new_tokens=128,
    #         pad_token_id=tokenizer.eos_token_id,
    #         return_dict_in_generate=True,
    #         output_hidden_states=True,
    #     )
    #     end_time = time.time()
    #     print("\n")

    #     sequence = result["sequences"]
    #     past_key_values = result["past_key_values"]

    #     total_time += end_time - start_time
    #     total_tokens += sequence.size(1)

    #     # Calculate average time per token
    #     avg_time_per_token = total_time / 128
    #     print(f"Average time per token: {avg_time_per_token} seconds")


class TimerStreamer():
    def __init__(self, *args, **kwargs):
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.start_decoding = time.time()
        self.decoding_iterations += 1

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapgate',action='store_true')
    parser.add_argument('--size',type=int,default=64)
    args = parser.parse_args()
    main()


