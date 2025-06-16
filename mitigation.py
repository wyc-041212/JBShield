import torch
import json
import gc
import numpy as np
from tqdm import tqdm

from config import model_paths
from utils import load_model, get_judge_scores, get_output_prompt
from utils import interpret_difference_matrix, cosine_similarity


"""
For real applications, concept manipulation should be applied only when 
jailbreak prompts are detected which means this function should be called 
only when toxic and jailbreak concepts are both detected.

Here we use a simple version as all test data are jailbreak prompts.
"""
# Model with concept manipulation
class JBShieldM:
    def __init__(
        self,
        model,
        tokenizer,
        mean_harmless_embedding,
        mean_harmful_embedding,
        base_safety_vector,
        base_jailbreak_vector,
        threshold_safety,
        threshold_jailbreak,
        delta_safety,
        delta_jailbreak,
        selected_safety_layer_index,
        selected_jailbreak_layer_index,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mean_harmless_embedding = mean_harmless_embedding
        self.mean_harmful_embedding = mean_harmful_embedding
        self.base_safety_vector = base_safety_vector
        self.base_jailbreak_vector = base_jailbreak_vector
        self.threshold_safety = threshold_safety
        self.threshold_jailbreak = threshold_jailbreak
        """
        These deltas (scaling factor in our paper) can to be carefully tuned to ensure 
        that the model outputs readable text.
        """
        self.delta_safety = delta_safety
        self.delta_jailbreak = delta_jailbreak
        self.selected_safety_layer_index = selected_safety_layer_index
        self.selected_jailbreak_layer_index = selected_jailbreak_layer_index
        self.hooks = []
        # self.count = 20

    def detection(self, embeddings1, base_embedding, base_vector, threshold):
        """
        For real applications, concept manipulation should be applied only when jailbreak prompts are detected
        which means this function should be called only when toxic and jailbreak concepts are both detected.
        Here we use a simpler version as all test data are jailbreak prompts.
        """
        results = []
        for embed in embeddings1:
            vec, _ = interpret_difference_matrix(
                self.model,
                self.tokenizer,
                embed,
                base_embedding,
                return_tokens=False,
            )
            if cosine_similarity(vec, base_vector).item() >= threshold:
                results.append(1.0)
            else:
                results.append(0.0)
        return results

    def hook_fn_safety(self, module, input, output):  # 在前向传播时注入干预向量
        """
        Hook function to add a vector to the output of the model.
        """
        """
        output 是一个 tuple → output[0] 是主要的输出张量 → tmp = output[0] 表示提取出这个张量
        """
        tmp = output[0]
        toxic_concept_detection = self.detection(
            tmp,  # embeddings1
            self.mean_harmless_embedding[self.selected_safety_layer_index - 1],
            self.base_safety_vector,
            self.threshold_safety,
        )
        # # Manipulating only the first few tokens can improve readability
        # 只对前几个 token 的 embedding 做干预，而不是整句话，会让模型输出更自然、可读性更好
        # if toxic_concept_detection and self.count >= 0:
        if toxic_concept_detection:  # any() ？
            tmp = tmp + self.delta_safety * self.base_safety_vector.to(
                torch.float16
            ).to(tmp.device)
        new_output = (tmp, *output[1:])
        return new_output

    def hook_fn_jailbreak(self, module, input, output):
        """
        Hook function to add a vector to the output of the model.
        """
        tmp = output[0]
        jailbreak_concept_detection = self.detection(
            tmp,
            self.mean_harmful_embedding[self.selected_jailbreak_layer_index - 1],
            self.base_jailbreak_vector,
            self.threshold_jailbreak,
        )
        # # Manipulating only the first few tokens can improve readability
        # if jailbreak_concept_detection and self.count >= 0:
        if jailbreak_concept_detection:
            tmp = tmp + self.delta_jailbreak * self.base_jailbreak_vector.to(
                torch.float16
            ).to(tmp.device)
        new_output = (tmp, *output[1:])
        return new_output

    # 将两个 forward hook 注册到模型的指定 transformer 层，使得模型在前向传播这些层时，会调用你自己写的干预函数（hook_fn_safety, hook_fn_jailbreak）
    def register_hooks(self):
        hook_safety = self.model.model.layers[
            self.selected_safety_layer_index - 1
        ].register_forward_hook(self.hook_fn_safety)
        self.hooks.append(hook_safety)
        hook_jailbreak = self.model.model.layers[
            self.selected_jailbreak_layer_index - 1
        ].register_forward_hook(self.hook_fn_jailbreak)
        self.hooks.append(hook_jailbreak)

    # 模型在测试多个攻击时，需要更换 base vector 或 delta 值（各种 jailbreak type 训练出的 base vector 和 delta）
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def prepare_mitigation_data(model_name):
    """
    Prepare model response for mitigation evaluation
    """
    # Load model
    model, tokenizer = load_model(model_name, model_paths)

    # Load data 加载原始 jailbreak prompt 数据
    jailbreaks = ["gcg", "puzzler", "saa", "autodan", "drattack", "pair", "ijp", "base64", "zulu",]
    jailbreak_prompts = {}
    goals = {}
    for jailbreak in jailbreaks:
        path = f"data/mitigation/{jailbreak}/{model_name}.json"
        with open(path) as f:
            data = json.load(f)
        jailbreak_prompts[jailbreak] = [item["jailbreak"] for item in data]
        goals[jailbreak] = [item["goal"] for item in data]

    # 加载通用向量（所有攻击共享）
    mean_harmless_embedding = torch.load(
        f"./vectors/{model_name}/mean_harmless_embedding.pt"
    )
    mean_harmful_embedding = torch.load(
        f"./vectors/{model_name}/mean_harmful_embedding.pt"
    )
    base_safety_vector = torch.load(
        f"./vectors/{model_name}/calibration_safety_vector.pt"
    )

    # 加载每种攻击的“干预层索引”（critical layers）
    [
        selected_safety_layer_index,
        selected_jailbreak_layer_index_gcg,
        selected_jailbreak_layer_index_puzzler,
        selected_jailbreak_layer_index_saa,
        selected_jailbreak_layer_index_autodan,
        selected_jailbreak_layer_index_drattack,
        selected_jailbreak_layer_index_pair,
        selected_jailbreak_layer_index_ijp,
        selected_jailbreak_layer_index_base64,
        selected_jailbreak_layer_index_zulu,
    ] = torch.load(f"./vectors/{model_name}/layer_indexs.pt")
    selected_jailbreak_layer_indexs = [
        selected_jailbreak_layer_index_gcg,
        selected_jailbreak_layer_index_puzzler,
        selected_jailbreak_layer_index_saa,
        selected_jailbreak_layer_index_autodan,
        selected_jailbreak_layer_index_drattack,
        selected_jailbreak_layer_index_pair,
        selected_jailbreak_layer_index_ijp,
        selected_jailbreak_layer_index_base64,
        selected_jailbreak_layer_index_zulu,
    ]
    print("Selected safety layer index: {}".format(selected_safety_layer_index))
    print(
        "Selected gcg jailbreak layer indexs: {}".format(
            selected_jailbreak_layer_indexs
        )
    )

    for i, jailbreak in enumerate(jailbreaks):
        print("Prepara response for {}".format(jailbreak))
        # load vectors and thresholds
        base_jailbreak_vector = torch.load(
            f"./vectors/{model_name}/calibration_jailbreak_vector_{jailbreak}.pt",
            weights_only=False
        )
        threshold_safety = torch.load(
            f"./vectors/{model_name}/thershold_safety_{jailbreak}.pt",
            weights_only=False
        )
        threshold_jailbreak = torch.load(
            f"./vectors/{model_name}/thershold_jailbreak_{jailbreak}.pt",
            weights_only=False
        )
        delta_safety = torch.load(
            f"./vectors/{model_name}/delta_safety.pt",
            weights_only=False
        )
        delta_jailbreak = torch.load(
            f"./vectors/{model_name}/delta_jailbreak_{jailbreak}.pt",
            weights_only=False
        )
        selected_jailbreak_layer_index = selected_jailbreak_layer_indexs[i]

        # Create JBShieldM object
        jbshield_m = JBShieldM(
            model,
            tokenizer,
            mean_harmless_embedding,
            mean_harmful_embedding,
            base_safety_vector,
            base_jailbreak_vector,
            threshold_safety,
            threshold_jailbreak,
            delta_safety,
            delta_jailbreak,
            selected_safety_layer_index,
            selected_jailbreak_layer_index,
        )
        jbshield_m.register_hooks()  # 注册 hook

        # Generate outputs
        outputs = []
        for prompt in tqdm(jailbreak_prompts[jailbreak]):
            outputs.append(
                get_output_prompt(
                    jbshield_m.model,
                    model_name,
                    jbshield_m.tokenizer,
                    prompt,
                    max_new_tokens=50,
                )
            )
            gc.collect()
            torch.cuda.empty_cache()  # 强制清空缓存释放显存
            print(outputs[-1])

        # Save jailbreaks and outputs in json files
        with open(f"./data/mitigation/{jailbreak}/{model_name}.json", "w") as f:
            json.dump(
                [
                    {
                        "goal": goals[jailbreak][i],
                        "jailbreak": jailbreak_prompts[jailbreak][i],
                        "response": outputs[i],
                    }
                    for i in range(len(jailbreak_prompts[jailbreak]))
                ],
                f,
            )
        
        jbshield_m.remove_hooks()  # 避免当前攻击类型的 hook 干扰下一个攻击的处理


def evaluate_mitigation():
    # Load judge model 加载一个“打分用”的评估模型 mistral-sorry-bench
    print("Loading judge model...")
    judge_model, judge_tokenizer = load_model("mistral-sorry-bench", model_paths)

    model_names = [
        "mistral",
        "llama-2",
        "llama-3",
        "vicuna-7b",
        "vicuna-13b",

        "deepseek-r1",
        "qwen-2.5"
    ]
    for model_name in model_names:
        # Load data
        jailbreaks = ["ijp", "gcg", "saa", "autodan", "pair", "drattack", "puzzler", "zulu", "base64",]
        jailbreak_prompts = {}
        responses = {}
        for jailbreak in jailbreaks:
            path = f"data/mitigation/{jailbreak}/{model_name}.json"
            with open(path) as f:
                data = json.load(f)
            jailbreak_prompts[jailbreak] = [item["jailbreak"] for item in data]
            responses[jailbreak] = [item["response"] for item in data]

        for jailbreak in jailbreaks:
            print("Evaluating mitigation for {} on {}".format(jailbreak, model_name))
            results = []
            for prompt, response in tqdm(
                zip(jailbreak_prompts[jailbreak], responses[jailbreak])
            ):
                # Get judge model prediction
                results.append(
                    get_judge_scores(
                        model_name, judge_model, judge_tokenizer, prompt, response
                    )
                )
            print("Attack success rate: {}%".format(np.mean(results)))


if __name__ == "__main__":
    # Prepare responses for test
    # for model_name in ["mistral", "llama-2", "llama-3", "vicuna-7b", "vicuna-13b"]:
    # for model_name in ["deepseek-r1", "qwen-2.5"]:
    #   prepare_mitigation_data(model_name)
    
    # Run this script to evaluate the mitigation performance of JBShield-M on 5 llms
    evaluate_mitigation()

# An example for run this script to evaluate the mitigation on 5 llms
# python mitigation.py > ./logs/JBShield-M.log
