import argparse
from tqdm import tqdm
from typing import Dict
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import gc
from openai import OpenAI, APIError, Timeout, APIConnectionError
import torch
from datetime import datetime
from utils import load_file, save_file_jsonl, metric_max_over_ground_truths,\
exact_match_score, match, qa_f1_score, save_file_json, \
num_tokens_from_string, check_string_exist, postprocess_output, \
PROMPT_DICT, MODEL_PROMPT_KEY_MAPPING, fewshot_examples



def call_openai_api(openai_client: OpenAI, prompt: [Dict], model="gpt-3.5-turbo-0125", temperature=0.0, top_p=0.95, max_tokens=50, chat_completions=True):
    # https://platform.openai.com/docs/guides/text-generation
    if chat_completions:
    # Chat completions API
        try:
            response = openai_client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant to answer questions."},
                    {"role": "user", "content": prompt}
                ],
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f"\nERROR: {e} =========")
            return "ERROR: API error outputs"
    else:
        # Completions API
        try:
            response = openai_client.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                prompt=prompt,
            )
            result = response.choices[0].text
        except (APIError, Timeout, APIConnectionError):
            result = "ERROR: API error outputs"

    return result


def call_model(prompts, model, temperature=0.8, top_p=0.95, max_new_tokens=50):
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    preds = model.generate(prompts, sampling_params)

    preds = [pred.outputs[0].text for pred in preds]

    return preds


def get_prompt_name(item, args):
    """ Infer prompt key from the model name and retrieval mode """

    prompt_key = MODEL_PROMPT_KEY_MAPPING[args.model_name]

    if "do_retrieval" in item:
        if item["do_retrieval"] == 1:
            prompt_name = f"{prompt_key}_retrieval"
        elif item["do_retrieval"] == 0:
            prompt_name = f"{prompt_key}_no_retrieval"
    elif args.retrieval_mode == "adaptive_retrieval":
            if args.prompt_method == "TAARE":
                prompt_name =  f"{prompt_key}_adaptive_retrieval_TAARE"
            elif args.prompt_method == "vanilla":
                prompt_name = f"{prompt_key}_adaptive_retrieval"

    return prompt_name


def calculate_tokens(item):
    q_token_num = num_tokens_from_string(item["question"])
    item["q_token_num"] = q_token_num
    context_token_num = num_tokens_from_string(item["evidence"])
    item["context_token_num"] = context_token_num


def format_context(item, args):
    if "do_retrieval" in item:
        if item["do_retrieval"] == 1:
            retrieval_result = item["context"][:args.doc_top_n]

            if isinstance(retrieval_result[0], str):
                evidences = [f"[{i+1}] {context}" for i, context in enumerate(retrieval_result)] 
            else:
                evidences = [f"[{i+1}] {context['title'].strip() if 'title' in context else ''}\n{context['text'].strip() if 'text' in context else ''}" for i, context in enumerate(retrieval_result)] 
            item["evidence"] = "\n".join(evidences)

            calculate_tokens(item)

        elif item["do_retrieval"] == 0:
            item["evidence"] = ""
            calculate_tokens(item)

    elif args.retrieval_mode == "adaptive_retrieval":
            item["evidence"] = ""
            calculate_tokens(item)


def run_batch_inferece(args, input_data, model=None, isOpenAI=None, 
                       openai_client=None, chat_completions=None):
    
    for idx in tqdm(range(len(input_data))):

        item = input_data[idx]
        item["today"] = datetime.today().strftime('%Y-%m-%d')
        item["fewshot_examples"] = fewshot_examples

        prompt_name = get_prompt_name(item, args)
        format_context(item, args)
        formatted_prompt = PROMPT_DICT[prompt_name].format_map(item)
        
        # print(f"============= prompt =================")
        # print(f"{formatted_prompt}\n")

        if isOpenAI:
            text = call_openai_api(
                    openai_client=openai_client, 
                    prompt=formatted_prompt, 
                    model=args.model_name, 
                    temperature=args.temperature, 
                    top_p=args.top_p,
                    max_tokens=args.max_tokens, 
                    chat_completions=chat_completions
                )
            text = postprocess_output(text, formatted_prompt)
        else:
            predictions = call_model([formatted_prompt], model=model, 
                                    temperature=args.temperature, 
                                    top_p=args.top_p, 
                                    max_new_tokens=args.max_tokens
                                    )
            
            text = predictions[0]
            text = postprocess_output(text, formatted_prompt)

        if "do_retrieval" not in item:
            item["do_retrieve_pred"] = text
            item["do_retrieval"] = check_string_exist(text)
        else:
            item["model_prediction"] = text

    return input_data


def load_model(args):
    model = LLM(model=args.model_name, 
                tensor_parallel_size=args.world_size,
                trust_remote_code=True,
                seed=args.seed
                )

    return model


def load_openai(args):
    with open(args.openai_config_path) as f:
        openai_api_key = f.read()

    openai_client = OpenAI(api_key=openai_api_key)

    if "gpt-4" in args.model_name or "gpt-3.5" in args.model_name:
        chat_completions = True
    else:
        chat_completions = False
    
    return openai_client, chat_completions


def main(args):

    isOpenAI = True if args.model_name in \
        ["text-davinci-003", "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-0125-preview"] else False

    ########### load model ###########
    openai_client, chat_completions, model = None, None, None
    if isOpenAI:
        openai_client, chat_completions = load_openai(args)
    else:
        model = load_model(args)
    
    ########### load dataset ###########
    input_data = load_file(args.input_data_path)
    print(f"# total input_data: {len(input_data)}")
    print(f"{input_data[0]}")

    if args.data_source != "retrievalqa":
        input_data = [item for item in input_data if item["data_source"] == args.data_source]
    if args.limit_input > 0:
        input_data = input_data[:args.limit_input]
    print(f"\nselected data #: {len(input_data)}, data source: {args.data_source}")

    ########### prepare retrieval context ###########
    if args.retrieval_mode == "always_retrieval":
        for item in input_data:
            item["do_retrieval"] = 1
    elif args.retrieval_mode == "no_retrieval":
        for item in input_data:
            item["do_retrieval"] = 0
    elif args.retrieval_mode == "adaptive_retrieval":
        # prompt model to decide whether to retrieve
        input_data = run_batch_inferece(
                    model=model, 
                    input_data=input_data, 
                    isOpenAI=isOpenAI, 
                    openai_client=openai_client, 
                    chat_completions=chat_completions, 
                    args=args
                )
        
        # reload model before inference
        if not isOpenAI:
            # Delete the llm object and free the memory
            destroy_model_parallel()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()
            print("Successfully delete the llm pipeline and free the GPU memory!")

            model = load_model(args)

    
    count = sum([item["do_retrieval"] for item in input_data])
    print(f"\n\n ========================== total retrieval: {count} ========================== \n")

    ########### Run prediction ###########
    input_data = run_batch_inferece(
                    model=model, 
                    input_data=input_data, 
                    isOpenAI=isOpenAI, 
                    openai_client=openai_client, 
                    chat_completions=chat_completions, 
                    args=args
                )
    
    ########### Calculate metrics ###########
    em_total, f1_total, acc_total, match_total = 0, 0, 0, 0
    for item in input_data:
        pred = item["model_prediction"]
        gts = item["ground_truth"]

        em_score = 1.0 if metric_max_over_ground_truths(exact_match_score, pred, gts) else 0.0
        accuracy_score = 1.0 if gts[0] in pred else 0.0
        match_score = match(pred, gts)  # loose match
        f1_score = metric_max_over_ground_truths(qa_f1_score, pred, gts)

        item["em_score"] = em_score
        item["accuracy_score"] = accuracy_score
        item["match_score"] = match_score
        item["f1_score"] = f1_score

        em_total += em_score
        f1_total += f1_score
        acc_total += accuracy_score
        match_total += match_score

    total_q_tokens = sum([item["q_token_num"] for item in input_data])
    total_context_tokens = sum([item["context_token_num"] for item in input_data])
    estimate_q_cost = total_q_tokens/1000*0.0005
    estimate_context_cost = total_context_tokens/1000*0.0005
    estimate_no_retrieval_cost = estimate_q_cost
    estimate_always_retrieval_cost = estimate_q_cost + estimate_context_cost

    saved_cost_rate = 1 - estimate_q_cost / (estimate_q_cost + estimate_context_cost)
    total_retrieval =  sum([item["do_retrieval"] for item in input_data])

    print(f"\n ======= estimate no retrieval (q) API cost: {estimate_no_retrieval_cost}, total tokens #: {total_q_tokens} ================")
    print(f" ======= estimate always retrieval (q+context) API cost: {estimate_always_retrieval_cost}, total tokens #: {total_context_tokens+total_q_tokens} ================")
    print(f" ======= total retrieval: [{total_retrieval}/{len(input_data)}] ================\n")


    total_score = {
        "data_source": args.data_source,
        "total_data_count": len(input_data), 
        "retrieval_frequency": total_retrieval,
        "retrieval_rate": round(total_retrieval/len(input_data)*100, 1),
        "match_score": round(match_total/len(input_data)*100, 1), 
        "f1_score": round(f1_total/len(input_data)*100, 1), 
        "em_score": round(em_total/len(input_data)*100, 1), 
        "accuracy_score": round(acc_total/len(input_data)*100, 1), 
        "match_total": match_total,
        "f1_total": f1_total,
        "em_total": em_total,
        "accuracy_total": acc_total,
        "total_q_tokens": total_q_tokens,
        "total_context_tokens": total_context_tokens,
        "total_no_retrieval_tokens": total_q_tokens,
        "total_always_retrieval_tokens": total_context_tokens,
        "estimate_no_retrieval_cost": estimate_no_retrieval_cost,
        "estimate_always_retrieval_cost": estimate_always_retrieval_cost,
        "saved_cost_rate": saved_cost_rate,
        'args': vars(args)
    }

    print()
    print(total_score)

    # remove 'evidence' before saving results
    for item in input_data:
        if "evidence" in item:
            del item["evidence"]
        if "today" in item:
            del item["today"]
        if "fewshot_examples" in item:
            del item["fewshot_examples"]

    save_file_json(total_score, args.output_score_path)
    save_file_jsonl(input_data, args.output_prediction_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--openai_config_path', type=str, default=None, help='OpenAI Config file path')
    parser.add_argument('--data_source', type=str, default="retrievalqa")
    parser.add_argument('--retrieval_mode', type=str, default="no_retrieval")
    parser.add_argument('--input_data_path', type=str, default=None, help='Input data path')
    parser.add_argument('--output_score_path', type=str, default=None, help='Output json file path')
    parser.add_argument('--output_prediction_path', type=str, default=None, help='Output jsonl file path')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-0125', help='OpenAI model name')
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--doc_top_n', type=int, default=5)
    parser.add_argument('--limit_input', type=int, default=0)
    parser.add_argument('--prompt_method', type=str, default="vanilla")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")

    args = parser.parse_args()

    main(args)

