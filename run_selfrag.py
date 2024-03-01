"""
Code adapted from Self-RAG: https://github.com/AkariAsai/self-rag
"""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import random
import torch
import numpy as np
from tqdm import tqdm
import argparse
import re
from tqdm import tqdm
from utils import load_file, save_file_jsonl, metric_max_over_ground_truths,\
exact_match_score, match, qa_f1_score, save_file_json, \
num_tokens_from_string, PROMPT_DICT,\
load_special_tokens, control_tokens



def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, " ")

    if "</s>" in answer:
        answer = answer.replace("</s>", " ")
    if "\n" in answer:
        answer = answer.replace("\n", " ")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", " ")

    # add space between sentences
    sentences = re.split('(?<=[\.\?\!])\s*', answer)
    answer = ' '.join(sentences) 

    return answer


def call_model_rerank_w_scores_batch(prompt, evidences, model, max_tokens=15,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, 
                                     retrieval_mode="adaptive_retrieval", closed=False,
                                     temperature=0.0, top_p=1.0
                                     ):
    results = {}
    if retrieval_mode != "always_retrieval":
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, logprobs=32016)
        preds = model.generate([prompt], sampling_params)
        pred_token_ids = preds[0].outputs[0].token_ids
        pred_text = preds[0].outputs[0].text
        pred_log_probs = preds[0].outputs[0].logprobs
        results["no_retrieval"] = pred_text

        do_retrieve_pred = pred_text

    # save relevance token scores
    if retrieval_mode == "always_retrieval":
        do_retrieval = True

    elif retrieval_mode == "no_retrieval":
        do_retrieval = False

    else:
        if threshold is not None:
            score_dict = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict[tok] = float(prob)
            do_retrieval = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieval = "[Retrieval]" in pred_text
    
    if do_retrieval is True:
        if isinstance(evidences[0], str):
            evidence_augmented_inputs = [f"{prompt}[Retrieval]<paragraph>{context}</paragraph>" for i, context in enumerate(evidences)] 
        else:
            evidence_augmented_inputs = [f"{prompt}[Retrieval]<paragraph>{context['title'].strip() if 'title' in context else ''}\n{context['text'].strip() if 'text' in context else ''}</paragraph>" for i, context in enumerate(evidences)] 

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, logprobs=5000)
        preds = model.generate(evidence_augmented_inputs, sampling_params)

        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / \
                max(len(pred.outputs[0].token_ids), 1)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            if grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        grd_score_dict[p_idx][token] = np.exp(float(prob))

            if ut_tokens is not None:
                utility_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in ut_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(float(prob))

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values())))

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum(
                    [ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
            else:
                utility_score = 0.0

            if use_seqscore is True:
                final_score = np.exp(seq_score) + w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score

            overall_scores[p_idx] = {"final_score": final_score,
                                     "relevance_score": relevance_score,
                                     "ground_score": ground_score,
                                     "utility_score": utility_score,
                                     "relevance_score_dict": relevance_score_dict,
                                     "grd_score_dict": grd_score_dict,
                                     "ut_score_dict": utility_score}
            results["retrieval_{}".format(p_idx)] = {
                "pred": pred_text, "score": final_score, "ctx": evidences[p_idx]}

    else:
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params)
        pred = preds[0].outputs[0].text
    
    # Aggregating answers
    if len(results) == 1:
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieval, do_retrieve_pred
    else:
        answer2score = {}
        if closed is True:
            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(
                answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
        else:
            path2score = {key: item["score"] for key,
                          item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(),
                               key=lambda x: x[1], reverse=True)[0][0]
            best_option = results[best_path]["pred"]

            do_retrieve_pred = best_option
        return best_option, results, do_retrieval, do_retrieve_pred


def process_data_evidences(demonstration, top_n):
    # ctx_key = "ctxs" if "ctxs" in demonstration else "top_contexts"
    ctx_key = "context"
    prompt = PROMPT_DICT["prompt_no_input"].format_map(demonstration)
    evidences = demonstration[ctx_key][:top_n]

    return prompt, evidences


def preprocess_input_data(dataset, task=None):
    new_data = []
    # if task in TASK_INST:
    #     instruction = TASK_INST[task]
    # else:
    #     instruction = None

    instruction = None
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        else:
            prompt = instruction + "\n\n## Input:\n\n" + \
                item["question"] if instruction is not None else item["question"]
            item["instruction"] = prompt
        new_data.append(item)

    return new_data


def extract_folder_path(path: str) -> str:
	"""Extract the parent folder from the path
	"""
	try:
		idx = path.rfind('/')
		parent_path = path[:idx]
		return parent_path
	except OSError:
		print(f"The provided path is not valid: {path}")
		raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_data_path', type=str)
    parser.add_argument('--output_score_path', type=str, default=None, help='Output json file path')
    parser.add_argument('--output_prediction_path', type=str, default=None, help='Output jsonl file path')
    parser.add_argument('--limit_input', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--data_source', type=str, default="retrievalqa")
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_tokens', type=int, default=50)
    parser.add_argument("--doc_top_n", type=int, default=5,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    # Decoding hyperparams
    parser.add_argument('--threshold', type=float,
                        default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true",
                        help="use ground score")
    parser.add_argument(
        "--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width",  type=int,
                        default=2, help="beam search width")
    parser.add_argument("--max_depth",  type=int,
                        default=2, help="tree depth width")
    parser.add_argument("--w_rel",  type=float, default=1.0,
                        help="reward weight for document relevance")
    parser.add_argument("--w_sup",  type=float, default=1.0,
                        help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use",  type=float, default=0.5,
                        help="reward weight for overall completeness / utility.")
    parser.add_argument('--retrieval_mode', type=str, help="mode to control retrieval.",
                        default="default", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieval'],)
    parser.add_argument('--metric', type=str, help="metric to be used during evaluation")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ########### load dataset ###########
    input_data = load_file(args.input_data_path)
    print(f"# total input_data: {len(input_data)}")

    if args.data_source != "retrievalqa":
        input_data = [item for item in input_data if item["data_source"] == args.data_source]
    if args.limit_input > 0:
        input_data = input_data[:args.limit_input]
    print(f"\nselected data #: {len(input_data)}, data source: {args.data_source}")
    print(input_data[0])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if args.dtype is not None:
        model = LLM(model=args.model_name, 
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)
    else:
        model = LLM(model=args.model_name, 
                    tensor_parallel_size=args.world_size,)

    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    def generate(prompt, evidences, max_tokens):
        # return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_tokens=max_tokens,
        #                                         rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
        #                                         threshold=args.threshold, max_depth=args.max_depth, use_seqscore=args.use_seqscore,
        #                                         w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, mode=args.mode, closed=args.task in ["fever", "arc_c"])
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_tokens=max_tokens,
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                threshold=args.threshold, use_seqscore=args.use_seqscore,
                                                w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use, 
                                                retrieval_mode=args.retrieval_mode, closed=args.task in ["fever", "arc_c"],
                                                temperature=args.temperature, top_p=args.top_p
                                                )


    preds = []
    prompts = []
    metric_results = []
    all_results = []
    count = 0
    total_q_tokens = 0
    total_context_tokens = 0
    for i, row in tqdm(enumerate(input_data)):
        results = {}

        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        _, evidences = process_data_evidences(row, top_n=args.doc_top_n)

        pred, results, do_retrieval, do_retrieve_pred = generate(
            prompt, evidences, max_tokens=args.max_tokens,)
        
        q_tokens = num_tokens_from_string(row["question"])
        context_tokens = 0
        if do_retrieval:
            if isinstance(evidences[0], str):
                concat_evidences = [f"{context}" for i, context in enumerate(evidences)] 
            else:
                concat_evidences = [f"{context['title'].strip() if 'title' in context else ''}\n{context['text'].strip() if 'text' in context else ''}" for i, context in enumerate(evidences)] 
            total_context = "\n".join(concat_evidences)
            context_tokens = num_tokens_from_string(total_context)
        
        total_q_tokens += q_tokens
        total_context_tokens += context_tokens
        row["q_token_num"] = q_tokens
        row["context_token_num"] = context_tokens
        row["do_retrieve_pred"] = do_retrieve_pred

        if type(pred) is str and pred[0] == "#" or pred[0] == ":":
            pred = pred[1:]

        pred = postprocess_answer_option_conditioned(pred)
        prompts.append(prompt)
        preds.append(pred)
        all_results.append(results)
        if do_retrieval == 1:
            count += 1

        if args.metric == "match":
            metric_result = match(pred, row["ground_truth"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)

        row["model_prediction"] = pred
        row["do_retrieval"] = do_retrieval


    ########### Calculate metrics ###########
    em_total, f1_total, acc_total, match_total = 0, 0, 0, 0
    # for item in final_results:
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
        'args': vars(args)
    }

    print()
    print(total_score)

    # remove 'evidence'
    for item in input_data:
        if "evidence" in item:
            del item["evidence"]

    save_file_json(total_score, args.output_score_path)
    save_file_jsonl(input_data, args.output_prediction_path)


if __name__ == "__main__":
    main()

