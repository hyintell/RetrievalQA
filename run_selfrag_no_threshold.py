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
num_tokens_from_string, load_special_tokens, control_tokens


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


def format_prompt(input, paragraph=None):
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt


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
    assert "selfrag" in args.model_name

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

    model = LLM(args.model_name, dtype="half")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=args.max_tokens, 
        skip_special_tokens=False,
        logprobs=32016
    )

    # Get token ids for reflection tokens.
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0
    total_q_tokens = 0
    total_context_tokens = 0
    # initial prediction to see any [Retrieval] token exists
    for idx in tqdm(range(len(input_data))):
        item = input_data[idx]
        question = item["question"]
        # prepare prompts 
        prompt_no_retrieval = format_prompt(question, paragraph=None)

        results = {}
        preds = model.generate([prompt_no_retrieval], sampling_params)
        pred_text = preds[0].outputs[0].text
        results["no_retrieval"] = pred_text

        item["do_retrieve_pred"] = pred_text

        do_retrieval = 0
        final_prompt = prompt_no_retrieval

        # if check_string_exist(pred_text):
        if "[Retrieval]" in pred_text:
            do_retrieval = 1

            evidences = item["context"]
            # add paragraph in the format and re-generate
            if isinstance(evidences[0], str):
                prompt_w_retrieval = [
                    format_prompt(question, paragraph=evidence) for evidence in evidences
                ]
            else:
                prompt_w_retrieval = []
                for evidence in evidences:
                    concat_evidence = ""
                    if 'title' in evidence:
                        concat_evidence += evidence["title"] + ". "
                    if "text" in evidence:
                        concat_evidence += evidence["text"]

                    prompt_w_retrieval.append(concat_evidence)

            sampling_params = SamplingParams(
                temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, logprobs=5000)
            preds = model.generate(prompt_w_retrieval, sampling_params)

            ### to find the best paragraph option from the top_n_doc
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

                if args.use_seqscore is True:
                    final_score = np.exp(seq_score) + args.w_rel * relevance_score + \
                        args.w_sup * ground_score + args.w_use * utility_score
                else:
                    final_score = args.w_rel * relevance_score + \
                        args.w_sup * ground_score + args.w_use * utility_score

                overall_scores[p_idx] = {"final_score": final_score,
                                        "relevance_score": relevance_score,
                                        "ground_score": ground_score,
                                        "utility_score": utility_score,
                                        "relevance_score_dict": relevance_score_dict,
                                        "grd_score_dict": grd_score_dict,
                                        "ut_score_dict": utility_score}
                results["retrieval_{}".format(p_idx)] = {
                    "pred": pred_text, "score": final_score, "ctx": evidences[p_idx]}
        
        print(f"\n +++++++++++++ do_retrieval = {do_retrieval} +++++++++++++ \n\n")

        # Aggregating answers
        if len(results) == 1:
            postprocessed_pred = postprocess_answer_option_conditioned(pred_text)
            best_predict = postprocessed_pred
        else:
            path2score = {key: item["score"] for key,
                        item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(),
                            key=lambda x: x[1], reverse=True)[0][0]
            best_option = results[best_path]["pred"]

            final_prompt = results[best_path]["ctx"]

            best_predict = best_option

        # calculate token num
        q_tokens = num_tokens_from_string(question)
        context_tokens = 0
        if do_retrieval == 1:
            if isinstance(evidences[0], str):
                concat_evidences = [f"{context}" for i, context in enumerate(evidences)] 
            else:
                concat_evidences = [f"{context['title'].strip() if 'title' in context else ''}\n{context['text'].strip() if 'text' in context else ''}" for i, context in enumerate(evidences)] 
            total_context = "\n".join(concat_evidences)
            context_tokens = num_tokens_from_string(total_context)
        item["q_token_num"] = q_tokens
        item["context_token_num"] = context_tokens
        print(f"\n q_tokens: {q_tokens}, context_tokens: {context_tokens}")
        
        if type(best_predict) is str and best_predict[0] == "#" or best_predict[0] == ":":
            best_predict = best_predict[1:]

        best_predict = postprocess_answer_option_conditioned(best_predict)

        prompts.append(final_prompt)
        preds.append(best_predict)
        all_results.append(results)

        if do_retrieval == 1:
            count += 1

        if args.metric == "match":
            metric_result = match(best_predict, item["ground_truth"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)

        item["model_prediction"] = best_predict
        item["do_retrieval"] = do_retrieval
        
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

