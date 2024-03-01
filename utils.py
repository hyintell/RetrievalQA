import jsonlines
import json
import re
import pathlib
import string
from collections import Counter
import numpy as np
import tiktoken


MODEL_PROMPT_KEY_MAPPING = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama",
    "microsoft/phi-2": "phi2",
    "meta-llama/Llama-2-7b-chat-hf": "llama",
    "gpt-3.5-turbo-0125": "openai_chat",
    "gpt-4-0125-preview": "openai_chat",
}


fewshot_examples = (
    "Question: what is the water boiling point?\nAnswer: [No].\n\n"
    "Question: As of today, name the 5 NFL teams that have never actually played in a super bowl?\nAnswer: [Yes].\n\n"
    "Question: What is the capital of France?\nAnswer: [No].\n\n"
    "Question: What is Walter de la Pole's occupation?\nAnswer: [Yes].\n\n"
)


PROMPT_DICT = {
    "openai_chat_adaptive_retrieval_TAARE": (
        "Today is {today}. Given a question, determine whether you need to retrieve external resources, such as real-time search engines, Wikipedia, or databases, to answer the question correctly. Only answer \"[Yes]\" or \"[No]\".\n\n"
        "Here are some examples:\n\n"
        "{fewshot_examples}\n\n"
        "Question: {question}\n" 
        "Answer:"
    ),
    "openai_chat_adaptive_retrieval": (
        "Given a question, determine whether you need to retrieve external resources, such as real-time search engines, Wikipedia, or databases, to answer the question correctly. Only answer \"[Yes]\" or \"[No]\".\n\n"
        "Question: {question}\n" 
        "Answer:"
    ),
    "openai_chat_no_retrieval": (
        "Please use your own knowledge to answer the questions. Only include the answer in your response and try to be concise. If you do not know the answer, just say \"I don't know\".\n\nQuestion: {question}\nAnswer:"
    ),
    "openai_chat_retrieval": (
        "Please answer the question based on the provided context. Only include the answer in your response and try to be concise. If you do not know the answer, just say \"I don't know\".\n\n"
        "Paragraph:\n{evidence}\n\n"
        "Question: {question}\n"
        "Answer:" 
    ),
    "llama_adaptive_retrieval_TAARE": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful assistant to answer questions.\n"
        "<</SYS>>\n\n"
        "Today is {today}. Given a question, determine whether you need to retrieve external resources, such as real-time search engines, Wikipedia, or databases, to answer the question correctly. Only answer \"[Yes]\" or \"[No]\".\n\n"
        "Here are some examples:\n\n"
        "{fewshot_examples}\n\n"
        "Question: {question}\n" 
        "Answer:[/INST]"
    ),
    "llama_adaptive_retrieval": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful assistant to answer questions.\n"
        "<</SYS>>\n\n"
        "Given a question, determine whether you need to retrieve external resources, such as real-time search engines, Wikipedia, or databases, to answer the question correctly. Only answer \"[Yes]\" or \"[No]\".\n\n"
        "Question: {question}\n" 
        "Answer:[/INST]"
    ),
    "llama_no_retrieval": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful assistant to answer questions.\n"
        "<</SYS>>\n\n"
        "Please use your own knowledge to answer the questions. Only include the answer in your response and try to be concise. If you do not know the answer, just say \"I don't know\".\n\nQuestion: {question}\nAnswer:[/INST]"
    ),
    "llama_retrieval": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful assistant to answer questions.\n"
        "<</SYS>>\n\n"
        "Please answer the question based on the provided context. Only include the answer in your response and try to be concise. If you do not know the answer, just say \"I don't know\".\n\n"
        "Paragraph:\n{evidence}\n\n"
        "Question: {question}\n"
        "Answer:[/INST]" 
    ),
    "phi2_adaptive_retrieval_TAARE": (
        "Instruct: Today is {today}. Given a question, determine whether you need to retrieve external resources, such as real-time search engines, Wikipedia, or databases, to answer the question correctly. Only answer \"[Yes]\" or \"[No]\".\n\nHere are some examples:\n\n{fewshot_examples}\n\nQuestion: {question}\n"
        "Output:"
    ),
    "phi2_adaptive_retrieval": (
        "Instruct: Given a question, determine whether you need to retrieve external resources, such as real-time search engines, Wikipedia, or databases, to answer the question correctly. Only answer \"[Yes]\" or \"[No]\".\n\nQuestion: {question}\n"
        "Output:"
    ),
    "phi2_no_retrieval": (
        "Instruct: Please use your own knowledge to answer the questions. Only include the answer in your response and try to be concise. If you do not know the answer, just say \"I don't know\".\n\nQuestion: {question}\n"
        "Output:"
    ),
    "phi2_retrieval": (
        "Instruct: Please answer the question based on the provided context. Only include the answer in your response and try to be concise. If you do not know the answer, just say \"I don't know\".\n\nParagraph:\n{evidence}\n\nQuestion: {question}\n"
        "Output:"
    ),
    "tinyllama_adaptive_retrieval_TAARE": (
        "<|system|>\n"
        "You are a helpful assistant to answer questions.</s>\n"
        "<|user|>\n"
        "Today is {today}. Given a question, determine whether you need to retrieve external resources, such as real-time search engines, Wikipedia, or databases, to answer the question correctly. Only answer \"[Yes]\" or \"[No]\".\n\nHere are some examples:\n\n{fewshot_examples}\n\nQuestion: {question}\nAnswer:</s>\n"
        "<|assistant|>"
    ),
    "tinyllama_adaptive_retrieval": (
        "<|system|>\n"
        "You are a helpful assistant to answer questions.</s>\n"
        "<|user|>\n"
        "Given a question, determine whether you need to retrieve external resources, such as real-time search engines, Wikipedia, or databases, to answer the question correctly. Only answer \"[Yes]\" or \"[No]\".\n\nQuestion: {question}\nAnswer:</s>\n"
        "<|assistant|>"
    ),
    "tinyllama_no_retrieval": (
        "<|system|>\n"
        "You are a helpful assistant to answer questions.</s>\n"
        "<|user|>\n"
        "Please use your own knowledge to answer the questions. Only include the answer in your response and try to be concise. If you do not know the answer, just say \"I don't know\".\n\nQuestion: {question}\nAnswer:</s>\n"
        "<|assistant|>"
    ),
    "tinyllama_retrieval": (
        "<|system|>\n"
        "You are a helpful assistant to answer questions.</s>\n"
        "<|user|>\n"
        "Please answer the question based on the provided context. Only include the answer in your response and try to be concise. If you do not know the answer, just say \"I don't know\".\n\nParagraph:\n{evidence}\n\nQuestion: {question}\nAnswer:</s>\n"
        "<|assistant|>"
    ),
    "always_retrieval": (
        "Instruct: With this context\n\n{evidence}\n\nQuestion: {question}\nOutput:"
    ),
    "prompt_input": (
        "### Instruction:\n{question}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{question}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{evidence}\n\n### Instruction:\n{question}\n\n### Response:"
    ),
}


# for Self-RAG
rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]",
                          "[Retrieval]", "[Continue to Use Evidence]"]
utility_tokens_names = ["[Utility:1]", "[Utility:2]",
                        "[Utility:3]", "[Utility:4]", "[Utility:5]"]
ground_tokens_names = ["[Fully supported]",
                       "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


def postprocess_output(pred, input=None):
    if input is not None:
        # remove the original input in the output if have
        if input in pred:
            pred = pred[len(input):]

    # for Self-RAG
    pred = pred.replace("</s>", "")

    # for Phi
    if "<|im_start|>" in pred:
        idx = pred.rfind("<|im_start|>assistant")
        if idx != -1:
            pred = pred[idx+len("<|im_start|>assistant"):]  
    pred = pred.replace("<|im_end|>", "")

    # for TinyLlama
    if "<|assistant|>" in pred:
        idx = pred.rfind("<|assistant|>")
        if idx != -1:
            pred = pred[idx+len("<|assistant|>"):]
        pred = pred.replace("<|user|>", "") 
        pred = pred.replace("<|assistant|>", "")
        pred = pred.replace("<|system|>", "")

    # for Llama and Mistral
    if "[/INST]" in pred:
        idx = pred.rfind("Answer:[/INST]")
        if idx != -1:
            pred = pred[idx+len("Answer:[/INST]"):]  

    pred = pred.replace("\n", " ")
    # remove extra white space
    pred = " ".join(pred.split())

    return pred


def check_string_exist(pred):

    match_strings = set(["\"yes\"", "'yes'", "[\"yes\"]", "['yes']", "[yes]", "yes"])
    pred = pred.replace(",", "")
    pred = pred.replace(".", "")
    pred = pred.replace("\"", "")
    pred_words = pred.split(" ")
    pred_words = [word.lower().strip() for word in pred_words]
    pred_words = set(pred_words)

    match_results = list(match_strings.intersection(pred_words))

    if len(match_results) > 0:
        return 1
    else:
        return 0


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def num_tokens_from_string(prompt: str, encoding_name: str = "gpt-3.5-turbo-0125") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(prompt))

    return num_tokens


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


def load_jsonlines(file_path):
    with jsonlines.open(file_path, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(file_path):
    if file_path.endswith(".json"):
        input_data = json.load(open(file_path))
    else:
        input_data = load_jsonlines(file_path)

    return input_data


def save_file_jsonl(data, fp):
    parent_folder = extract_folder_path(fp)
    pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
    print(parent_folder)

    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


def save_file_json(data, fp):
    parent_folder = extract_folder_path(fp)
    pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True)
    print(parent_folder)

    with open(fp, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))
        

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))


def f1(decoded_preds, decoded_labels):
    f1_all = []
    for prediction, answers in zip(decoded_preds, decoded_labels):
        if type(answers) == list:
            if len(answers) == 0:
                return 0
            f1_all.append(np.max([qa_f1_score(prediction, gt)
                          for gt in answers]))
        else:
            f1_all.append(qa_f1_score(prediction, answers))
    return 100 * np.mean(f1_all)


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def match(prediction, ground_truth):
    norm_pred = normalize_answer(prediction)
    for gt in ground_truth:
        norm_gt = normalize_answer(gt)
        if norm_gt in norm_pred:
            return 1
    return 0
