# %% [markdown]
# ### To do evaluation using the hotpotqa script, we need to supply a json file with:
# 
# answer: a dict with QA _id as key -> answer as a string
# 
# sp: a dict with QA _id as key -> list of [title, sent_id]

# %%
import os
import sys
from pathlib import Path
import json
import torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

DATA_DIR = Path("/home/h/SDAIA-Final-Project/qa_system/data/hotpot_dev_fullwiki_v1.json")


from qa_system.retrieval import Retriever # noqa
from qa_system.reranker import Reranker # noqa
from qa_system.llm import LLM # noqa
from qa_system.query_rewriter.rewriter import QueryRewriter # noqa
from qa_system.pipeline import QAPipeline # noqa

# %%
DATASET_PATH = os.path.join(os.getcwd(), "..", "data", "hotpot_dev_fullwiki_v1.json")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset not found at {DATA_DIR}")


def load_jsonl_or_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = f.read().strip()
        if not raw:
            return []
        if "\n" in raw and raw.lstrip().startswith("{"):
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        return json.loads(raw)
    

# %%
retriever = Retriever()
reranker = Reranker()
llm = LLM()
query_rewriter = QueryRewriter()
dataset = load_jsonl_or_json(DATA_DIR)

# %%
import hashlib 
import tqdm



documents = []
documents_ids = []
document_ids_to_sp = {}



duplicate_count = 0
for entry in tqdm.tqdm(dataset[:]):
    question_id = entry['_id']
    question = entry['question']
    paragraphs = entry['context']
    for paragraph in paragraphs:
        title = paragraph[0]
        sentences = paragraph[1]
        for idx, sentence in enumerate(sentences):
            # hash the title and sentence id to create a unique document id
            doc = f"{title}: {sentence}"
            doc_id = hashlib.md5(doc.encode()).hexdigest()

            # check if the document id already exists (hash collision or actual duplicate)
            if doc_id in document_ids_to_sp:
                duplicate_count += 1
                continue
            
            document_ids_to_sp[doc_id] = [title, idx]
            documents.append(doc)
            documents_ids.append(doc_id)


print(f"Total duplicate documents: {duplicate_count}")
assert len(documents) == len(documents_ids)
print(f"Total documents: {len(documents)}")



# %% [markdown]
# ### Eval flow
# 1. iterate over q in the dataset
# 2. pass the q to the qa_pipeline
# 3. save the answer in the answer dict
# 4. get each sp in supporting sentences from the qa_pipeline 
# 5. convert each sp to doc ids and retrieve the [title, sent_id] from the document_ids_to_sp

# %% [markdown]
# ## Eval 
# 

# %%
import sys
import ujson as json
import re
import string
from collections import Counter
import pickle

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


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction: dict, gold: dict):
    

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    return metrics




# %%
def build_no_retriever_pipeline():
    return QAPipeline(retriever=None, reranker=reranker, llm=llm, query_rewriter=query_rewriter)

def build_no_reranker_pipeline():
    return QAPipeline(retriever=retriever, reranker=None, llm=llm, query_rewriter=query_rewriter)

def build_no_query_rewriter_pipeline():
    return QAPipeline(retriever=retriever, reranker=reranker, llm=llm, query_rewriter=None)

def build_full_pipeline():
    return QAPipeline(retriever=retriever, reranker=reranker, llm=llm, query_rewriter=query_rewriter)


# %%
import gc
configuration = {"Direct LLM only": build_no_retriever_pipeline,
                "Retriever only": build_no_reranker_pipeline,
                "Retriever + Reranker": build_no_query_rewriter_pipeline,
                "Retriever + Reranker + Query Rewriter": build_full_pipeline}

MAX_ITER = 1000
results = {}
for config_name, build_pipeline in configuration.items():
    print(f"Evaluating {config_name}...")
    pipeline = build_pipeline()

    answer = {}
    sp = {}
    truth = {}

    
    for entry in tqdm.tqdm(dataset[:MAX_ITER]):
        torch.cuda.empty_cache()

        with torch.no_grad():
            q = entry['question']
            truth[q] = entry['answer']
            pred = pipeline.answer_question(q)
            pred_answer = pred['answer']
            pred_sp = [document_ids_to_sp[d['id']] for d in pred['contexts'][:2]]

        answer[entry['_id']] = pred_answer
        sp[entry['_id']] = pred_sp
    
    prediction = {
        'answer' : answer,
        'sp' : sp
    }

    gold = dataset
    print(f"{config_name} results:")
    results[config_name] = eval(prediction, gold[:MAX_ITER])
    print(results[config_name])
    torch.cuda.empty_cache()
    del pipeline
    gc.collect()


import datetime
with open(f'results-{MAX_ITER}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.json', 'w') as f:
    json.dump(results, f, indent=4)


