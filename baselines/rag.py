import os
import sys
import json
from tqdm import tqdm

sys.path.append('.')
from llm_api import get_llm_response
from utils import evaluate_qa_exact_match, quad_to_text

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore



model_name = 'claude-3-haiku-20240307'
model_id = 'claude'

method = 'rag'

retrieved_docs_num = 4


with open('data/knowledge.jsonl', 'r', encoding='utf-8') as f:
    knowledge_quadruples = [json.loads(line) for line in f]


def load_qa_jsonl(path):
    questions, answers = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            questions.append(obj['question'])
            answers.append(obj['answer'])
    return questions, answers


h_questions, h_answers = load_qa_jsonl('data/h_qa.jsonl')
c1_questions, c1_answers = load_qa_jsonl('data/c1_qa.jsonl')
c2_questions, c2_answers = load_qa_jsonl('data/c2_qa.jsonl')
c3_questions, c3_answers = load_qa_jsonl('data/c3_qa.jsonl')
cs_questions, cs_answers = load_qa_jsonl('data/cs_qa.jsonl')


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

docs = []
for i, q in enumerate(knowledge_quadruples):
    text = quad_to_text(q, with_timestamp=True)
    docs.append(Document(page_content=text, metadata={
        "id": i,
        "subject": q.get("subject", ""),
        "relation": q.get("relation", ""),
        "object": q.get("object", ""),
        "timestamp": q.get("timestamp", ""),
        "source": q.get("source", ""),
    }))

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": retrieved_docs_num})


def rag_pipeline(query):
    retrieved = retriever.invoke(query)
    retrieved_text = "\n".join([d.page_content for d in retrieved])

    prompt = f"""
# Task
Answer the question using ONLY the retrieved factual statements.

# Question
{query}

# Retrieved Facts
{retrieved_text}

# Instructions
1. Return ONLY the final answer as a word or noun phrase.
2. If multiple answers are required, return a single comma-separated list (chronological order if applicable).
3. Do NOT include explanations, reasoning, citations, or extra words.

Answer:
""".lstrip()

    response = get_llm_response(prompt=prompt, model_name=model_name)
    predicted_answer = response.strip()

    process_info = {
        "query": query,
        "retrieved_k": retrieved_docs_num,
        "retrieved_facts": [d.page_content for d in retrieved],
        "retrieved_meta": [d.metadata for d in retrieved],
        "final_answer": predicted_answer,
        "response_raw": response
    }
    return predicted_answer, process_info



# flag = True
flag = False
for task, questions, groundtruths in [
    ('h', h_questions, h_answers),
    ('c1', c1_questions, c1_answers),
    ('c2', c2_questions, c2_answers),
    ('c3', c3_questions, c3_answers),
    ('cs', cs_questions, cs_answers),
]:
    predictions = []
    output_dir = f'outputs/{method}/{model_id}/{task}'
    os.makedirs(output_dir, exist_ok=True)

    for i, (ques, gold) in enumerate(tqdm(list(zip(questions, groundtruths)), total=len(questions))):
        record_path = os.path.join(output_dir, f'record_{i}.json')

        if os.path.exists(record_path):
            with open(record_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            pred = record.get("final_answer", "").strip()
        else:
            if flag:
                print(f"Example question for [{task}]: {ques}")
                flag = False
                pred, process_info = '', {'query': ques, 'final_answer': '', 'response_raw': ''}
            else:
                pred, process_info = rag_pipeline(ques)
            process_info["gold"] = gold
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(process_info, f, ensure_ascii=False, indent=2)

        predictions.append(pred)

    acc = evaluate_qa_exact_match(predictions, groundtruths)
    print('====================')
    print(f"[{task} QA] Exact Match Accuracy: {acc:.4f}")
    print('====================')