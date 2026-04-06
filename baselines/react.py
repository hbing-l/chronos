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

method = 'react'

retrieved_docs_num = 4
max_steps = 3


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


react_step_json_schema = {
    "title": "ReActStep",
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["search", "final"]},
        "search_query": {"type": "string", "description": "If action=search, non-empty; else empty string."},
        "final_answer": {"type": "string", "description": "If action=final, non-empty; else empty string."}
    },
    "required": ["action", "search_query", "final_answer"],
    "additionalProperties": False
}

def dedup_docs(existing_docs, new_docs):
    seen = set()
    merged = []
    for d in existing_docs + new_docs:
        key = d.page_content
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
    return merged


def react_rag_pipeline(question):
    gathered_docs = []
    steps = []

    initial_docs = retriever.invoke(question)
    gathered_docs = dedup_docs(gathered_docs, initial_docs)

    for t in range(max_steps):
        context = "\n".join([d.page_content for d in gathered_docs])

        step_prompt = f"""
# Task
You are answering a question using ONLY retrieved factual statements.
You may either:
- propose a new retrieval query to fetch more facts, OR
- provide the final answer.

# Question
{question}

# Retrieved Facts So Far
{context}

# Instructions
1. You must output JSON with keys: action, search_query, final_answer.
2. Decide action:
   - Choose action="final" ONLY if the retrieved facts explicitly contain the answer.
   - Otherwise choose action="search".

3. If action="search", search_query MUST be a short keyword query (NOT a sentence), 3–10 tokens, and MUST include:
   - the main entity in the question (who/what the question is about),
   - the relation type (e.g., held_by / head coach / CEO / capital / founded),
   - any time constraint if mentioned (year/month/date).
   Set final_answer to "".

4. If action="final", final_answer must be ONLY the answer (a word or noun phrase; if multiple, comma-separated in chronological order).
   Set search_query to "".

5. Do NOT include explanations, reasoning, citations, or extra words.

6. If the question is commonsense and does not involve temporal information, answer it directly without retrieval.

# Output JSON
Return strictly JSON:
{{"action":"search","search_query":"...","final_answer":""}}
or
{{"action":"final","search_query":"","final_answer":"..."}}

Now respond:
""".lstrip()

        if 'claude' in model_name:
            step_raw = get_llm_response(
                prompt=step_prompt,
                model_name=model_name,
                # json_schema=react_step_json_schema
            )
        else:
            step_raw = get_llm_response(
                prompt=step_prompt,
                model_name=model_name,
                json_schema=react_step_json_schema
            )

        try:
            step_obj = json.loads(step_raw)
        except Exception:
            step_obj = {"action": "final", "search_query": "", "final_answer": step_raw.strip()}

        action = (step_obj.get("action") or "").strip()

        if action == "final":
            final_answer = (step_obj.get("final_answer") or "").strip()
            steps.append({"t": t, "decision": step_obj, "retrieved_added": 0})
            process_info = {
                "query": question,
                "retrieved_k": retrieved_docs_num,
                "max_steps": max_steps,
                "steps": steps,
                "retrieved_facts": [d.page_content for d in gathered_docs],
                "retrieved_meta": [d.metadata for d in gathered_docs],
                "final_answer": final_answer,
                "response_raw": step_raw
            }
            return final_answer, process_info

        search_query = (step_obj.get("search_query") or "").strip()
        if not search_query:
            search_query = question

        new_docs = retriever.invoke(search_query)
        before = len(gathered_docs)
        gathered_docs = dedup_docs(gathered_docs, new_docs)
        added = len(gathered_docs) - before

        steps.append({
            "t": t,
            "decision": step_obj,
            "search_query": search_query,
            "retrieved_added": added
        })

        if added == 0:
            break

    final_context = "\n".join([d.page_content for d in gathered_docs])

    final_prompt = f"""
# Task
Answer the question using ONLY the retrieved factual statements.

# Question
{question}

# Retrieved Facts
{final_context}

# Instructions
1. Use ONLY the retrieved facts above. Do not use outside knowledge.
2. Return ONLY the final answer as a word or noun phrase.
3. If multiple answers are required, return a single comma-separated list (chronological order if applicable).
4. Do NOT include explanations, reasoning, citations, or extra words.

Answer:
""".lstrip()

    final_raw = get_llm_response(prompt=final_prompt, model_name=model_name)
    final_answer = final_raw.strip()

    process_info = {
        "query": question,
        "retrieved_k": retrieved_docs_num,
        "max_steps": max_steps,
        "steps": steps,
        "retrieved_facts": [d.page_content for d in gathered_docs],
        "retrieved_meta": [d.metadata for d in gathered_docs],
        "final_answer": final_answer,
        "response_raw": final_raw
    }
    return final_answer, process_info


# flag = True
flag = False

# makeup = True
makeup = False
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
            if makeup and (evaluate_qa_exact_match([pred], [gold]) == 1) and task in ['cs']:
                pred, process_info = react_rag_pipeline(ques)
                process_info["gold"] = gold
                with open(record_path, 'w', encoding='utf-8') as f:
                    json.dump(process_info, f, ensure_ascii=False, indent=2)
        else:
            if flag:
                print(f"Example question for [{task}]: {ques}")
                flag = False
                pred, process_info = '', {'query': ques, 'final_answer': '', 'response_raw': ''}
            else:
                pred, process_info = react_rag_pipeline(ques)
            process_info["gold"] = gold
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(process_info, f, ensure_ascii=False, indent=2)

        predictions.append(pred)

    acc = evaluate_qa_exact_match(predictions, groundtruths)
    print('====================')
    print(f"[{task} QA] Exact Match Accuracy: {acc:.4f}")
    print('====================')