import os
import sys
import json
from tqdm import tqdm

sys.path.append('.')
from llm_api import get_llm_response
from utils import evaluate_qa_exact_match, quad_to_text, parse_date




model_name = 'claude-3-haiku-20240307'
model_id = 'claude'


method = 'direct'

#### data
with open('data/knowledge.jsonl', 'r') as f:
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


#### Pipeline
def direct_qa_pipeline(query):
    prompt = """
# Task
Answer the question.

# Question
{question}

# Instructions
1. Use only your own knowledge (no external tools or retrieval).
2. Return only the final answer as a word or noun phrase.
3. If multiple answers are required, provide them as a single comma-separated list.
4. Do NOT include explanations, reasoning, or any additional words.

Answer:
""".lstrip()

    response = get_llm_response(
        prompt=prompt.format(question=query),
        model_name=model_name
    )
    predicted_answer = response.strip()

    process_info = {
        "query": query,
        "final_answer": predicted_answer,
        "response_raw": response
    }
    return predicted_answer, process_info


#### Main
for task, questions, groundtruths in [
    ('h', h_questions, h_answers),
    ('c1', c1_questions, c1_answers),
    ('c2', c2_questions, c2_answers),
    ('c3', c3_questions, c3_answers),
    ('cs', cs_questions, cs_answers)]:
    
    predictions = []
    output_dir = f'outputs/{method}/{model_id}/{task}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (ques, gold) in enumerate(tqdm(zip(questions, groundtruths), total=len(questions))):

        record_path = os.path.join(output_dir, f'record_{i}.json')

        if os.path.exists(record_path):
            with open(record_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
                pred = record['final_answer']
                process_info = record
        else:
            pred, process_info = direct_qa_pipeline(ques)
            process_info['gold'] = gold
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(process_info, f, ensure_ascii=False, indent=2)

        predictions.append(pred)
        
    
    acc = evaluate_qa_exact_match(predictions, groundtruths)
    print('====================')
    print(f"[{task} QA] Exact Match Accuracy: {acc:.4f}")
    print('====================')

    # break
