import os
import sys
import json
import math
import re
from tqdm import tqdm
from collections import defaultdict

sys.path.append('.')
from llm_api import get_llm_response
from utils import evaluate_qa_exact_match, parse_date, quad_to_text

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore



model_name = 'claude-3-haiku-20240307'
model_id = 'claude'



print(f"Using model: {model_name} ({model_id})")

method = 'chronos'

retrieved_docs_num = 4



#### data
def load_qa_jsonl(path):
    questions, answers = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            questions.append(obj['question'])
            answers.append(obj['answer'])
    return questions, answers


with open('data/knowledge.jsonl', 'r', encoding='utf-8') as f:
    knowledge_quadruples = [json.loads(line) for line in f]


h_questions, h_answers = load_qa_jsonl('data/h_qa.jsonl')
c1_questions, c1_answers = load_qa_jsonl('data/c1_qa.jsonl')
c2_questions, c2_answers = load_qa_jsonl('data/c2_qa.jsonl')
c3_questions, c3_answers = load_qa_jsonl('data/c3_qa.jsonl')
cs_questions, cs_answers = load_qa_jsonl('data/cs_qa.jsonl')


#### Build vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

docs = []
for i, q in enumerate(knowledge_quadruples):
    text = quad_to_text(q, with_timestamp=False)
    doc = Document(
        page_content=text,
        metadata={
            "id": i,
            "subject": q["subject"],
            "relation": q["relation"],
            "object": q["object"],
            "timestamp": q["timestamp"]
        }
    )
    docs.append(doc)

# Cosine Similarity by default
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": retrieved_docs_num})


def time_aware_retrieve(query, ref_time_start, ref_time_end, vector_store, k_final=5, k_candidates=20, alpha=0.75, tau_days=180):
    # alpha: weight for semantic similarity; (1 - alpha) is the weight for temporal relevance
    # tau_days: time decay scale; smaller values favor more recent timestamps
    # ref_date: reference time (recommended to use the query's time_end or time_start)

    # 1) Retrieve candidates using semantic similarity
    results = vector_store.similarity_search_with_score(
        query, k=k_candidates
    )
    # results: List[(Document, similarity_score)]

    rescored = []
    for doc, sim in results:
        ts = parse_date(doc.metadata["timestamp"])

        # 2) Compute distance to the time interval
        if ts < ref_time_start:
            delta_days = (ref_time_start - ts).days
        elif ts > ref_time_end:
            delta_days = (ts - ref_time_end).days
        else:
            delta_days = 0  # inside interval → max temporal relevance

        # 3) Temporal relevance score
        time_score = math.exp(-delta_days / tau_days)

        # 4) Combined score
        final_score = alpha * float(sim) + (1 - alpha) * time_score

        rescored.append((final_score, doc))

    # 5) Re-rank and return top-k
    rescored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in rescored[:k_final]]
    return top_docs


# event evolution graph
def construct_eeg(knowledge_quadruples, time_start="2024-01-01", time_end="2025-11-30"):
    sorted_knowledge_quadruples = sorted(
        knowledge_quadruples,
        key=lambda q: parse_date(q["timestamp"])
    )

    t_start = parse_date(time_start)
    t_end = parse_date(time_end)

    filtered_knowledge = [
        q for q in sorted_knowledge_quadruples
        if t_start <= parse_date(q["timestamp"]) <= t_end
    ]

    # temporal context
    eeg_temporal = [quad_to_text(f, with_timestamp=True) for f in filtered_knowledge]
    eeg_temporal_context = "[Temporal view]\n" + "\n".join(eeg_temporal) + "\n"

    # subject context
    subject_groups = defaultdict(list)
    for q in knowledge_quadruples:
        subject_groups[q["subject"]].append(q)
    
    eeg_subject_context = ''
    for subject, quads in subject_groups.items():
        quads_sorted = sorted(quads, key=lambda q: parse_date(q["timestamp"]))
        texts = [quad_to_text(f, with_timestamp=True) for f in quads_sorted]
        eeg_subject_context += f"\n[Subject: {subject}]\n" + "\n".join(texts) + "\n"

    eeg_text = eeg_temporal_context + "\n" + eeg_subject_context
    
    return eeg_text, filtered_knowledge


def extract_answer_from_response(response: str) -> str:
    response = response.strip()
    match = re.search(r"(?m)^Answer:\s*(.+)\s*$", response)
    if match:
        predicted_answer = match.group(1).strip()
    else:
        lines = [l.strip() for l in response.splitlines() if l.strip()]
        predicted_answer = lines[-1] if lines else ""
    return predicted_answer


#### Pipeline
def chronos_qa_pipeline(query):
    # 1. query analysis
    query_analysis_prompt_template = """
# Task
Analyze the query and extract:
1. All entities mentioned in the query.
2. The time scope that should be considered for answering.
3. A rewritten version of the query with all time expressions removed (query_without_time).

# Information
Query: {query}

# Instructions
1. Extract all entities into the "entities" field. Keep them clean and concise.
2. All dates must be formatted as YYYY-MM-DD.
3. If the query contains temporal information, set "has_time" to true; otherwise, set it to false.
4. If the query refers to a single day, use the same date for both "time_start" and "time_end".
5. Produce a natural-language query in 'query_without_time' that preserves meaning but removes all temporal expressions.


# Response Format
Return the output strictly in the following JSON format:
{{
    "entities": ["<entity_1>", "<entity_2>", "..."],
    "has_time": true/false,
    "time": ["<time_start>", "<time_end>"],
    "query_without_time": "<rewritten_query_without_any_time_expression>"
}}

Now, provide your response:
"""
    query_analysis_json_schema = {
        "title": "QueryAnalysis",
        "description": "Structured extraction of entities, time scope, and a time-neutral rewritten query.",
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": "A list of clean, concise entities extracted from the query.",
                "items": {
                    "type": "string",
                    "description": "An entity mentioned in the query."
                }
            },
            "has_time": {
                "type": "boolean",
                "description": "Whether the query explicitly contains a temporal constraint."
            },
            "time": {
                "type": "array",
                "description": "A list containing the start and end dates in YYYY-MM-DD format.",
                "items": {
                    "type": "string",
                    "description": "The time_start or time_end value in YYYY-MM-DD format."
                },
                "minItems": 2,
                "maxItems": 2
            },
            "query_without_time": {
                "type": "string",
                "description": "The rewritten version of the original query with all time expressions removed."
            }
        },
        "required": ["entities", "has_time", "time", "query_without_time"],
        "additionalProperties": False
    }

    response = get_llm_response(
        prompt=query_analysis_prompt_template.format(query=query),
        model_name=model_name,
        json_schema=query_analysis_json_schema if 'claude' not in model_id else None
    )
    json_response = json.loads(response)
    entities = json_response['entities']
    has_time = json_response['has_time']

    time_start, time_end = json_response["time"]     
    query_without_time = json_response['query_without_time']

    # 2. retrieve relevant knowledge
    retrieved_knowledge = []
    for entity in entities:
        # docs = retriever.invoke(entity)
        docs = time_aware_retrieve(
            query=entity,
            ref_time_start=parse_date(time_start),
            ref_time_end=parse_date(time_end),
            vector_store=vector_store,
            k_final=retrieved_docs_num
        )
        quad = [{
            'subject': doc.metadata['subject'],
            'relation': doc.metadata['relation'],
            'object': doc.metadata['object'],
            'timestamp': doc.metadata['timestamp'],
        } for doc in docs]
        retrieved_knowledge.extend(quad)

    # 4. reconstruct history
    reconstruct_history_prompt_template = """
# Task
Generate time-aware knowledge triples relevant to the given query.

# Inputs
- Query (without temporal expressions): {query_without_time}
- Time Window: [{time_start}, {time_end}]

# Instructions
1. Use only information that falls within the specified time window.
2. Identify events or facts relevant to the query that occurred during this period.
3. Represent each event as a knowledge triple in the form:
   (subject, relation, object, timestamp).
4. The timestamp should reflect when the event occurred.
5. If multiple valid events exist, return all of them.
6. Do not use external knowledge outside the given time window.

# Output Format
Return a JSON list of knowledge quadruples:

[
  {{
    "subject": "...",
    "relation": "...",
    "object": "...",
    "timestamp": "YYYY-MM-DD"
  }}
]
"""
    response = get_llm_response(
        prompt=reconstruct_history_prompt_template.format(query=query_without_time, time_start=time_start, time_end=time_end),
        model_name=model_name
    )
    history_quad = json.loads(response)

    retrieved_knowledge.extend(history_quad)

    # 5. graph construction
    if len(retrieved_knowledge) == 0:
        evidence_from_EEG, filtered_knowledge = "No relevant contemporary information found.", []
    else:
        evidence_from_EEG, filtered_knowledge = construct_eeg(retrieved_knowledge, time_start, time_end)

    # 6. final QA (CoT-style; parse the last "Answer:" line)
    final_qa_prompt = """
    # Task
    Answer the question step by step using ONLY the provided information.

    # Information
    Question: {question}
    Evidence: {evidence_from_EEG}

    # Instructions
    1. Use ONLY the information in Historical Context and Contemporary Information. Do NOT use outside knowledge.
    2. You may reason freely.
    3. The final line MUST be exactly: Answer: <final answer>
    4. The final answer must be a word or noun phrase.
    5. If multiple answers are required, output a single comma-separated list (noun phrases) in chronological order when applicable.
    6. Do NOT write anything after the Answer line.

    Now, answer:
    """.lstrip()

    response = get_llm_response(
        prompt=final_qa_prompt.format(
            question=query,
            evidence_from_EEG=evidence_from_EEG
        ),
        model_name=model_name
    )

    initial_predicted_answer = extract_answer_from_response(response)
    initial_reasoning = response


    # 7. Event Augmentation (Fallback QA with one-time self-reflection)
    reflect_json_schema = {
        "title": "EvidenceAugmentDecision",
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["retrieve", "augment"]},

            # If action="retrieve"
            "search_query": {
                "type": "string",
                "description": "Non-empty iff action=retrieve; otherwise empty."
            },

            # If action="augment"
            "quads": {
                "type": "array",
                "description": "Non-empty iff action=augment; otherwise empty.",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "relation": {"type": "string"},
                        "object": {"type": "string"},
                        "timestamp": {"type": "string", "description": "YYYY-MM-DD"}
                    },
                    "required": ["subject", "relation", "object", "timestamp"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["action", "search_query", "quads"],
        "additionalProperties": False
    }

    reflect_prompt = f"""
    # Task
    Decide how to improve evidence coverage for answering the question.
    You must choose exactly ONE action:
    - retrieve: propose ONE focused search query to retrieve missing evidence.
    - augment: directly propose additional knowledge quadruples to supplement the Event Evolution Graph (EEG).

    # Question
    {query}

    # Current Evidence (from EEG)
    {evidence_from_EEG}

    # Proposed Answer
    {initial_predicted_answer}

    # Rules
    1. Use ONLY the evidence above to judge whether something is missing. Do NOT use outside knowledge to invent facts.
    2. If you choose action="retrieve":
    - search_query MUST be non-empty (a focused query string).
    - quads MUST be an empty list [].
    3. If you choose action="augment":
    - quads MUST be non-empty (a list of knowledge quadruples).
    - search_query MUST be an empty string "".
    4. Quadruple format:
    - timestamp MUST be in YYYY-MM-DD.
    - Each quad must be directly useful to answer the question or to connect key entities/events.
    5. Output JSON ONLY. No explanations.

    # Output JSON
    {{
    "action": "retrieve" | "augment",
    "search_query": "",
    "quads": [
        {{
        "subject": "",
        "relation": "",
        "object": "",
        "timestamp": "YYYY-MM-DD"
        }}
    ]
    }}
    """.lstrip()

    reflect_resp = get_llm_response(
        prompt=reflect_prompt,
        model_name=model_name,
        json_schema=reflect_json_schema if 'claude' not in model_id else None,
    ).strip()

    reflect_obj = json.loads(reflect_resp)
    action = reflect_obj["action"]
    search_query = reflect_obj["search_query"]
    quads = reflect_obj["quads"]

    fallback_trace = []
    fallback_trace.append({
        "stage": "reflect_0",
        "action": action,
        "search_query": search_query,
        "quads": quads,
    })

    if action == "augment":
        accumulated_knowledge = list(retrieved_knowledge)
        accumulated_knowledge.extend(quads)
        evidence_from_EEG2, filtered_knowledge = construct_eeg(accumulated_knowledge, time_start, time_end)
    else:
        use_time_aware = True
        if use_time_aware:
            dlist = time_aware_retrieve(
                query=search_query,
                ref_time_start=parse_date(time_start),
                ref_time_end=parse_date(time_end),
                vector_store=vector_store,
                k_final=retrieved_docs_num,
            )
        else:
            dlist = retriever.invoke(search_query)

        new_quads = [{
            "subject": d.metadata["subject"],
            "relation": d.metadata["relation"],
            "object": d.metadata["object"],
            "timestamp": d.metadata["timestamp"],
        } for d in dlist]

        accumulated_knowledge = list(retrieved_knowledge)
        seen = set((q["subject"], q["relation"], q["object"], q["timestamp"]) for q in accumulated_knowledge)
        for q in new_quads:
            key = (q["subject"], q["relation"], q["object"], q["timestamp"])
            if key not in seen:
                accumulated_knowledge.append(q)
                seen.add(key)
        accumulated_knowledge.sort(key=lambda q: parse_date(q["timestamp"]))

        fallback_trace.append({
            "stage": "search_1",
            "search_query": search_query,
            "added_facts": [quad_to_text(q) for q in new_quads],
            "final_evidence_size": len(accumulated_knowledge),
        })

    if len(accumulated_knowledge) == 0:
        evidence_from_EEG2 = "No evidence available."
    else:
        evidence_from_EEG2, _ = construct_eeg(accumulated_knowledge, time_start, time_end)

    response= get_llm_response(
        prompt=final_qa_prompt.format(
            question=query,
            evidence_from_EEG=evidence_from_EEG2,
        ),
        model_name=model_name,
    ).strip()

    predicted_answer = extract_answer_from_response(response)

    process_info = {
        "query": query,
        "entities": entities,
        "has_time": has_time,
        "time_start": time_start,
        "time_end": time_end,
        "query_without_time": query_without_time,
        "retrieved_knowledge": retrieved_knowledge,
        "retrieved_knowledge_text": [quad_to_text(q) for q in retrieved_knowledge],
        "filtered_knowledge": filtered_knowledge,
        "history_context": history_quad,
        "evidence_from_EEG": evidence_from_EEG,
        "initial_reasoning": initial_reasoning,
        "initial_answer": initial_predicted_answer,
        "fallback_reflect_trace": fallback_trace,
        "final_answer": predicted_answer
    }

    return predicted_answer, process_info



#### Main
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

    correct_cnt = 0
    seen_cnt = 0

    pbar = tqdm(list(zip(questions, groundtruths)), total=len(questions), desc=f"{task}")
    for i, (ques, gold) in enumerate(pbar):
        record_path = os.path.join(output_dir, f'record_{i}.json')

        if os.path.exists(record_path):
            with open(record_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            pred = record.get("final_answer", "").strip()
            em = evaluate_qa_exact_match([pred], [gold])
        else:
            pred, process_info = chronos_qa_pipeline(ques)
            process_info["gold"] = gold
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(process_info, f, ensure_ascii=False, indent=2)

        seen_cnt += 1
        em = evaluate_qa_exact_match([pred], [gold])  # 0/1
        correct_cnt += int(em)

        running_acc = correct_cnt / seen_cnt
        pbar.set_postfix(acc=f"{running_acc:.4f}", correct=f"{correct_cnt}/{seen_cnt}")

    final_acc = correct_cnt / max(seen_cnt, 1)
    print('====================')
    print(f"[{task} QA] Exact Match Accuracy: {final_acc:.4f}")
    print('====================')
