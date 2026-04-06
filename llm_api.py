import os
from openai import AzureOpenAI, OpenAI
import anthropic
from anthropic import AnthropicFoundry



####### Source Selection ########
api_source = 'openai'
# api_source = 'azure'
# api_source = 'claude'
# api_source = 'deepinfra'


####### Model List ########
''':
OpenAI Models:
    gpt-5-2025-08-07
    gpt-5-mini-2025-08-07
    gpt-5-nano-2025-08-07
    gpt-4.1-2025-04-14
    gpt-4.1-mini-2025-04-14
    gpt-4.1-nano-2025-04-14
    text-embedding-3-large
    text-embedding-3-small

    gpt-4o-2024-11-20 (knowledge cutoff Oct 01, 2023)
    gpt-4o-mini-2024-07-18 (knowledge cutoff Oct 01, 2023)
    gpt-3.5-turbo-0125 (knowledge cutoff Sep 01, 2021)

Azure Models:
    gpt-5 (2025-08-07)
    gpt-5-mini (2025-08-07)
    gpt-5-nano (2025-08-07)
    gpt-4.1 (2025-04-14)
    gpt-4.1-mini (2025-04-14)
    gpt-4.1-nano (2025-04-14)
    claude-haiku-4-5
    claude-sonnet-4-5
    claude-opus-4-5
    DeepSeek-V3.1
    Llama-3.3-70B-Instruct
    Meta-Llama-3.1-8B-Instruct (knowledge cutoff Dec, 2023)
    text-embedding-3-large
    text-embedding-3-small

    gpt-4o-mini (2024-07-18, knowledge cutoff Oct 01, 2023)
    
Claude Models:
    claude-3-haiku-20240307 (knowledge cutoff Aug 2023)
'''


####### Loading API Key ########
if api_source == 'openai' and os.path.exists('openai_api.key'):
    with open('openai_api.key', 'r') as f:
        api_key = f.read().strip()
    openai_client = OpenAI(api_key=api_key)
elif api_source == 'azure' and os.path.exists('azure_api.key'):
    with open('azure_api.key', 'r') as f:
        api_key = f.read().strip()
    openai_client = OpenAI(api_key=api_key,
        base_url='',
    )
    claude_client = AnthropicFoundry(api_key=api_key,
        base_url='',
    )
elif api_source == 'claude' and os.path.exists('claude_api.key'):
    with open('claude_api.key', 'r') as f:
        api_key = f.read().strip()
    claude_client = anthropic.Anthropic(api_key=api_key)
elif api_source == 'deepinfra' and os.path.exists('deepinfra_api.key'):
    with open('deepinfra_api.key', 'r') as f:
        api_key = f.read().strip()
    openai_client = OpenAI(api_key=api_key,
        base_url="",
    )
else:
    raise ValueError("No available API sources.")



# max_tokens will be depracated for OpenAI but support llama and other opensource models


####### LLM Response Functions ########
# for supporting different JSON structured outputs and varying API differences

# openai models
def _get_openai_llm_response(prompt, model_name='', json_schema=None, temperature=0, max_tokens=4096):
    if 'gpt-5' in model_name: # only temperature 1 allowed for gpt-5
        temperature = 1

    if json_schema:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema["title"],
                    "description": json_schema.pop("description", ""),
                    "schema": json_schema,
                    "strict": True
                }
            },
            temperature=temperature,
            max_completion_tokens=max_tokens
        ).choices[0].message.content
    else:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "text"},
            temperature=temperature,
            max_completion_tokens=max_tokens
        ).choices[0].message.content

    return response


# anthropic claude models
def _get_claude_llm_response(prompt, model_name='', json_schema=None, temperature=0, max_tokens=4096):
    if json_schema:
        json_schema['additionalProperties'] = False
        response = claude_client.beta.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            betas=["structured-outputs-2025-11-13"],
            output_format={
                "type": "json_schema",
                "schema": json_schema
            },
            temperature=temperature,
            max_tokens=max_tokens
        ).content[0].text
    else:
        response = claude_client.messages.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        ).content[0].text
    return response

# azure opensource models
def _get_other_llm_response(prompt, model_name='', json_schema=None, temperature=0, max_tokens=4096):
    if json_schema:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_object",
                "schema": json_schema
            },
            temperature=temperature,
            max_tokens=max_tokens
        ).choices[0].message.content
    else:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "text"},
            temperature=temperature,
            max_tokens=max_tokens
        ).choices[0].message.content
    return response


####### Unified LLM calling Functions ########
def get_llm_response(prompt, model_name='', json_schema=None, temperature=0, max_tokens=4096):
    if 'gpt' in model_name:
        response = _get_openai_llm_response(prompt, model_name, json_schema, temperature, max_tokens)
    elif 'claude' in model_name:
        response = _get_claude_llm_response(prompt, model_name, json_schema, temperature, max_tokens)
    else:
        response = _get_other_llm_response(prompt, model_name, json_schema, temperature, max_tokens)
    return response


def get_llm_embedding(texts, model='text-embedding-3-small'):
    if isinstance(texts, str):
        texts = [texts]
    response = openai_client.embeddings.create(
        input=texts,
        model=model
    )
    embedding = [D.embedding for D in response.data]
    return embedding



