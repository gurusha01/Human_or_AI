import json
import re
import os
from openai import AzureOpenAI, OpenAI
import time

def clean_text(text):
    
    numbers = re.findall(r'(\d+)\n', text)
    cleaned_text = text
    for number in numbers:
        cleaned_text = cleaned_text.replace(f'{number}\n', '')
    # print(cleaned_text, numbers)
    return cleaned_text

def get_paper(paper_address, paper_number):
    final_address = f'{paper_address}/parsed_pdfs/{paper_number}.pdf.json'
    with open(final_address, "r", encoding='utf-8') as file:
        data = json.load(file)
    data_dict = {}
    sections = data.get('metadata', {}).get('sections', [])
    final_string = ""
    for section in sections:
        heading = section.get('heading')
        text = section.get('text')
        text = clean_text(text)
        data_dict[heading] = text
        if(heading!= None and text!= None):
            final_string += "##"+heading + "\n\n" + text
    return final_string

def make_prompt(prompt_template, paper_contents):
    formatted_prompt = prompt_template.format(
            PaperInPromptFormat=paper_contents,
        )
    return formatted_prompt

def llm_call(prompt, llm_name, temperature):
    api_key = os.environ.get("OPENAI_API_KEY")
    api_endpoint = os.environ.get("OPENAI_ENDPOINT")
    api_version = os.environ.get("OPENAI_VERSION")

    client = AzureOpenAI(api_key=api_key, azure_endpoint=api_endpoint, api_version=api_version)
    
    while True:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model = llm_name, temperature= temperature, messages=messages)
            # print(response)
            return response.choices[0].message.content
        except Exception as e:
            pass
            print(f"Got error {e}. Sleeping for 5 seconds...")
            time.sleep(5)      

def extract_answer(answer_string):
    pattern = r"<Answer>(.*?)<\/Answer>"
    match = re.search(pattern, answer_string, re.DOTALL)
    answer = match.group(1) if match else answer_string
    return answer

def write_review(paper_address, paper_number, Review):
    # Check if the directory 'reviews_llm' exists, and if not, create it
    reviews_llm_dir = os.path.join(paper_address, 'reviews_llm')
    if not os.path.exists(reviews_llm_dir):
        os.makedirs(reviews_llm_dir)
    
    # Define the final address for the JSON file
    final_address = os.path.join(reviews_llm_dir, f'{paper_number}.json')
    
    # Create the JSON object
    json_object = {
        "reviews": [
            {
                "comments": Review,
            }
        ],
    }

    # Write the JSON object to the final address
    with open(final_address, 'w', encoding='utf-8') as json_file:
        json.dump(json_object, json_file, ensure_ascii=False, indent=4)

    return final_address

