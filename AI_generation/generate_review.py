from utils import *
import yaml
import sys

'''
Read the parsed pdf -- need a function get_paper
add it into the prompt -- need a function make_prompt

send that prompt to GPT -- need a function llm
extract the final review -- need a function extract_answer

write the review at the right place in the right format -- a function write_review

sys args - paper_address, language_model, prompt_template
'''
def generate_review(paper_address, paper_number,  language_model, prompt_template_name):
    PaperString = get_paper(paper_address, paper_number)
    prompt_path = "AI_generation/prompts.yaml"
    with open(prompt_path) as f:
        prompts = yaml.safe_load(f)
    prompt_template = prompts.get(prompt_template_name, None)
    Prompt = make_prompt(prompt_template, PaperString)
    LLMReview = llm_call(Prompt, language_model, 0)
    ExtractedReview = extract_answer(LLMReview)
    write_review(paper_address, paper_number, ExtractedReview)

# generate_review("data/acl_2017/dev", "37", "gpt-35-turbo", "paper_prompt_1")

