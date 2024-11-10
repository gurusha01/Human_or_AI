from utils import *
import yaml
import sys

'''
Read the parsed pdf -- need a function get_paper
read the human written review -- need a function get_human_review
add both into the prompt -- need a function make_prompt


send that prompt to GPT -- need a function llm
extract the final review -- need a function extract_answer

write the review at the right place in the right format -- a function write_review

sys args - paper_address, language_model, prompt_template, level
'''
def generate_review(paper_address, paper_number,  language_model, prompt_template_name, level):
    PaperString = get_paper(paper_address, paper_number)
    HumanPrompt = get_human_review(paper_address, paper_number, level, language_model, 0.7)
    
    prompt_path = "AI_generation/prompts.yaml"
    with open(prompt_path) as f:
        prompts = yaml.safe_load(f)
    prompt_template = prompts.get(prompt_template_name, None)
    Prompt = make_prompt(prompt_template, PaperString, HumanPrompt)
    LLMReview = llm_call(Prompt, language_model, 0.7)
    ExtractedReview = extract_answer(LLMReview)
    # print("HUMAN REVIEW SUMMARIZED:", HumanPrompt)
    # print("*************************************")
    # print("PROMPT FOR REVIEW GENERATION:", Prompt)
    # print("*************************************")
    # print("LLM GENERATED PROMPT:", ExtractedReview)
    # print("*************************************")
    
    write_review(paper_address, paper_number, ExtractedReview, level)


# generate_review("data/acl_2017/dev", "37", "gpt-35-turbo", "paper_prompt_1")

