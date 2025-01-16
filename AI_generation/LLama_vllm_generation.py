import os
import vllm
import torch
import transformers
from utils import *
import sys

# VLLM by default has logs on, this is to switch it off
# os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
sys.stdout = open("playground.txt", "w")
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# NUM_GPU_UTILIZATION = 4
NUM_GPU_UTILIZATION = 1
PER_GPU_UTILIZATION = 0.8
batch_size = 10
outputpath_model = "reviews_llama_3_1_70b"

print(f"Using GPUs :{torch.cuda.device_count()}")

# MODEL = '/assets/models/meta-llama-3.1-instruct-70b'
MODEL = '/assets/models/meta-llama-3.1-instruct-8b' #single_gpu
# MODEL = "/assets/models/meta-llama-3.2-instruct-3b" #single_gpu
print("MODEL :",MODEL)

def get_tokenizer_and_config(model_name):
    tokenizer_args = {}

    if "gemma-2" in model_name.lower():
        # Gemma 2 was likely trained with right padding:
        # https://github.com/huggingface/transformers/issues/30004
        tokenizer_args['padding_side'] = "right"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, **tokenizer_args
    )

    if getattr(tokenizer, 'pad_token', None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    config = transformers.AutoConfig.from_pretrained(model_name)

    return tokenizer, config

def load_model():
    tokenizer, config = get_tokenizer_and_config(MODEL)

    context_window = min(
         getattr(config, 'sliding_window', None) or config.max_position_embeddings,
         config.max_position_embeddings, 25600
    )

    model = vllm.LLM(MODEL, trust_remote_code=True, dtype=torch.float16,
                     gpu_memory_utilization=PER_GPU_UTILIZATION,
                     tensor_parallel_size=NUM_GPU_UTILIZATION,
                     # this size is to avoid GPU OOM errors for LLaMa 3.1 which has a 128K context window
                     max_model_len = context_window)
                    # quantization = 'fp8')

    return tokenizer,model

def prompt_model(batch_prompts):
    # VLLM can directly tokenize, but applying the chat template is a bit complex for batched inference.
    # and this helps with simulating max_new_tokens which VLLM does not support.
    strings = tokenizer.apply_chat_template(batch_prompts, tokenize=False, add_generation_prompt=True)
    _tokens = tokenizer(strings, add_special_tokens=False, padding="longest", return_tensors="pt")
    max_input_len = _tokens.input_ids.shape[-1]
    tokens  = [ tokenizer(string, add_special_tokens=False).input_ids for string in strings ]
    tokens  = [ vllm.TokensPrompt(prompt_token_ids=tok_ids) for tok_ids in tokens ]

    # generation arguments in this separate object (mostly has one-to-one mapping with HF tokenizers)
    max_new_tokens = 8000
    gen_kwargs = vllm.SamplingParams(temperature=0, top_p=0.9, max_tokens=max_new_tokens + max_input_len)
    # outputs have both tokens and detokenized strings for direct use.
    outputs = model.generate(tokens, sampling_params=gen_kwargs)
    return outputs

def process_llm_output(outputs,reviews_llm_dir,papername_list,level):
    out_strings = [[ response.text for response in output.outputs ] for output in outputs]
    for (each_output,paper_number) in zip(out_strings,papername_list):
        print("$"*50)
        print(each_output[0])    
        print("$"*50)
        answer = extract_answer(each_output[0])
        if level == "level3":
            return answer
        else:
            write_review(reviews_llm_dir, paper_number, answer) 

def generate_level1(data_dir,guideline_in_prompt,output_format,prompt_template):
    folders = os.listdir(data_dir) #dev,test,train folders
    batch_prompts =[]
    for folder in folders:
        folderpath = os.path.join(data_dir,folder) #../data/nips_2013-2017/2017/test
        parsed_pdfs = os.path.join(folderpath,"parsed_pdfs")
        
        output_path = os.path.join(folderpath,outputpath_model)
        os.makedirs(output_path, exist_ok=True)
        
        file_output_path = os.path.join(output_path,"level1")
        os.makedirs(file_output_path, exist_ok=True)
        
        papername_list = []
        for each_paper in os.listdir(parsed_pdfs):
            paper_name = each_paper.split('.')[0]
            papername_list.append(each_paper)
            PaperString = get_paper(folderpath,paper_name)
            complete_prompt = prompt_template.format(guidelines= guideline_in_prompt,
                                                     PaperInPromptFormat=PaperString) + output_format
            print('#'*50)
            print(complete_prompt)
            print('#'*50)
            batch_prompts.append([dict(role='system', content=output_format),
                                          dict(role='user', content=complete_prompt)])
            
            #Time to process
            if len(batch_prompts)== batch_size:
                outputs = prompt_model(batch_prompts) 
                process_llm_output(outputs,file_output_path,papername_list,"1")
                batch_prompts =[]   
                papername_list = []
                
def generate_level2(data_dir,guideline_in_prompt,output_format,prompt_template):
    folders = os.listdir(data_dir) #dev,test,train folders
    batch_prompts =[]
    for folder in folders:
        folderpath = os.path.join(data_dir,folder) #../data/nips_2013-2017/2017/test
        parsed_pdfs = os.path.join(folderpath,"parsed_pdfs")
        output_path = os.path.join(folderpath,outputpath_model)
        os.makedirs(output_path, exist_ok=True)
        file_output_path = os.path.join(output_path,"level2")
        os.makedirs(file_output_path, exist_ok=True)
        papername_list = []
        for each_paper in os.listdir(parsed_pdfs): 
            paper_name = each_paper.split('.')[0]
            papername_list.append(paper_name)
            PaperString = get_paper(folderpath,paper_name)
            complete_prompt = prompt_template.format(guidelines= guideline_in_prompt,
                                                     PaperInPromptFormat=PaperString) + output_format
            print('#'*50)
            print(complete_prompt)
            print('#'*50)
            batch_prompts.append([dict(role='system', content=output_format),
                                          dict(role='user', content=complete_prompt)])
            #Time to process
            if len(batch_prompts)== batch_size:
                outputs = prompt_model(batch_prompts) 
                process_llm_output(outputs,file_output_path,papername_list,"2")
                batch_prompts =[]   
                papername_list = []


def generate_level4(data_dir,output_format,prompt_template):
    folders = os.listdir(data_dir) #dev,test,train folders
    batch_prompts =[]
    
    for folder in folders:
        folderpath = os.path.join(data_dir,folder) #../data/nips_2013-2017/2017/test
        human_reviews = os.path.join(folderpath,"reviews")
        output_path = os.path.join(folderpath,outputpath_model)
        os.makedirs(output_path, exist_ok=True)
        
        file_output_path = os.path.join(output_path,"level4")
        os.makedirs(file_output_path, exist_ok=True)
        
        papername_list = []
        for each_paper in os.listdir(human_reviews): 
            paper_name = each_paper.split('.')[0]
            print(f"paper_name : {paper_name}")
            human_reviews = get_human_review_all(folderpath,paper_name)
            
            for each_review in human_reviews:
                papername_list.append(paper_name)
                complete_prompt = prompt_template.format(humanreview = each_review) + output_format
                print('#'*50)
                print(complete_prompt)
                print('#'*50)
                batch_prompts.append([dict(role='system', content=output_format),
                                            dict(role='user', content=complete_prompt)])
                #Time to process
                if len(batch_prompts)== batch_size:
                    outputs = prompt_model(batch_prompts) 
                    process_llm_output(outputs,file_output_path,paper_name,"4")
                    batch_prompts =[] 
                    papername_list = []


def generate_llm_review(data_dir,level,conference):
    prompt_path = "./prompts.yaml"
    guidelines_path = "./guidelines.yaml"

    with open(prompt_path) as f:
        promptsyaml = yaml.safe_load(f)
        
    with open(guidelines_path) as f:
        guidelinesyaml = yaml.safe_load(f)
        
    guideline_in_prompt = guidelinesyaml.get(conference,None)
    output_format = promptsyaml.get("output_format",None)
    
    # folders = os.listdir(data_dir) #dev,test,train folders
    if level == '1':
        prompt_template_name = f"level_{level}_prompt"
        prompt_template = promptsyaml.get(prompt_template_name, None)
        generate_level1(data_dir,guideline_in_prompt,output_format,prompt_template)
        
    elif level == '2':
        prompt_template_name = f"level_{level}_prompt"
        prompt_template = promptsyaml.get(prompt_template_name, None)
        generate_level2(data_dir,guideline_in_prompt,output_format,prompt_template)
        
    # elif level == '3':
    #     build_prompt_3(data_dir,guideline_in_prompt,output_format)
        
    elif level == '4':
        prompt_template_name = f"level_{level}_prompt"
        prompt_template = promptsyaml.get(prompt_template_name, None)
        generate_level4(data_dir,output_format,prompt_template)
    
    # for folder in folders:
    #     folderpath = os.path.join(data_dir,folder) #../data/nips_2013-2017/2017/test
    #     parsed_pdfs = os.path.join(folderpath,"parsed_pdfs")
    #     human_reviews = os.path.join(folderpath,"reviews")
    #     output_path = os.path.join(folderpath,"reviews_llama_3_1_70b")
        
    #     os.makedirs(output_path, exist_ok=True)
            
    #     for each_paper in os.listdir(parsed_pdfs): 
    #         paper_name = each_paper.split('.')[0]
    #         PaperString = get_paper(folderpath,paper_name)

    #         if level in ("1","2","4"): #Direct batching
    #             prompt_template_name = f"level_{level}_prompt"
    #             prompt_template = prompts.get(prompt_template_name, None)
    #             if level in ('1','2'): #only guidelines and paper, no human review
    #                 complete_prompt = prompt_template.format(guidelines= guideline_in_prompt,
    #                                                  PaperInPromptFormat=PaperString )
                
    #                 batch_prompts.append([dict(role='system', content=output_format),
    #                                       dict(role='user', content=complete_prompt)])
                    
    #             elif level == "4":# human written review needed
    #                 human_reviews = get_human_review_all(folderpath,paper_name)
    #                 for each_rev in human_reviews:
    #                     complete_prompt = prompt_template.format(humanreview= each_rev)
    #                     print(f"Level4 prompt: {complete_prompt}")
    #                     batch_prompts.append([dict(role='system', content=output_format),
    #                                       dict(role='user', content=complete_prompt)])
    #                     if len(batch_prompts)== batch_size:
    #                         outputs = prompt_model(batch_prompts) 
    #                         process_llm_output(outputs,file_output_path,paper_name)
    #                         batch_prompts =[] 
                            
    #             file_output_path = os.path.join(output_path,f"level{level}")

    # write_review(paper_address, paper_number, ExtractedReview, level)
 
                
    #     folders = os.listdir(base_dir) #dev,test,train folders
    #     for folder in folders:
    #         folderpath = os.path.join(base_dir,folder) #../data/nips_2013-2017/2017/test
    #         parsed_pdfs = os.path.join(folderpath,"parsed_pdfs")
    #         # human_reviews = os.path.join(folderpath,"reviews")
    #         output_path = os.path.join(folderpath,"reviews_llama_3_1_70b")
    #         if not os.path.exists(output_path):
    #             os.makedirs(output_path)
    #             print(f"Folder '{output_path}' created.")
    #         else:
    #             print(f"Folder '{output_path}' already exists.")
    #         for each_paper in os.listdir(parsed_pdfs): 
    #             paper_name = each_paper.split('.')[0]
    #             PaperString = get_paper(folderpath,paper_name)
    #             # print(f"\n\n*********NEW PAPER {Total_count} : {paper_name}******")
    #             # print(PaperString)
    #             if level == "level1":
    #                 complete_prompt = level1 + '\n' + guidelines + '\n' + "Paper:" + PaperString
    #                 file_output_path = os.path.join(output_path,"level1")
    #                 os.makedirs(file_output_path) if not os.path.exists(file_output_path) else print(f"already exists")
    #                 print(f"Level1 prompt: {complete_prompt}")
    #                 batch_prompts.append([dict(role='system', content=output_format),
    #                                       dict(role='user', content=complete_prompt)])
    #                 if len(batch_prompts)== batch_size:
    #                     outputs = prompt_model(batch_prompts) 
    #                     process_llm_output(outputs,file_output_path,paper_name,level)
    #                     batch_prompts =[]      
    #                 break
    #             elif level == "level2":
    #                 complete_prompt = level2 + '\n' + guidelines + '\n' + "Paper:" + PaperString
    #                 file_output_path = os.path.join(output_path,"level2")
    #                 os.makedirs(file_output_path) if not os.path.exists(file_output_path) else print(f"already exists")
    #                 print(f"Level2 prompt: {complete_prompt}")
    #                 batch_prompts.append([dict(role='system', content=output_format),
    #                                       dict(role='user', content=complete_prompt)])
    #                 if len(batch_prompts)== batch_size:
    #                     outputs = prompt_model(batch_prompts) 
    #                     process_llm_output(outputs,file_output_path,paper_name,level)
    #                     batch_prompts =[]    
                        
    #             elif level == "level3":
    #                 file_output_path = os.path.join(output_path,"level3")
    #                 os.makedirs(file_output_path) if not os.path.exists(file_output_path) else print(f"already exists")
    #                 human_reviews = get_human_review_all(folderpath,paper_name)
    #                 for each_rev in human_reviews:
    #                     summarization_prompt = level3_summarizationprompt + '\n'+ "Review:" + each_rev
    #                     prompt = [[dict(role='system', content=output_format),
    #                                       dict(role='user', content=summarization_prompt)]]
    #                     outputs = prompt_model(prompt)
    #                     keypoints = process_llm_output(outputs,file_output_path,paper_name,level)
    #                     complete_prompt = level3_generation + "\n" + "keypoints:" + keypoints+ '\n' + guidelines + '\n' + "Paper:" + PaperString
    #                     print(f"Level3 prompt: {complete_prompt}")
    #                     batch_prompts.append([dict(role='system', content=output_format),
    #                                       dict(role='user', content=complete_prompt)])
    #                     if len(batch_prompts)== batch_size:
    #                         outputs = prompt_model(batch_prompts) 
    #                         process_llm_output(outputs,file_output_path,paper_name,"some string")
    #                         batch_prompts =[]   
                            
    #             elif level == "level4": 
    #                 file_output_path = os.path.join(output_path,"level4")
    #                 os.makedirs(file_output_path) if not os.path.exists(file_output_path) else print(f"already exists")
    #                 human_reviews = get_human_review_all(folderpath,paper_name)
    #                 for each_rev in human_reviews:
    #                     complete_prompt = level4 + '\n' + "Review :" + each_rev
    #                     print(f"Level4 prompt: {complete_prompt}")
    #                     batch_prompts.append([dict(role='system', content=output_format),
    #                                       dict(role='user', content=complete_prompt)])
    #                     if len(batch_prompts)== batch_size:
    #                         outputs = prompt_model(batch_prompts) 
    #                         process_llm_output(outputs,file_output_path,paper_name)
    #                         batch_prompts =[] 


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: script.py '../data/nips_2013-2017/2017/' '1' 'nips_2017'")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    level = sys.argv[2]
    conference = sys.argv[3]
    print("Processing directory :",data_dir)
    print("Level :",level)
    print("Using guidelines from :",conference)
    
    tokenizer, model = load_model()
    # print("Successfully loaded model!")
    
    # data_dir = [ "../data/acl_2017/",
    #                  "../data/conll_2016/",
    #                   "../data/iclr_2017/",
    #                    "../data/nips_2013-2017/2013/",
    #                     "../data/nips_2013-2017/2014/",
    #                      "../data/nips_2013-2017/2015/",
    #                       "../data/nips_2013-2017/2016/",
    #                        "../data/nips_2013-2017/2017/"]

    # data_dir = ["../data/nips_2013-2017/2017/"]
        
    # output = get_llm_output(data_dir,level,conference)
    generate_llm_review(data_dir,level,conference)