import os
import vllm
import torch
import transformers
from utils import *
import sys
import time
import json

# VLLM by default has logs on, this is to switch it off
# os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,5,6"
NUM_GPU_UTILIZATION = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
PER_GPU_UTILIZATION = 0.9
batch_size = 40

print(f"Using GPUs :{torch.cuda.device_count()}")

MODEL = '/assets/models/meta-llama-3.3-instruct-70b'
# MODEL = '/assets/models/meta-llama-3.1-instruct-8b' #single_gpu*48GB
# MODEL = "/assets/models/meta-llama-3.2-instruct-3b" #single_gpu*48GB
print("MODEL :",MODEL)
outputpath_model = MODEL.split('/')[-1]

def get_examples(expid):
    with open("ICLexamples.json",'r') as file:
        all_examples_dict = json.load(file)

    if expid == 'experiment_1':#all classes needed
        pass
    elif expid == 'experiment_2':
        human_count = 8
        llm_count = (human_count//4)//2
        # print(llm_count)
        classes = [("examples_ai",all_examples_dict['level1_gpt4o'][:llm_count]+
                    all_examples_dict['level1_llama'][:llm_count]+
                    all_examples_dict['level2_gpt4o'][:llm_count]+all_examples_dict['level2_llama'][:llm_count]+
                    all_examples_dict['level3_gpt4o'][:llm_count]+all_examples_dict['level3_llama'][:llm_count]+
                    all_examples_dict['level4_gpt4o'][:llm_count]+all_examples_dict['level4_llama'][:llm_count]),
                   ("examples_human", all_examples_dict['human_examples'][:human_count])]
        
    elif expid == 'experiment_3':
        acceptable_count = 8 #level 4 and human
        human_count = acceptable_count//2
        llm_count_acc = acceptable_count//4
        
        unacceptable_count = 6
        llm_count_unacc = (unacceptable_count//3)//2
        classes = [("examples_acceptable", 
                    all_examples_dict['level4_gpt4o'][:llm_count_acc]+all_examples_dict['level4_llama'][:llm_count_acc]+all_examples_dict['human_examples'][:human_count]),
                   ("examples_notacceptable",
                    all_examples_dict['level1_gpt4o'][:llm_count_unacc]+all_examples_dict['level1_llama'][:llm_count_unacc]+
                    all_examples_dict['level2_gpt4o'][:llm_count_unacc]+all_examples_dict['level2_llama'][:llm_count_unacc]+
                    all_examples_dict['level3_gpt4o'][:llm_count_unacc]+all_examples_dict['level3_llama'][:llm_count_unacc])]
        
    elif expid == 'experiment_4':
        per_class_count = 6   #############
        llmcount = per_class_count//2
        classes = [("level1_examples", all_examples_dict['level1_gpt4o'][:llmcount]+all_examples_dict['level1_llama']           [:llmcount]),
                   ("level2_examples", all_examples_dict['level2_gpt4o'][:llmcount]+all_examples_dict['level2_llama'][:llmcount]),
                   ("level3_examples", all_examples_dict['level3_gpt4o'][:llmcount]+all_examples_dict['level3_llama'][:llmcount]),
                   ("level4_examples", all_examples_dict['level4_gpt4o'][:llmcount]+all_examples_dict['level4_llama'][:llmcount]),
                   ("human_examples", all_examples_dict['human_examples'][:per_class_count])]
    #shuffling
    return classes
        
# all_ai_examples = level1_examples+level2_examples+level3_examples+level4_examples
skipped_set = []
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
         config.max_position_embeddings, 51200
    )

    model = vllm.LLM(MODEL, trust_remote_code=True, dtype=torch.float16,
                     gpu_memory_utilization=PER_GPU_UTILIZATION,
                     tensor_parallel_size=NUM_GPU_UTILIZATION,
                     # this size is to avoid GPU OOM errors for LLaMa 3.1 which has a 128K context window
                     max_model_len = context_window,
                     enforce_eager=True)
                    # quantization = 'fp8')

    return tokenizer,model

def prompt_model(batch_prompts):
    # VLLM can directly tokenize, but applying the chat template is a bit complex for batched inference.
    # and this helps with simulating max_new_tokens which VLLM does not support.
    strings = tokenizer.apply_chat_template(batch_prompts, tokenize=False, add_generation_prompt=True)
    tokens_dict = tokenizer(strings, add_special_tokens=False, padding=True, return_tensors="pt")
    max_input_len = tokens_dict["input_ids"].shape[-1]
    tokens = [vllm.TokensPrompt(prompt_token_ids=tok.tolist()) for tok in tokens_dict["input_ids"]]

    
    # _tokens = tokenizer(strings, add_special_tokens=False, padding="longest", return_tensors="pt")
    # max_input_len = _tokens.input_ids.shape[-1]
    # tokens  = [ tokenizer(string, add_special_tokens=False).input_ids for string in strings ]
    # tokens  = [ vllm.TokensPrompt(prompt_token_ids=tok_ids) for tok_ids in tokens ]

    # generation arguments in this separate object (mostly has one-to-one mapping with HF tokenizers)
    max_new_tokens = 8000
    gen_kwargs = vllm.SamplingParams(temperature=0.2, top_p=0.8, max_tokens=max_new_tokens + max_input_len)
    # outputs have both tokens and detokenized strings for direct use.
    outputs = model.generate(tokens, sampling_params=gen_kwargs)
    return outputs
def process_llm_output(outputs,output_path):
    
    out_strings = [[ response.text for response in output.outputs ] for output in outputs]
    
    for each_output,outfile_path in zip(out_strings,output_path):
        # print(outfile_path)
        # print("%"*50)
        # print(each_output)
        try:
            # print(each_output)
            match = re.search(r"\{.*\}", each_output[0]) 
            json_string = match.group(0)
            # print("22222222")
            # print(json_string)
            parsed_data = json.loads(json_string)
            with open(outfile_path, "w") as file:
                json.dump(parsed_data, file, indent=4)
            print(f"Wrote : {outfile_path}")
        except:
            skipped_set.append(outfile_path)
            print("\nskipped : ",outfile_path)
            print(each_output)

# def run_experiment1(system_prompt,user_prompt):
#     # folders = os.listdir(data_dir) #dev,test,train folders
#     folders = ['dev','test','train']
#     batch_prompts =[]
#     for folder in folders:
#         folderpath = os.path.join(data_dir,folder) #../data/nips_2013-2017/2017/test
#         parsed_pdfs = os.path.join(folderpath,"parsed_pdfs")
        
#         output_path = os.path.join(folderpath,outputpath_model)
#         os.makedirs(output_path, exist_ok=True)
        
#         file_output_path = os.path.join(output_path,"level1")
#         os.makedirs(file_output_path, exist_ok=True)
        
#         papername_list = []

#         for each_paper in os.listdir(parsed_pdfs):
#             paper_name = each_paper.split('.')[0]
#             papername_list.append(paper_name)
#             PaperString = get_paper(folderpath,paper_name)
#             complete_prompt = prompt_template.format(guidelines= guideline_in_prompt,
#                                                      PaperInPromptFormat=PaperString)
#                                                     #  OutputFormat=output_format)
#             print('#'*50)
#             print(complete_prompt)
#             # print('#'*50)
#             batch_prompts.append([dict(role='user', content=complete_prompt)])
            
#             #Time to process
#             if len(batch_prompts)== batch_size:
#                 outputs = prompt_model(batch_prompts) 
#                 process_llm_output(outputs,file_output_path,papername_list)
#                 batch_prompts =[]   
#                 papername_list = []
#         if batch_prompts:
#             outputs = prompt_model(batch_prompts) 
#             process_llm_output(outputs,file_output_path,papername_list)
#             batch_prompts =[]   
#             papername_list = []
                
# def run_experiment2(system_prompt,user_prompt,mode,experiment_id):
#     classes = get_examples(experiment_id)
#     path_prefix = "/home/naveeja/Project/Human_or_AI/Data_Preprocessing/"
#     # examples_level1 = "Example 1:''' lndlknfdlknlf '''"
#     examples_given = True
#     if not examples_given:
#         complete_system_prompt = system_prompt
#     else:
#         example_dict = {}
#         for str,listname in classes:
#             # print(str)
#             example_string  = ""
#             for i in range(len(listname)):
#                 path_to_review = path_prefix + listname[i]
#                 with open(path_to_review,"r") as file:
#                     review = file.read()
#                 example_string += f"Example{i+1} : '''{review}'''" + "\n"
                
#             example_dict[str] = example_string 
#         # print(system_prompt)
#         complete_system_prompt = system_prompt.format(examples_ai = example_dict["examples_ai"],
#                                                     examples_human = example_dict["examples_human"])

#     # print(complete_system_prompt)
#     if mode == "testing":
#         # file_path = "./all_paths_testdata.txt" 
#         file_path = "./subtestset.txt" 
#         # file_path = "./skipped_subsettest.txt" 
#     else: #calibration mode
#         file_path = "./all_paths_calibrationdata copy.txt" 
        
#     with open(file_path, "r") as file:
#         all_paths = file.readlines()

#     startidx = 0
#     all_paths = [path.strip() for path in all_paths][startidx:]
#     batch_prompts =[]
#     file_paths = []
#     for each_path in all_paths:
#         # if each_path not in all_examples: #take only the new 
#         with open(each_path,"r") as file:
#                 review = file.read()
#         complete_user_prompt = user_prompt.format(input_review = review)
#         batch_prompts.append([dict(role='system', content=complete_system_prompt),
#                             dict(role='user', content=complete_user_prompt)])
#         print('*'*40)
#         print(complete_system_prompt)
#         print(complete_user_prompt)
#         print('*'*40)
#         # /home/naveeja/Project/Human_or_AI/Data_Preprocessing/cleandata/acl_2017/dev/meta-llama-Llama-3.3-70B-Instruct/level3/173_1.txt
#         # print(f"file path original :{each_path}")
#         file_output_path = each_path.replace("Data_Preprocessing/cleandata",f"AI_generation/LLamaDetectorResponses_{mode}_exp{experiment_id[-1]}").replace(".txt",".json")
#         # print(f"file_output_path while appending: {file_output_path}")
#         file_paths.append(file_output_path)
#         os.makedirs(os.path.dirname(file_output_path), exist_ok=True)
#         if len(batch_prompts) == batch_size:
#             outputs = prompt_model(batch_prompts) 
#             process_llm_output(outputs,file_paths)
#             batch_prompts =[] 
#             file_paths = []
            
            
#     if batch_prompts:
#         outputs = prompt_model(batch_prompts) 
#         process_llm_output(outputs,file_paths)
#         batch_prompts =[] 
#         file_paths = []
        
# def run_experiment3(system_prompt,user_prompt):
#     # folders = os.listdir(data_dir) #dev,test,train folders
#     folders = ['dev','test','train']
#     batch_prompts =[]
    
#     for folder in folders:
#         folderpath = os.path.join(data_dir,folder) #../data/nips_2013-2017/2017/test
#         human_reviews = os.path.join(folderpath,"reviews")
#         output_path = os.path.join(folderpath,outputpath_model)
#         os.makedirs(output_path, exist_ok=True)
        
#         file_output_path = os.path.join(output_path,"level3")
#         os.makedirs(file_output_path, exist_ok=True)
        
#         papername_list = []
#         for each_paper in os.listdir(human_reviews): 
#             paper_name = each_paper.split('.')[0]
#             # print(f"paper_name : {paper_name}")
#             human_reviews = get_human_review_all(folderpath,paper_name)
#             PaperString = get_paper(folderpath,paper_name)
#             for each_review in human_reviews:
#                 papername_list.append(paper_name)
#                 complete_prompt = summarize_prompt.format(humanreview = each_review)
#                                                         #   OutputFormat=output_format)
                
                
#                 print(f"Level 3 summarization :\n{complete_prompt}")
#                 print('#'*50)
                
#                 batch_prompts.append([dict(role='user', content=complete_prompt)])
#                 #Time to process
#                 if len(batch_prompts)== batch_size:
#                     outputs = prompt_model(batch_prompts) 
#                     keypoints = process_llm_output(outputs,file_output_path,papername_list,False)
#                     batch_prompts =[] 
#                     for each_summarized in keypoints:
#                         complete_prompt = generatn_prompt.format(summarized_humanreview = each_summarized,
#                                                                 guidelines=guideline_in_prompt,
#                                                                 PaperInPromptFormat=PaperString)
#                                                                 # OutputFormat=output_format)
#                         print(f"Level 3 generation :\n{complete_prompt}")
#                         print('#'*50)
                        
#                         batch_prompts.append([dict(role='user', content=complete_prompt)])
#                     outputs = prompt_model(batch_prompts) 
#                     process_llm_output(outputs,file_output_path,papername_list)
#                     batch_prompts =[] 
#                     papername_list = []
#         if batch_prompts:
#             outputs = prompt_model(batch_prompts) 
#             keypoints = process_llm_output(outputs,file_output_path,papername_list,False)
#             batch_prompts =[] 
#             for each_summarized in keypoints:
#                 complete_prompt = generatn_prompt.format(summarized_humanreview = each_summarized,
#                                                         guidelines=guideline_in_prompt,
#                                                         PaperInPromptFormat=PaperString)
#                                                         # OutputFormat=output_format)
#                 print(f"Level 3 generation :\n{complete_prompt}")
#                 print('#'*50)
#                 batch_prompts.append([dict(role='user', content=complete_prompt)])
#             outputs = prompt_model(batch_prompts) 
#             process_llm_output(outputs,file_output_path,papername_list)
#             batch_prompts =[] 
#             papername_list = []                 

def run_experiment(system_prompt,user_prompt,mode,experiment_id):
    
    classes = get_examples(experiment_id)
    path_prefix = "/home/naveeja/Project/Human_or_AI/Data_Preprocessing/"
    # examples_level1 = "Example 1:''' lndlknfdlknlf '''"
    examples_given = True
    if not examples_given:
        complete_system_prompt = system_prompt
    else:
        example_dict = {}
        for str,listname in classes:
            # print(str)
            example_string  = ""
            for i in range(len(listname)):
                path_to_review = path_prefix + listname[i]
                with open(path_to_review,"r") as file:
                    review = file.read()
                example_string += f"Example{i+1} : '''{review}'''" + "\n"
                
            example_dict[str] = example_string 
        # print(system_prompt)
        
        if experiment_id == 'experiment_1':
            pass
        elif experiment_id == 'experiment_2':
            complete_system_prompt = system_prompt.format(examples_ai = example_dict["examples_ai"],
                                                    examples_human = example_dict["examples_human"])
        elif experiment_id == 'experiment_3':
            complete_system_prompt = system_prompt.format(examples_acceptable = example_dict["examples_acceptable"],examples_notacceptable = example_dict["examples_notacceptable"])
            
        elif experiment_id == 'experiment_4':
            complete_system_prompt = system_prompt.format(examples_level1 = example_dict["level1_examples"],
                                                        examples_level2 = example_dict["level2_examples"],
                                                        examples_level3 = example_dict["level3_examples"],
                                                        examples_level4 = example_dict["level4_examples"],
                                                        examples_human = example_dict["human_examples"])

    # print(complete_system_prompt)
    if mode == "testing":
        # file_path = "./all_paths_testdata.txt" 
        file_path = "./subtestset.txt" 
        # file_path = "./skipped_subsettest.txt" 
    else: #calibration mode
        file_path = "./all_paths_calibrationdata.txt" 
        
    with open(file_path, "r") as file:
        all_paths = file.readlines()

    startidx = 0
    all_paths = [path.strip() for path in all_paths][startidx:]
    batch_prompts =[]
    file_paths = []
    for each_path in all_paths:
        with open(each_path,"r") as file:
                review = file.read()
        complete_user_prompt = user_prompt.format(input_review = review)
        batch_prompts.append([dict(role='system', content=complete_system_prompt),
                            dict(role='user', content=complete_user_prompt)])
        print('*'*40)
        print(complete_system_prompt)
        print(complete_user_prompt)
        print('*'*40)
        # /home/naveeja/Project/Human_or_AI/Data_Preprocessing/cleandata/acl_2017/dev/meta-llama-Llama-3.3-70B-Instruct/level3/173_1.txt
        # print(f"file path original :{each_path}")
        file_output_path = each_path.replace("Data_Preprocessing/cleandata",f"AI_generation/LLamaDetectorResponses_{mode}_exp{experiment_id[-1]}").replace(".txt",".json")
        # print(f"file_output_path while appending: {file_output_path}")
        file_paths.append(file_output_path)
        os.makedirs(os.path.dirname(file_output_path), exist_ok=True)
        if len(batch_prompts) == batch_size:
            outputs = prompt_model(batch_prompts) 
            process_llm_output(outputs,file_paths)
            batch_prompts =[] 
            file_paths = []
            
            
    if batch_prompts:
        outputs = prompt_model(batch_prompts) 
        process_llm_output(outputs,file_paths)
        batch_prompts =[] 
        file_paths = []
    
def do_classify(experiment_id,mode):
    prompt_path = "./detector_prompts.yaml"
    
    with open(prompt_path) as f:
        promptsyaml = yaml.safe_load(f)
        
    user_prompt = promptsyaml.get("user_prompt", None)
    if experiment_id == 'experiment_1':
        prompt_template_name = "Experiment1_system_prompt"
        
    elif experiment_id == 'experiment_2':
        prompt_template_name = "Experiment2_system_prompt"
        
    elif experiment_id == 'experiment_3':
        prompt_template_name = "Experiment3_system_prompt"
        
    elif experiment_id == 'experiment_4':
        prompt_template_name = "Experiment4_system_prompt"
        
    system_prompt = promptsyaml.get(prompt_template_name, None)
    run_experiment(system_prompt,user_prompt,mode,experiment_id)
    
#python3 LLama_as_detector.py experiment_4 calibration
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py 'experiment_1" "calibration")
        sys.exit(1)
        
    experiment_id = sys.argv[1]
    mode = sys.argv[2] #mode = calibration, mode = testing
    if experiment_id not in ("experiment_1","experiment_2","experiment_3","experiment_4"):
        print("Invalid experiment number")
        print("Usage: script.py 'experiment_1" "calibration")
        sys.exit(1)
        
    if mode not in ("calibration","testing"):
        print("Invalid mode")
        print("Usage: script.py 'experiment_1" "calibration")
        sys.exit(1)
        
    start_time = time.time()
    loggingfile = f"loggings_detector.txt"
    sys.stdout = open(loggingfile, "w")
    tokenizer, model = load_model()
    print("Successfully loaded model!")
    
    do_classify(experiment_id,mode)
    print(f"Took {(time.time()-start_time)/60} min")
    print("skipped : ",skipped_set)
    with open("skipped_subsettest.txt","w") as file:
        file.write("\n".join(skipped_set))