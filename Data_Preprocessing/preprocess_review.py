from ftfy import fix_text
import re
import os
import json
import sys
ai_generation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../AI_generation"))
# Add AI_generation to sys.path
sys.path.append(ai_generation_path)

from utils import *

def remove_markdown(text):
    text = re.sub(r"#+\s*", "", text) # Remove headings (###, ##, #)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # Remove *italic*
    text = re.sub(r"_(.*?)_", r"\1", text)  # Remove _italic_
    text = re.sub(r"\n{2,}", "\n", text)  # Reduce multiple newlines to 1
    text = text.strip() 
    return text


def process_reviews(rev_type):
    output_dir = "./cleandata"
    if rev_type == "human":
        models = ["reviews"]
        human = True
    else:
        models = ['meta-llama-Llama-3.3-70B-Instruct']
        human = False
        
    for each_conf in data_dir:
        print(f"Processing :{each_conf}")
        for f in folders:
            devpath = os.path.join(each_conf,f) #../data/acl_2017/dev
            for model in models:
                modelpath = os.path.join(devpath,model) #../data/acl_2017/dev/meta-llama-Llama-3.3-70B-Instruct
                if not human: #LLM review
                    for each_level in os.listdir(modelpath):  #level1, level2, level3, level4
                        levelpath = os.path.join(modelpath,each_level) #../data/acl_2017/dev/meta-llama-Llama-3.3-70B-Instruct/level1
                        for each_json in os.listdir(levelpath):#list of jsonfiles
                            json_name =each_json.split('.')[0]
                            inputpath = os.path.join(levelpath,each_json)#....file.json
                            
                            with open(inputpath, "r", encoding='utf-8') as file:
                                data = json.load(file)
                            # if "iclr" in each_conf:
                            #     data = preprocess_iclr(data)
                            # Extract comments
                            comments = [review["comments"] for review in data["reviews"]]
                            review_count = 1
                            # print(f"inputpath :{inputpath}")
                            for each_review in comments:

                                cleaned = remove_markdown(each_review)
                                cleaned = fix_text(cleaned)
                                
                                output_path = levelpath.replace("../data", output_dir)
                                os.makedirs(output_path, exist_ok=True)
                                output_path = os.path.join(output_path,json_name+f"_{review_count}"+".txt")
                                # print(f"output_path :{output_path}")
                                with open(output_path,'w', encoding="utf-8") as f:
                                    f.write(cleaned)
                                review_count += 1
                else:
                    for each_json in os.listdir(modelpath):#list of jsonfiles
                        json_name =each_json.split('.')[0]
                        inputpath = os.path.join(modelpath,each_json)#....file.json
                        with open(inputpath, "r", encoding='utf-8') as file:
                            data = json.load(file)
                        if "iclr" in each_conf:
                            data = preprocess_iclr(data)
                        # Extract comments
                        comments = [review["comments"] for review in data["reviews"]]
                        review_count = 1
                        # print(f"inputpath :{inputpath}")
                        for each_review in comments:

                            cleaned = remove_markdown(each_review)
                            cleaned = fix_text(cleaned)
                            
                            output_path = modelpath.replace("../data", output_dir)
                            os.makedirs(output_path, exist_ok=True)
                            output_path = os.path.join(output_path,json_name+f"_{review_count}"+".txt")
                            # print(f"output_path :{output_path}")
                            with open(output_path,'w', encoding="utf-8") as f:
                                f.write(cleaned)
                            review_count += 1
                    
    
if __name__ == "__main__":
    data_dir = [ "../data/acl_2017/",
                 "../data/conll_2016/",
                  "../data/iclr_2017/",
                   "../data/nips_2013-2017/2013/",
                    "../data/nips_2013-2017/2014/",
                     "../data/nips_2013-2017/2015/",
                      "../data/nips_2013-2017/2016/",
                       "../data/nips_2013-2017/2017/"]
    folders = ['dev','test','train']

    process_reviews("llm") #processreviews("human") for processing humanreviews