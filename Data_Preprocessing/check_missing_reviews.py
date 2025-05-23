import re
import os
import json
import sys
ai_generation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../AI_generation"))
# Add AI_generation to sys.path
sys.path.append(ai_generation_path)

from utils import *


def check_reviews():

    models = ['meta-llama-Llama-3.3-70B-Instruct'] #Add folders to verify here        
    for each_conf in data_dir:
        print(f"\n\nProcessing :{each_conf}")
        for f in folders:
            devpath = os.path.join(each_conf,f) #../data/acl_2017/dev
            print(devpath)
            for model in models:
                print(f"Model : {model}")
                modelpath = os.path.join(devpath,model) #../data/acl_2017/dev/meta-llama-Llama-3.3-70B-Instruct
                for each_level in os.listdir(modelpath):  #level1, level2, level3, level4
                    levelpath = os.path.join(modelpath,each_level) #../data/acl_2017/dev/meta-llama-Llama-3.3-70B-Instruct/level1
                    count_review = 0
                    for each_json in os.listdir(levelpath):#list of jsonfiles
                        json_name =each_json.split('.')[0]
                        inputpath = os.path.join(levelpath,each_json)#....file.json
                        
                        with open(inputpath, "r", encoding='utf-8') as file:
                            data = json.load(file)
                        comments = [review["comments"] for review in data["reviews"]]
                        count_review +=len(comments)
                        for each_review in comments:
                            if not each_review:
                                print(f"{inputpath} contains empty reviews")
                    print(f"#reviews in {each_level}: {count_review}")

    
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

    check_reviews() #Function to check if any generated reviews = ""  and to count total reviews