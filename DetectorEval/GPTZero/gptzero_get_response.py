# import requests

# url = "https://api.gptzero.me/v2/model-versions/ai-scan"
# headers = {
#     "Accept": "application/json"
# }

# response = requests.get(url, headers=headers)

# if response.status_code == 200:
#     output = response.json() 
#     for out in output:
#         print(out)
# else:
#     print(f"Error: {response.status_code}, {response.text}")
# #Latest = 2025-01-09-base

import requests
import json
import os
import sys

sys.stdout= open("loggings.txt","w")

url = "https://api.gptzero.me/v2/predict/files"
headers = {
    "Accept": "application/json",
    "x-api-key": "b4d94c480a0c413da2d3c0d980925e1a"  # Replace with your actual API key
}                 

data = {
    # "version": "2025-01-09-base",  # Replace with the correct value
    "modelVersion": "2025-01-09-base",  # Replace with the correct value
    # "apiVersion": "your_api_version",  # Replace with the correct value
}

def load_paths():
    all_paths = []
    conf_list = [ "acl_2017/","conll_2016/",
                  "iclr_2017/",
                  "nips_2013-2017/2013/","nips_2013-2017/2014/","nips_2013-2017/2015/",
                "nips_2013-2017/2016/","nips_2013-2017/2017/"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
    cleandata_path = os.path.join(repo_root,"Data_Preprocessing","cleandata")
    
    folders = ['dev','test','train']
    for each_conf in conf_list:
        data_dir = os.path.join(cleandata_path,each_conf)#../data/acl_2017/
        for folder in folders:
            folderpath = os.path.join(data_dir,folder) #../data/acl_2017/dev
            models= os.listdir(folderpath)#"llama,gpt4o,reviews"
            for each_model in models:
                model_path = os.path.join(folderpath,each_model)#../data/acl_2017/dev/gpt_4o_latest
                if each_model == "reviews": #human written reviews has no levels
                    txt_files = os.listdir(model_path)#../data/acl_2017/dev/reviews
                    for each_txt in txt_files:
                        all_paths.append(os.path.join(model_path,f"{each_txt}"))
                else:
                    levels = os.listdir(model_path)#level1,level2,...
                    for level in levels:
                        levelpath =  os.path.join(model_path,level)
                        txt_files = os.listdir(levelpath)
                        for each_txt in txt_files:
                            all_paths.append(os.path.join(levelpath,f"{each_txt}"))
    
    conf_counts = {conf: sum(1 for path in all_paths if conf in path) for conf in conf_list}
    # print(conf_counts)
    return all_paths

def generate_reponse():
    all_paths = load_paths()
    # print(all_paths[:10])
    # print(len(all_paths))
    for each_path in all_paths:
        output_addr = each_path.replace("Data_Preprocessing/cleandata","DetectorEval/GPTZero/Responses").replace('.txt','.json')
        dir_path = os.path.dirname(output_addr)
        os.makedirs(dir_path, exist_ok=True) 
        
        files = {
            "files": open(each_path, "rb"),
        }
        print(f"Processing {each_path}")
        
        response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code == 200:
            output = response.json() # Print the JSON response
            
            with open(output_addr, 'w', encoding='utf-8') as json_file:
                json.dump(output, json_file, ensure_ascii=False, indent=4)
            print("Written :",output_addr)
        else:
            print(f"Error: {response.status_code}, {response.text}")
            


if __name__=="__main__":
    generate_reponse()