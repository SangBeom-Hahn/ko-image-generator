import os
import pandas as pd
import json
from tqdm import tqdm

ROOT_PATH = "/workspace/data/text_data/"
DATA_PATH = "/workspace/data/text_data/common"
SOCIAL_DATA_PATH = "/workspace/data/text_data/social"
TECH_DATA_PATH = "/workspace/data/text_data/tech"

social_file_name = ["1113_social_train_set_1210529.csv","1113_social_valid_set_151316.csv"]
tech_file_name = ["1113_tech_train_set_1195228.csv" ,"1113_tech_valid_set_149403.csv"]

common = os.path.join(DATA_PATH, 'common_data.json')
df_common = pd.read_json(common)
df_common.columns = ['ko', 'en']

for social in tqdm(social_file_name) :
    social_path = os.path.join(SOCIAL_DATA_PATH, social)
    df_social = pd.read_csv(social_path)
    df_social = df_social[['ko', 'en']]
    df_common = pd.concat([df_common, df_social])
    
for tech in tqdm(tech_file_name) :
    tech_path = os.path.join(TECH_DATA_PATH, tech) 
    df_tech = pd.read_csv(tech_path)
    df_tech = df_tech[['ko', 'en']]
    df_common = pd.concat([df_common, df_tech])
    
df_common = df_common.reset_index(drop = True)
df_common.to_csv(os.path.join(ROOT_PATH, 'text_data.csv'), index=False)
print(df_common)

df_common = df_common.to_dict(orient='records')

with open(os.path.join(ROOT_PATH, 'text_data.json'), 'w') as f : 
 	json.dump(df_common, f, indent=4)