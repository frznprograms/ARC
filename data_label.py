from pydantic import BaseModel
from typing import List
import json
import os
import pandas as pd
from openai import OpenAI
import glob

class MessageInfo(BaseModel):
    spam: str
    advertisement: str
    irrelevant_content: str
    non_visitor_rant: str
    toxicity:str 

class Messages(BaseModel):
    messages: List[MessageInfo]

class BatchProcess():
    def __init__(self, df:pd.DataFrame, start_index: int, batch_start_number:int, system_message:str, client:OpenAI):
        """ 
        prepares raw jd from df, sends it to llms and writes outputs to json
        
        """
        self.df = df
        self.start_index = start_index #index of the dataframe to START
        self.batch_start_number = batch_start_number #batch number to START
        self.system_message = system_message
        self.client = client

    def process(self, no_of_batches:int, batch_size:int) -> None:
        #prepare the messages in the batch
        for i in range(no_of_batches):
            print(f"Processing batch {i}...")
            
            messages, row_ids, checkpoint = self.prepare_llm_inputs(self.df, self.start_index, batch_size) #process before llm
            llm_structured_output = self.get_structured_output_from_llm(self.system_message, messages) #pass to llm, get responses

            print("Processing row_ids...", row_ids)

            if len(llm_structured_output.messages) != len(row_ids):
                print(messages)
                print(llm_structured_output.messages)
                raise ValueError(f"Error: Expected batch size of {batch_size}, but got jobids: {len(row_ids)} and llmoutputs: {len(llm_structured_output.messages)}.")

            batch_jds = self.append_jb_ids(row_ids, llm_structured_output) #process output, append jd ids to the response

            self.write_to_file(batch_jds, checkpoint, self.batch_start_number) #write to json for the batch
            
            self.start_index = checkpoint

            self.batch_start_number += 1
        


       
    # need to iterate through the rows in the df for that batch from the start checkpoint, 
    def prepare_llm_inputs(self, df: pd.DataFrame, checkpoint:int, batch_size:int) -> tuple[str, List[int], int]:
        """ 
        This function iterates through the rows in the dataframe in that batch and prepares multiple raw JDs into one message/string for the llm
        returns the updated checkpoint, input prompt which is a string and the list of their job ids for addition later
        """
        messages =""
        job_ids = []
        counter = 1
        for i in range(checkpoint, checkpoint+batch_size):
            job_id = int(df.iloc[i]['id'])
            raw_jd = df.iloc[i]['user_message'].strip()
            job_ids.append(job_id)
            messages += f"<Job {counter}>\n{raw_jd}\n</Job {counter}>\n\n"
            counter+=1
        # Check if the number of job ids does not match the batch size
        if len(job_ids) != batch_size:
            
            raise ValueError(f"Error: Expected batch size of {batch_size}, but got {len(job_ids)}.")

        return (messages, job_ids, checkpoint + batch_size)
    

            
    def get_structured_output_from_llm(self, system_message: str, user_message:str) -> Messages:
        """ 
        sends the messages to gemini, returns a Messages object
        """
        response = self.client.beta.chat.completions.parse(
            model="gemini-2.0-flash-lite",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            response_format=Messages,
        )
        return response.choices[0].message.parsed
    
    # we need to append the job ids to each message, saves output to json file
    def append_jb_ids(self, job_ids:List, messages_list: Messages) -> List:
        batch_jds = []
        for idx, message in enumerate(messages_list.messages):
            temp = message.model_dump()
            #print(temp, idx)
            temp['job_id'] = job_ids[idx]
            batch_jds.append(temp)

        return batch_jds
    
    def write_to_file(self, batch_jds:List, row_checkpoint, batch_no) -> None:
            row_checkpoint -= 1
            dir_name = "extracted_labels"
            os.makedirs(dir_name, exist_ok=True)

            filename = f"{dir_name}/batch_{batch_no}_row_{row_checkpoint}_extracted_labels.json"

            with open(filename, "w") as f:
                json.dump(batch_jds, f, indent=4) 


system_message = '''
  <task>
    Analyze Google reviews to gauge review quality by detecting problematic content.
  </task>
  
  <detection_criteria>
    <spam>Detect reviews that are fake, duplicate, or artificially generated</spam>
    <advertisement>Detect reviews that primarily promote other businesses or services</advertisement>
    <irrelevant_content>Detect reviews that don't relate to the actual business or location</irrelevant_content>
    <non_visitor_rant>Detect rants from users who likely never visited the location</non_visitor_rant>
    <toxicity>Detect reviews containing rudeness, racism, hate speech, harassment, or other toxic behavior</toxicity>
  </detection_criteria>
  
  <labeling_instructions>
    Label each detection category with 1 if the issue is present, 0 if not present.
  </labeling_instructions>
  
  <input_format>
    Review text will be provided for analysis.
  </input_format>
  
  <response_format>
    {
      "spam": 0 or 1,
      "advertisement": 0 or 1, 
      "irrelevant_content": 0 or 1,
      "non_visitor_rant": 0 or 1,
      "toxicity": 0 or 1
    }
  </response_format>
  
  <example>
    Input: "This place is terrible! I heard from my friend it's dirty and overpriced. Never going there!"
    Output: {"spam": 0, "advertisement": 0, "irrelevant_content": 0, "non_visitor_rant": 1, "toxicity": 0}
  </example>
'''


if __name__ == "__main__":

    """
    this script loads in the preprocessed user messages and gets the LLM to generate labels for it, it also combines the json files into a csv
    """

    # load in the dataset to be labelled
    df = pd.read_csv("data/preprocessed_reviews.csv")

    # shuffle and sample
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # get the client
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # initialise the batch process object to label the data
    batch = BatchProcess(shuffled_df, 0, 0, system_message, client)

    # start the labelling, 500 batches of 20
    batch.process(500, 20)

    # get all json files in the folder
    json_files = glob.glob("extracted_labels/*.json")

    combined_data = []

    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                combined_data.extend(data)   # merge lists
            else:
                combined_data.append(data)   # append objects

    # save into one file
    with open("combined.json", "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    # conver this to a dataframe
    combined_data_df = pd.DataFrame(combined_data)
    # rename the column 
    combined_data_df = combined_data_df.rename(columns={"job_id": "id"})

    # incase we didnt finish labelling all the data, do a left join with the original dataset
    labeled_final = pd.merge(combined_data_df, shuffled_df, on="id", how="left")

    # keep only the relevant columns
    labeled_final = labeled_final[[
        "user_message", 
        "spam", 
        "advertisement", 
        "irrelevant_content", 
        "non_visitor_rant", 
        "toxicity"
    ]]

    # rename them for clarity
    df = df.rename(columns={
        "user_message": "text",
        "irrelevant_content": "relevance",
        "non_visitor_rant": "rant"
    })


    # save the final labelled dataset
    labeled_final.to_csv("data/real_review_dataset_final_cleaned.csv", index=False, encoding="utf-8")