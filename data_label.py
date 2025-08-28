from pydantic import BaseModel
from typing import List
import json
import os
import pandas as pd
from openai import OpenAI


class MessageInfo(BaseModel):
    Required_Skills: str
    Educational_Requirements: str
    Experience_Level: str
    Preferred_Qualifications: str
    Compensation_and_Benefits:str 

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
            
            messages, job_ids, checkpoint = self.prepare_llm_inputs(self.df, self.start_index, batch_size) #process before llm
            llm_structured_output = self.get_structured_output_from_llm(self.system_message, messages) #pass to llm, get responses

            print("Processing job_ids...", job_ids)

            if len(llm_structured_output.messages) != len(job_ids):
                print(messages)
                print(llm_structured_output.messages)
                raise ValueError(f"Error: Expected batch size of {batch_size}, but got jobids: {len(job_ids)} and llmoutputs: {len(llm_structured_output.messages)}.")

            batch_jds = self.append_jb_ids(job_ids, llm_structured_output) #process output, append jd ids to the response

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
            job_id = int(df.iloc[i]['job_id'])
            raw_jd = df.iloc[i]['description'].strip()
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
            dir_name = "extracted_jds"
            os.makedirs(dir_name, exist_ok=True)

            filename = f"{dir_name}/batch_{batch_no}_row_{row_checkpoint}_extracted_jd.json"

            with open(filename, "w") as f:
                json.dump(batch_jds, f, indent=4) 
