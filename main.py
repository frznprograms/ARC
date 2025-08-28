from openai import OpenAI
import json
import pandas as pd
import os
from dotenv import load_dotenv
from data_label import BatchProcess

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

df = pd.read_csv("data/jobs_clean.csv")
len(df)

system_message = '''
You are to summarize the following features concisely from each job description: "Core Responsibilities, Required Skills, Educational Requirements, Experience Level, Preferred Qualifications, Compensation and Benefits.

Please return the output in the following format:

{
  "Core Responsibilities": "Summarize the core responsibilites of the job here",
  "Required Skills": "Summarize the required skills here",
  "Educational Requirements": "Summarize the educational requirements",
  "Experience Level": "Summarize the experience level in years, if none put N/A",
  "Preferred Qualifications": "Summarize the qualifications required",
  "Compensation and Benefits": "Summarize the monthly or hourly salary, if none put N/A"
}
return the output as a string, not markdown

'''

batch_process = BatchProcess(df=df, start_index=0, batch_start_number=0, system_message=system_message)