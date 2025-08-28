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

df = pd.read_csv("data.csv")
len(df)

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

batch_process = BatchProcess(df=df, start_index=0, batch_start_number=0, system_message=system_message)