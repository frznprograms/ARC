import json
import pandas as pd
import ast


if __name__ == "__main__":
  """
  This script handles the following processes:
  prepares the downloaded data for the LLM labelling, 
  this involves merging the meta data and the reviews data from two separate json files
  """

  dict_list = []
  count = 0
  with open("data/review-New_York_10.json", "r", encoding="utf-8") as f:
      for line_num, line in enumerate(f, start=1):
          line = line.strip()  # remove leading/trailing whitespace
          if not line:
              continue  # skip empty lines
          try:
              # Try JSON parsing first (double quotes)
              obj = json.loads(line)
          except json.JSONDecodeError:
              try:
                  # Fallback for Python dict strings (single quotes)
                  obj = ast.literal_eval(line)
              except Exception:
                  print(f"Skipping invalid line {line_num}: {line[:100]}...")
                  continue
          dict_list.append(obj)
          count+=1
          # take the first 100,000 datapoints
          if count > 100000:
              break
          
  # convert the dictionary containing the reviews to a dataframe
  df_reviews = pd.DataFrame(dict_list)
  # drop all the columns that do not contain any reviews            
  df_reviews = df_reviews.dropna(subset=['text']).reset_index(drop=True)

  dict_list = []

  with open("data/meta-New_York.json", "r", encoding="utf-8") as f:
      for line_num, line in enumerate(f, start=1):
          line = line.strip()  # remove leading/trailing whitespace
          if not line:
              continue  # skip empty lines
          try:
              # Try JSON parsing first (double quotes)
              obj = json.loads(line)
          except json.JSONDecodeError:
              try:
                  # Fallback for Python dict strings (single quotes)
                  obj = ast.literal_eval(line)
              except Exception:
                  print(f"Skipping invalid line {line_num}: {line[:100]}...")
                  continue
          dict_list.append(obj)

  # create a dataframe containing the meta data
  df_meta = pd.DataFrame(dict_list)
  # merge the meta data back with the reviews data
  merged_df = pd.merge(df_reviews, df_meta, on='gmap_id', how='inner')

  # this removes the duplicates (likely spam)
  merged_df = merged_df.drop_duplicates(subset=['name_x','text','gmap_id'])

  # remove the unneccessary columns
  final_columns = ['text','resp','description','category','name_y', 'rating']
  final_df = merged_df[final_columns]

  # convert the lists to string in the columns if any
  for col in final_df.columns:
      final_df[col] = final_df[col].apply(
          lambda x: ", ".join(x) if isinstance(x, list) else x
      )

  # Reset index so you get a new incrementing column
  final_df = final_df.reset_index(drop=True)
  final_df.index.name = "id"

  final_df = final_df.rename(columns={
      "name_y": "Business Name",
      "category": "Category",
      "description": "Description",
      "text": "Review",
      "rating": "Rating",
      "resp": "Response"
  })

  final_df["user_message"] = (
      "Business Name: " + final_df["Business Name"].astype(str).fillna("N/A") + "\n"
      "Category: " + final_df["Category"].astype(str).fillna("N/A") + "\n"
      "Description: " + final_df["Description"].astype(str).fillna("N/A") + "\n"
      "Review: " + final_df["Review"].astype(str).fillna("N/A") + "\n"
      "Rating: " + final_df["Rating"].astype(str).fillna("N/A") + "\n"
      "Response: " + final_df["Response"].astype(str).fillna("N/A")
  )

  # save to csv if required
  final_df.to_csv("data/preprocessed_reviews.csv", index=True)  

