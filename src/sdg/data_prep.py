import ast
import json

import pandas as pd

if __name__ == "__main__":
    """
    Preprocess review and metadata JSON files into a clean DataFrame for LLM labeling.

    This script does the following:
        1. Loads reviews JSON (line-delimited, up to 100,000 records).
        2. Loads business metadata JSON.
        3. Merges reviews with metadata on 'gmap_id'.
        4. Removes duplicates and unnecessary columns.
        5. Renames columns to standardized names.
        6. Creates a 'user_message' field (combined string prompt for LLMs).
        7. Saves the final DataFrame to CSV.

    Input:
        - data/review-New_York_10.json : JSON lines of reviews
        - data/meta-New_York.json      : JSON lines of metadata

    Output:
        - data/preprocessed_reviews.csv : Cleaned, merged DataFrame
    """
    # -----------------------------
    # Step 1: Load Reviews JSON
    # -----------------------------

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
            count += 1
            # take the first 100,000 datapoints
            if count > 100000:
                break

    # convert the dictionary containing the reviews to a dataframe
    df_reviews = pd.DataFrame(dict_list)
    # drop all the columns that do not contain any reviews
    df_reviews = df_reviews.dropna(subset=["text"]).reset_index(drop=True)

    # -----------------------------
    # Step 2: Load Metadata JSON
    # -----------------------------
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
     
    # -----------------------------
    # Step 3: Merge + Deduplicate
    # -----------------------------

    # merge the meta data back with the reviews data
    merged_df = pd.merge(df_reviews, df_meta, on="gmap_id", how="inner")

    # this removes the duplicates (likely spam)
    merged_df = merged_df.drop_duplicates(subset=["name_x", "text", "gmap_id"])

    # -----------------------------
    # Step 4: Select Final Columns
    # -----------------------------

    # remove the unneccessary columns
    final_columns = ["text", "resp", "description", "category", "name_y", "rating"]
    final_df = merged_df[final_columns]

    # convert the lists to string in the columns if any
    for col in final_df.columns:
        final_df[col] = final_df[col].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

    # Reset index so you get a new incrementing column
    final_df = final_df.reset_index(drop=True)
    final_df.index.name = "id"

    # -----------------------------
    # Step 5: Rename Columns
    # -----------------------------

    final_df = final_df.rename(
        columns={
            "name_y": "Business Name",
            "category": "Category",
            "description": "Description",
            "text": "Review",
            "rating": "Rating",
            "resp": "Response",
        }
    )
    # -----------------------------
    # Step 6: Construct user_message
    # -----------------------------
    final_df["user_message"] = (
        "Business Name: " + final_df["Business Name"].astype(str).fillna("N/A") + "\n"
        "Category: " + final_df["Category"].astype(str).fillna("N/A") + "\n"
        "Description: " + final_df["Description"].astype(str).fillna("N/A") + "\n"
        "Review: " + final_df["Review"].astype(str).fillna("N/A") + "\n"
        "Rating: " + final_df["Rating"].astype(str).fillna("N/A") + "\n"
        "Response: " + final_df["Response"].astype(str).fillna("N/A")
    )
    # -----------------------------
    # Step 7: Save to CSV
    # -----------------------------
    # save to csv if required
    final_df.to_csv("data/preprocessed_reviews.csv", index=True)
