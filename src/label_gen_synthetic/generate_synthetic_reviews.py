#!/usr/bin/env python3
"""
Synthetic Review Data Generator

This script generates synthetic Google review data using LLM (Gemini) for training classification models.
It creates a comprehensive dataset with 5 classification types:
- Spam
- Advertisement  
- Irrelevant Content
- Non-visitor Rant
- Toxicity

The script includes active saving functionality to prevent data loss and generates a final cleaned dataset.

Usage:
    python generate_synthetic_reviews.py

Requirements:
    - OpenAI library
    - pandas
    - python-dotenv
    - API key in .env file (OPENAI_API_KEY)
"""

import pandas as pd
import json
import random
import time
import os
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv


class SyntheticReviewGenerator:
    """Generator class for creating synthetic Google reviews"""
    
    def __init__(self, client: OpenAI):
        self.client = client
        
        # Business types and names for random selection
        self.business_types = {
            "restaurants": ["McDonald's", "Olive Garden", "Local Diner", "Pizza Palace", "Sushi Express", 
                          "Taco Bell", "Burger King", "Starbucks", "Subway", "KFC"],
            "hotels": ["Holiday Inn", "Marriott", "Best Western", "Motel 6", "Hilton", 
                      "Hampton Inn", "Days Inn", "Super 8", "La Quinta", "Comfort Inn"],
            "retail": ["Walmart", "Target", "Best Buy", "Home Depot", "Macy's", 
                      "Costco", "CVS Pharmacy", "Walgreens", "GameStop", "Barnes & Noble"],
            "services": ["Hair Salon Plus", "Quick Lube", "Dental Care Center", "Auto Repair Shop", 
                        "Dry Cleaners", "Nail Studio", "Fitness Center", "Pet Grooming", 
                        "Locksmith Services", "Plumbing Express"],
            "entertainment": ["Movie Theater", "Bowling Alley", "Mini Golf", "Arcade Zone", 
                            "Escape Room", "Laser Tag Arena", "Ice Rink", "Pool Hall", 
                            "Comedy Club", "Concert Venue"]
        }
        
        self.categories = {
            "restaurants": ["Restaurant", "Fast food restaurant", "Pizza restaurant", "Italian restaurant", 
                          "Mexican restaurant", "Chinese restaurant", "Cafe", "Bar & grill", 
                          "Buffet restaurant", "Seafood restaurant"],
            "hotels": ["Hotel", "Motel", "Bed & breakfast", "Resort hotel", "Extended stay hotel", 
                      "Inn", "Lodge", "Hostel", "Boutique hotel", "Airport hotel"],
            "retail": ["Department store", "Electronics store", "Grocery store", "Pharmacy", 
                      "Clothing store", "Hardware store", "Bookstore", "Toy store", 
                      "Sporting goods store", "Home goods store"],
            "services": ["Hair salon", "Auto repair shop", "Dental clinic", "Veterinary clinic", 
                        "Dry cleaner", "Fitness center", "Spa", "Locksmith", 
                        "Plumbing service", "HVAC contractor"],
            "entertainment": ["Movie theater", "Bowling alley", "Amusement center", "Night club", 
                            "Concert hall", "Sports bar", "Recreation center", "Gaming cafe", 
                            "Karaoke bar", "Dance studio"]
        }
        
        self.descriptions = [
            "Family-owned business serving the community for over 20 years",
            "Modern facility with state-of-the-art equipment and friendly staff",
            "Convenient location with ample parking and easy access",
            "Offering quality products and services at competitive prices",
            "Committed to customer satisfaction and excellence",
            "Professional team dedicated to meeting your needs",
            "Clean, comfortable environment with attention to detail",
            "Established business with a reputation for reliability",
            "Full-service facility providing comprehensive solutions",
            "Local favorite known for quality and value",
            None  # Some businesses don't have descriptions
        ]

    def get_random_business_info(self) -> Dict:
        """Get random business information"""
        business_type = random.choice(list(self.business_types.keys()))
        business_name = random.choice(self.business_types[business_type])
        category = random.choice(self.categories[business_type])
        description = random.choice(self.descriptions)
        return {
            "business_name": business_name,
            "category": category,
            "description": description
        }


def generate_reviews_with_llm(generator: SyntheticReviewGenerator, classification_type: str, 
                             num_reviews: int, batch_size: int = 10) -> List[Dict]:
    """Generate reviews using LLM in batches"""
    all_reviews = []
    
    # Define prompts for each classification type
    prompts = {
        "spam": """Generate fake/spam Google reviews that are clearly artificial. These should include:
- Overly positive generic reviews with no specific details
- Template-like reviews with poor grammar
- Fake enthusiastic reviews with excessive capitalization and exclamation marks
- Generic phrases like "good service good price" repeated
- Reviews that sound like they were written by bots or paid reviewers""",
        
        "advertisement": """Generate Google reviews that contain advertisements for other businesses. These should:
- Start with a brief comment about the current business
- Then promote a completely different business with specific details
- Include promotional codes, discounts, or special offers
- Mention specific business names, addresses, or contact information
- Be obviously trying to advertise something other than the business being reviewed""",
        
        "irrelevant_content": """Generate Google reviews that go completely off-topic. These should:
- Start with a brief comment about the business
- Then discuss completely unrelated topics like personal stories, random facts, current events
- Talk about hobbies, travel plans, pets, movies, books, weather, etc.
- Have no connection to the actual business or service being reviewed
- Be more like social media posts than business reviews""",
        
        "non_visitor_rant": """Generate Google reviews from people who clearly never visited the business. These should:
- Include phrases like "I heard from my friend", "someone told me", "read online"
- Be negative rants based on second-hand information
- Show no evidence of actual personal experience
- Make vague complaints without specific details
- Sound like rumors or gossip rather than firsthand experience""",
        
        "toxicity": """Generate Google reviews that are toxic and inappropriate. These should include:
- Rude, disrespectful language and personal attacks
- Excessive anger and aggressive tone
- Name-calling and insults toward staff or management
- Inappropriate language and unprofessional behavior
- Threats, harassment, or extremely hostile content
- Reviews that clearly violate platform guidelines"""
    }
    
    # Get example business data for the prompt
    restaurant_examples = ", ".join(generator.business_types["restaurants"][:5])
    hotel_examples = ", ".join(generator.business_types["hotels"][:5])
    retail_examples = ", ".join(generator.business_types["retail"][:5])
    service_examples = ", ".join(generator.business_types["services"][:5])
    entertainment_examples = ", ".join(generator.business_types["entertainment"][:5])
    
    restaurant_categories = ", ".join(generator.categories["restaurants"][:5])
    hotel_categories = ", ".join(generator.categories["hotels"][:5])
    retail_categories = ", ".join(generator.categories["retail"][:5])
    service_categories = ", ".join(generator.categories["services"][:5])
    entertainment_categories = ", ".join(generator.categories["entertainment"][:5])
    
    description_examples = "; ".join([desc for desc in generator.descriptions[:5] if desc is not None])
    
    system_prompt = f"""You are generating synthetic training data for a review classification system. 

{prompts[classification_type]}

For each review, provide the output in this EXACT JSON format:
{{
    "business_name": "creative business name (be creative, don't just copy examples)",
    "category": "appropriate business category", 
    "description": "business description or null",
    "review": "the generated review text",
    "rating": number between 1-5,
    "response": null
}}

BUSINESS NAME EXAMPLES (create similar but unique names):
- Restaurants: {restaurant_examples}
- Hotels: {hotel_examples}  
- Retail: {retail_examples}
- Services: {service_examples}
- Entertainment: {entertainment_examples}

CATEGORY EXAMPLES (use these or create similar):
- Restaurant categories: {restaurant_categories}
- Hotel categories: {hotel_categories}
- Retail categories: {retail_categories}
- Service categories: {service_categories}
- Entertainment categories: {entertainment_categories}

DESCRIPTION EXAMPLES: {description_examples}

Generate {batch_size} different reviews. Make each review unique and realistic. Create diverse business names and vary business types, locations, and review content significantly. Be creative with business names while keeping them realistic."""

    print(f"Generating {num_reviews} {classification_type} reviews...")
    
    for batch_start in range(0, num_reviews, batch_size):
        current_batch_size = min(batch_size, num_reviews - batch_start)
        
        try:
            response = generator.client.chat.completions.create(
                model="gemini-2.0-flash-lite",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {current_batch_size} {classification_type} reviews in the specified JSON format."}
                ],
                temperature=0.9  # High temperature for more variety
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Try to extract JSON objects from the response
            import re
            
            # Find all JSON objects in the response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    review_data = json.loads(json_str)
                    
                    # Add classification labels
                    review_data["label_spam"] = 1 if classification_type == "spam" else 0
                    review_data["label_advertisement"] = 1 if classification_type == "advertisement" else 0
                    review_data["label_irrelevant_content"] = 1 if classification_type == "irrelevant_content" else 0
                    review_data["label_non_visitor_rant"] = 1 if classification_type == "non_visitor_rant" else 0
                    review_data["label_toxicity"] = 1 if classification_type == "toxicity" else 0
                    
                    all_reviews.append(review_data)
                    
                except json.JSONDecodeError:
                    continue
            
            print(f"  Generated batch {batch_start//batch_size + 1}, total so far: {len(all_reviews)}")
            
        except Exception as e:
            print(f"Error generating batch {batch_start//batch_size + 1}: {e}")
            # Fallback to manual generation if LLM fails
            business_info = generator.get_random_business_info()
            fallback_review = {
                **business_info,
                "review": f"Fallback {classification_type} review for {business_info['business_name']}",
                "rating": random.randint(1, 5),
                "response": None,
                "label_spam": 1 if classification_type == "spam" else 0,
                "label_advertisement": 1 if classification_type == "advertisement" else 0,
                "label_irrelevant_content": 1 if classification_type == "irrelevant_content" else 0,
                "label_non_visitor_rant": 1 if classification_type == "non_visitor_rant" else 0,
                "label_toxicity": 1 if classification_type == "toxicity" else 0
            }
            all_reviews.append(fallback_review)
    
    return all_reviews[:num_reviews]  # Ensure we don't exceed requested number


def check_current_status():
    """Check current status by examining existing files"""
    status = {}
    files_to_check = {
        "advertisement": ["synthetic_reviews_advertisement_7200.csv", "synthetic_reviews_advertisement_3000.csv"],
        "irrelevant_content": ["synthetic_reviews_irrelevant_content_7200.csv", "synthetic_reviews_irrelevant_content_3000.csv"],
        "spam": ["synthetic_reviews_spam_7200.csv", "synthetic_reviews_spam_3000.csv"],
        "non_visitor_rant": ["synthetic_reviews_non_visitor_rant_3000.csv"],
        "toxicity": ["synthetic_reviews_toxicity_3000.csv"]
    }
    
    for class_type, filenames in files_to_check.items():
        current_count = 0
        for filename in filenames:
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename)
                    current_count = max(current_count, len(df))
                    print(f"Found {filename} with {len(df)} reviews")
                except:
                    pass
        status[class_type] = current_count
    
    return status


def generate_with_active_saving(generator: SyntheticReviewGenerator, class_type: str, 
                               num_reviews: int, batch_size: int = 20, delay: int = 3):
    """Generate reviews with active saving after each batch"""
    
    # Load existing data if available
    possible_files = [
        f"synthetic_reviews_{class_type}_3000.csv",
        f"synthetic_reviews_{class_type}_7200.csv"
    ]
    
    existing_reviews = []
    for filename in possible_files:
        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename)
                existing_reviews = existing_df.to_dict('records')
                print(f"  Loaded {len(existing_reviews)} existing reviews from {filename}")
                break
            except Exception as e:
                print(f"  Could not load {filename}: {e}")
    
    all_reviews = existing_reviews.copy()
    new_reviews_generated = 0
    target_total = len(existing_reviews) + num_reviews
    
    print(f"  Target: {target_total} total reviews ({len(existing_reviews)} existing + {num_reviews} new)")
    
    batches_needed = (num_reviews + batch_size - 1) // batch_size
    
    for batch_num in range(batches_needed):
        current_batch_size = min(batch_size, num_reviews - new_reviews_generated)
        
        print(f"  Batch {batch_num + 1}/{batches_needed}: Generating {current_batch_size} reviews...")
        
        # Add delay between requests
        if batch_num > 0:
            print(f"  Waiting {delay} seconds...")
            time.sleep(delay)
        
        try:
            batch_reviews = generate_reviews_with_llm(generator, class_type, current_batch_size, 
                                                    batch_size=current_batch_size)
            all_reviews.extend(batch_reviews)
            new_reviews_generated += len(batch_reviews)
            
            print(f"  Generated {len(batch_reviews)} reviews (Total: {len(all_reviews)}, New: {new_reviews_generated})")
            
            # ACTIVE SAVING: Save progress after each batch
            temp_filename = f"synthetic_reviews_{class_type}_progress.csv"
            final_filename = f"synthetic_reviews_{class_type}_3000.csv"
            
            try:
                progress_df = pd.DataFrame(all_reviews)
                progress_df.to_csv(temp_filename, index=False)
                print(f"  Progress saved to {temp_filename}")
                
                # If we've reached our target, save to final filename
                if len(all_reviews) >= target_total:
                    progress_df.to_csv(final_filename, index=False)
                    print(f"  Final dataset saved to {final_filename}")
                    
            except Exception as save_error:
                print(f"  Could not save progress: {save_error}")
            
        except Exception as e:
            print(f"  Error in batch {batch_num + 1}: {e}")
            print(f"  Waiting 10 seconds before retrying...")
            time.sleep(10)
            
            # Retry once
            try:
                batch_reviews = generate_reviews_with_llm(generator, class_type, current_batch_size, 
                                                        batch_size=current_batch_size)
                all_reviews.extend(batch_reviews)
                new_reviews_generated += len(batch_reviews)
                print(f"  Retry successful: {len(batch_reviews)} reviews")
                
                # Save after retry
                temp_filename = f"synthetic_reviews_{class_type}_progress.csv"
                try:
                    progress_df = pd.DataFrame(all_reviews)
                    progress_df.to_csv(temp_filename, index=False)
                    print(f"  Progress saved after retry")
                except Exception as save_error:
                    print(f"  Could not save after retry: {save_error}")
                    
            except Exception as e2:
                print(f"  Retry failed: {e2}")
                print(f"  Skipping this batch...")
                continue
        
        # Stop if we've generated enough
        if new_reviews_generated >= num_reviews:
            break
    
    return all_reviews, new_reviews_generated


def clean_and_combine_datasets():
    """Clean and combine all 5 CSV files into final dataset"""
    print("🧹 CLEANING AND COMBINING DATASETS")
    print("=" * 50)
    
    # Define the 5 CSV files
    csv_files = {
        "advertisement": "synthetic_reviews_advertisement_3000.csv",
        "irrelevant_content": "synthetic_reviews_irrelevant_content_3000.csv", 
        "non_visitor_rant": "synthetic_reviews_non_visitor_rant_3000.csv",
        "toxicity": "synthetic_reviews_toxicity_3000.csv",
        "spam": "synthetic_reviews_spam_7200.csv"  # This one might be 7200
    }
    
    # Load and clean each dataset
    cleaned_datasets = {}
    total_removed = 0
    total_kept = 0
    
    for class_type, filename in csv_files.items():
        if os.path.exists(filename):
            print(f"\n📁 Processing {filename}...")
            
            # Load the dataset
            df = pd.read_csv(filename)
            print(f"   Original size: {len(df)} reviews")
            
            # Check for fallback reviews (reviews that contain "Fallback" in the review text)
            fallback_mask = df['review'].str.contains('Fallback', case=False, na=False)
            fallback_count = fallback_mask.sum()
            
            if fallback_count > 0:
                print(f"   Found {fallback_count} fallback reviews")
                # Remove fallback reviews
                df_cleaned = df[~fallback_mask].copy()
                print(f"   After cleaning: {len(df_cleaned)} reviews")
            else:
                print(f"   No fallback reviews found")
                df_cleaned = df.copy()
            
            # Store the cleaned dataset
            cleaned_datasets[class_type] = df_cleaned
            total_removed += fallback_count
            total_kept += len(df_cleaned)
            
            print(f"   ✅ Cleaned {class_type} dataset ready")
            
        else:
            print(f"❌ File not found: {filename}")
    
    print(f"\n📊 CLEANING SUMMARY:")
    print(f"Total fallback reviews removed: {total_removed}")
    print(f"Total clean reviews kept: {total_kept}")
    
    # Combine all cleaned datasets
    print(f"\n🔗 COMBINING DATASETS...")
    combined_reviews = []
    
    for class_type, df_cleaned in cleaned_datasets.items():
        print(f"Adding {len(df_cleaned)} {class_type} reviews...")
        combined_reviews.extend(df_cleaned.to_dict('records'))
    
    # Create final combined DataFrame
    final_df = pd.DataFrame(combined_reviews)
    
    # Shuffle the dataset for better training
    random.seed(42)  # For reproducibility
    shuffled_indices = list(range(len(final_df)))
    random.shuffle(shuffled_indices)
    final_df = final_df.iloc[shuffled_indices].reset_index(drop=True)
    
    print(f"\n✅ Combined dataset created with {len(final_df)} total reviews")
    
    # Check label distribution
    print(f"\n📈 FINAL LABEL DISTRIBUTION:")
    label_cols = ['label_spam', 'label_advertisement', 'label_irrelevant_content', 'label_non_visitor_rant', 'label_toxicity']
    for label_col in label_cols:
        if label_col in final_df.columns:
            count = final_df[label_col].sum()
            print(f"{label_col.replace('label_', '').title()}: {count}")
    
    # Create user_message format if it doesn't exist
    if 'user_message' not in final_df.columns:
        print(f"\n📝 Creating user_message format...")
        final_df["user_message"] = (
            "Business Name: " + final_df["business_name"].astype(str).fillna("N/A") + "\n" +
            "Category: " + final_df["category"].astype(str).fillna("N/A") + "\n" +
            "Description: " + final_df["description"].astype(str).fillna("N/A") + "\n" +
            "Review: " + final_df["review"].astype(str).fillna("N/A") + "\n" +
            "Rating: " + final_df["rating"].astype(str) + "\n" +
            "Response: " + final_df["response"].astype(str).fillna("N/A")
        )
    
    # Create final dataset with only the required 6 columns and rename them
    print(f"\n📝 Creating final dataset with 6 columns...")
    final_clean_df = pd.DataFrame({
        'text': final_df['user_message'],
        'spam': final_df['label_spam'],
        'advertisement': final_df['label_advertisement'], 
        'relevance': final_df['label_irrelevant_content'],  # irrelevant_content -> relevance
        'rant': final_df['label_non_visitor_rant'],
        'toxicity': final_df['label_toxicity']
    })
    
    # Save the final combined dataset
    final_filename = "synthetic_review_dataset_final_cleaned.csv"
    final_clean_df.to_csv(final_filename, index=False)
    print(f"\n🎉 FINAL DATASET SAVED!")
    print(f"Filename: {final_filename}")
    print(f"Total reviews: {len(final_clean_df)}")
    print(f"Columns: {list(final_clean_df.columns)}")
    
    # Show final label distribution with new column names
    print(f"\n📈 FINAL LABEL DISTRIBUTION (New Column Names):")
    for col in ['spam', 'advertisement', 'relevance', 'rant', 'toxicity']:
        count = final_clean_df[col].sum()
        percentage = (count / len(final_clean_df)) * 100
        print(f"{col.title():15}: {count:,} ({percentage:.1f}%)")
    
    return final_clean_df


def main():
    """Main function to generate synthetic review dataset"""
    print("🚀 SYNTHETIC REVIEW DATA GENERATOR")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv(override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    print("✅ API key loaded successfully")
    
    # Initialize OpenAI client for Gemini
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    # Initialize generator
    generator = SyntheticReviewGenerator(client)
    print("✅ Generator initialized")
    
    # Check current status
    print("\n📊 CHECKING CURRENT STATUS")
    current_status = check_current_status()
    print("\nCurrent status:")
    for class_type, count in current_status.items():
        needed = max(0, 3000 - count)
        print(f"- {class_type.title()}: {count} -> need {needed} more")
    
    # Define what needs to be generated
    generation_plan = {}
    for class_type, current_count in current_status.items():
        needed = max(0, 3000 - current_count)
        if needed > 0:
            generation_plan[class_type] = needed
    
    print(f"\nGeneration plan: {generation_plan}")
    
    # Generate missing data if needed
    if generation_plan:
        print(f"\n🔄 STARTING GENERATION WITH ACTIVE SAVING")
        results = {}
        
        for class_type, num_needed in generation_plan.items():
            if num_needed > 0:
                print(f"\n=== Generating {num_needed} {class_type} reviews ===")
                
                try:
                    final_reviews, new_count = generate_with_active_saving(
                        generator, class_type, num_needed, batch_size=20, delay=3
                    )
                    
                    print(f"Generated {new_count} new {class_type} reviews")
                    print(f"Total {class_type} reviews: {len(final_reviews)}")
                    
                    # Final save
                    final_filename = f"synthetic_reviews_{class_type}_3000.csv"
                    final_df = pd.DataFrame(final_reviews)
                    final_df.to_csv(final_filename, index=False)
                    print(f"Final dataset saved to '{final_filename}'")
                    
                    # Clean up progress file
                    progress_filename = f"synthetic_reviews_{class_type}_progress.csv"
                    if os.path.exists(progress_filename):
                        try:
                            os.remove(progress_filename)
                            print(f"Cleaned up progress file")
                        except:
                            pass
                    
                    results[class_type] = len(final_reviews)
                    
                except Exception as e:
                    print(f"Failed to generate {class_type} reviews: {e}")
        
        print(f"\nGeneration complete! Results: {results}")
    else:
        print("✅ All datasets already exist with sufficient data")
    
    # Clean and combine datasets
    print(f"\n🧹 CLEANING AND COMBINING DATASETS")
    final_dataset = clean_and_combine_datasets()
    
    # Final verification
    print(f"\n🔍 FINAL VERIFICATION")
    fallback_check = final_dataset['text'].str.contains('Fallback', case=False, na=False).sum()
    print(f"Fallback reviews remaining: {fallback_check}")
    
    # Verify each review has exactly one label
    label_cols = ['spam', 'advertisement', 'relevance', 'rant', 'toxicity']
    final_dataset['total_labels'] = final_dataset[label_cols].sum(axis=1)
    single_label_count = (final_dataset['total_labels'] == 1).sum()
    print(f"Reviews with exactly 1 label: {single_label_count:,}/{len(final_dataset):,}")
    print(f"Validation status: {'PASSED' if single_label_count == len(final_dataset) else 'FAILED'}")
    
    print(f"\n🎉 SUCCESS!")
    print(f"Final dataset: synthetic_review_dataset_final_cleaned.csv")
    print(f"Total reviews: {len(final_dataset):,}")
    print(f"Ready for training! 🚀")


if __name__ == "__main__":
    main()
