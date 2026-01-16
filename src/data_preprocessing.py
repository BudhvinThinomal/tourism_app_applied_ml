import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('../models', exist_ok=True)
os.makedirs('../outputs', exist_ok=True)


print("=" * 80)
print("ENHANCED DATA PREPROCESSING")
print("Sri Lanka Tourism Experience Prediction System")
print("=" * 80)



# =============================================================================
# LOAD ALL DATASETS
# =============================================================================

print("\n" + "─" * 80)
print("LOADING ALL DATASETS")
print("─" * 80)

print("\nLoading datasets...")

# Core datasets
reviews_df = pd.read_csv('../data/Destination_Reviews_Final.csv')
accommodation_df = pd.read_csv('../data/Information_for_Accommodation.csv')
places_df = pd.read_csv('../data/Places_for_Travel_Dining.csv')
airbnb_df = pd.read_csv('../data/Airbnb_Plus_Personal_Contacts.csv')
tourist_df = pd.read_csv('../data/Tourist_Data.csv')

# Document dataset sizes
print("\n" + "─" * 65)
print(f"{'Dataset':<45} {'Rows':>10} {'Columns':>8}")
print("─" * 65)
print(f"{'Destination Reviews':<45} {reviews_df.shape[0]:>10,} {reviews_df.shape[1]:>8}")
print(f"{'Accommodation Information':<45} {accommodation_df.shape[0]:>10,} {accommodation_df.shape[1]:>8}")
print(f"{'Places (Restaurants, Spas, etc.)':<45} {places_df.shape[0]:>10,} {places_df.shape[1]:>8}")
print(f"{'Airbnb Listings [NEW]':<45} {airbnb_df.shape[0]:>10,} {airbnb_df.shape[1]:>8}")
print(f"{'Tourist Arrival Statistics [NEW]':<45} {tourist_df.shape[0]:>10,} {tourist_df.shape[1]:>8}")
print("─" * 65)
total_records = sum([df.shape[0] for df in [reviews_df, accommodation_df, places_df, airbnb_df, tourist_df]])
print(f"{'TOTAL RECORDS':<45} {total_records:>10,}")



# =============================================================================
# SENTIMENT ANALYSIS ON REVIEWS
# =============================================================================

print("\n" + "─" * 80)
print("SENTIMENT ANALYSIS")
print("─" * 80)

# Comprehensive tourism-specific sentiment lexicon
positive_words = {
    # General positive
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'beautiful',
    'lovely', 'nice', 'best', 'perfect', 'awesome', 'outstanding', 'superb',
    'brilliant', 'magnificent', 'marvelous', 'splendid', 'terrific', 'fabulous',
    
    # Tourism-specific positive
    'scenic', 'breathtaking', 'stunning', 'picturesque', 'peaceful', 'serene',
    'relaxing', 'enjoyable', 'memorable', 'recommended', 'worth', 'must',
    'clean', 'friendly', 'helpful', 'comfortable', 'convenient', 'spacious',
    'delicious', 'tasty', 'fresh', 'authentic', 'cultural', 'historic',
    'paradise', 'heaven', 'magical', 'enchanting', 'charming', 'cozy',
    
    # Experience-related positive
    'love', 'loved', 'like', 'liked', 'enjoy', 'enjoyed', 'appreciate',
    'impressed', 'satisfied', 'happy', 'pleased', 'delighted', 'blessed',
    'refreshing', 'rejuvenating', 'calming', 'soothing', 'inspiring',
    'unique', 'special', 'incredible', 'remarkable', 'extraordinary',
    'welcoming', 'hospitable', 'professional', 'efficient', 'safe',
    
    # Sri Lanka specific
    'pristine', 'lush', 'tropical', 'exotic', 'wildlife', 'elephants',
    'beaches', 'temples', 'heritage', 'spices', 'tea', 'ayurveda'
}

negative_words = {
    # General negative
    'bad', 'poor', 'terrible', 'horrible', 'awful', 'worst', 'disappointing',
    'disappointed', 'boring', 'dull', 'unpleasant', 'disgusting', 'pathetic',
    
    # Tourism-specific negative
    'dirty', 'crowded', 'overpriced', 'expensive', 'rude', 'unfriendly',
    'noisy', 'smelly', 'unsafe', 'dangerous', 'scam', 'fake', 'waste',
    'avoid', 'overrated', 'mediocre', 'average', 'nothing', 'lacking',
    
    # Experience-related negative
    'hate', 'hated', 'dislike', 'regret', 'frustrated', 'annoyed', 'angry',
    'upset', 'uncomfortable', 'inconvenient', 'difficult', 'problem', 'issue',
    'broken', 'damaged', 'old', 'outdated', 'neglected', 'abandoned',
    'slow', 'wait', 'waiting', 'queue', 'traffic', 'pollution',
    'unhelpful', 'unprofessional', 'ripoff', 'tourist trap', 'hassle'
}

intensifiers = {'very', 'really', 'extremely', 'absolutely', 'totally', 
                'highly', 'super', 'incredibly', 'exceptionally', 'quite'}

negations = {'not', 'no', 'never', "don't", "doesn't", "didn't", 
             "won't", "wouldn't", "isn't", "aren't", "wasn't", "weren't",
             'hardly', 'barely', 'scarcely'}


# Calculate sentiment score using enhanced lexicon-based approach
def calculate_sentiment(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0.0
    
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    
    if len(words) == 0:
        return 0.0
    
    pos_count = 0
    neg_count = 0
    negation_active = False
    
    for i, word in enumerate(words):
        if word in negations:
            negation_active = True
            continue
        
        intensity = 1.5 if (i > 0 and words[i-1] in intensifiers) else 1.0
        
        if word in positive_words:
            if negation_active:
                neg_count += intensity
            else:
                pos_count += intensity
            negation_active = False
        elif word in negative_words:
            if negation_active:
                pos_count += intensity
            else:
                neg_count += intensity
            negation_active = False
        else:
            negation_active = False
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return (pos_count - neg_count) / total


# Count occurrences of sentiment words
def count_sentiment_words(text, word_set):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    return len(words & word_set)


# Extract additional features from review text
def extract_review_features(text):
    if pd.isna(text) or not isinstance(text, str):
        return pd.Series({'exclamation_count': 0, 'question_count': 0, 'caps_ratio': 0})
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    return pd.Series({
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'caps_ratio': caps_ratio
    })


print("Applying sentiment analysis to reviews...")

reviews_df['sentiment_score'] = reviews_df['Review'].apply(calculate_sentiment)
reviews_df['positive_word_count'] = reviews_df['Review'].apply(lambda x: count_sentiment_words(x, positive_words))
reviews_df['negative_word_count'] = reviews_df['Review'].apply(lambda x: count_sentiment_words(x, negative_words))
reviews_df['review_length'] = reviews_df['Review'].str.len().fillna(0).astype(int)
reviews_df['word_count'] = reviews_df['Review'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

# Additional text features
extra_features = reviews_df['Review'].apply(extract_review_features)
reviews_df = pd.concat([reviews_df, extra_features], axis=1)

print(f"    * Sentiment analysis complete")
print(f"    * Sentiment range: [{reviews_df['sentiment_score'].min():.2f}, {reviews_df['sentiment_score'].max():.2f}]")
print(f"    * Mean sentiment: {reviews_df['sentiment_score'].mean():.3f}")



# =============================================================================
# DATA CLEANING
# =============================================================================

print("\n" + "─" * 80)
print("DATA CLEANING")
print("─" * 80)

# Clean district names in reviews
reviews_df['district_clean'] = reviews_df['District'].str.strip().str.lower()
reviews_df = reviews_df[~reviews_df['district_clean'].str.contains('"', na=False)]
print(f"    * Reviews after cleaning: {len(reviews_df):,}")

# Clean accommodation data
accommodation_df['district_clean'] = accommodation_df['District'].str.strip().str.lower()
grade_mapping = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
accommodation_df['grade_numeric'] = accommodation_df['Grade'].map(grade_mapping).fillna(2)
print(f"    * Accommodation records: {len(accommodation_df):,}")

# Clean places data
places_df['district_clean'] = places_df['District'].str.strip().str.lower()
places_df['grade_numeric'] = places_df['Grade'].map(grade_mapping).fillna(2)
print(f"    * Places records: {len(places_df):,}")

# Clean Airbnb data
airbnb_df.columns = airbnb_df.columns.str.strip().str.replace('\ufeff', '')
airbnb_df['stars'] = pd.to_numeric(airbnb_df['stars'], errors='coerce')
airbnb_df['numberOfGuests'] = pd.to_numeric(airbnb_df['numberOfGuests'], errors='coerce')
print(f"    * Airbnb records: {len(airbnb_df):,}")

# Clean Tourist data
tourist_df['month_num'] = pd.to_datetime(tourist_df['month'], format='%B').dt.month
print(f"    * Tourist statistics records: {len(tourist_df):,}")



# =============================================================================
# FEATURE AGGREGATION BY DESTINATION
# =============================================================================

print("\n" + "─" * 80)
print("FEATURE AGGREGATION")
print("─" * 80)

# Aggregate review features by destination
print("Aggregating review features by destination...")
destination_agg = reviews_df.groupby(['Destination', 'district_clean']).agg({
    'sentiment_score': ['mean', 'std', 'min', 'max', 'count'],
    'positive_word_count': ['mean', 'sum'],
    'negative_word_count': ['mean', 'sum'],
    'review_length': ['mean', 'std'],
    'word_count': 'mean',
    'exclamation_count': 'mean',
    'question_count': 'mean',
    'caps_ratio': 'mean'
}).reset_index()

# Flatten column names
destination_agg.columns = [
    'destination', 'district',
    'avg_sentiment', 'sentiment_std', 'min_sentiment', 'max_sentiment', 'review_count',
    'avg_positive_words', 'total_positive_words',
    'avg_negative_words', 'total_negative_words',
    'avg_review_length', 'review_length_std',
    'avg_word_count',
    'avg_exclamation', 'avg_question', 'avg_caps_ratio'
]

destination_agg['sentiment_std'] = destination_agg['sentiment_std'].fillna(0)
destination_agg['review_length_std'] = destination_agg['review_length_std'].fillna(0)

# Calculate sentiment consistency (inverse of std)
destination_agg['sentiment_consistency'] = 1 - destination_agg['sentiment_std'].clip(0, 1)

# Calculate positive-negative ratio
destination_agg['sentiment_ratio'] = (
    (destination_agg['total_positive_words'] + 1) / 
    (destination_agg['total_negative_words'] + 1)
)

print(f"    * Unique destinations: {len(destination_agg)}")


# Aggregate accommodation data by district
print("Aggregating accommodation data by district...")
accommodation_agg = accommodation_df.groupby('district_clean').agg({
    'Name': 'count',
    'Rooms': ['sum', 'mean', 'std'],
    'grade_numeric': ['mean', 'std'],
    'Type': 'nunique'
}).reset_index()

accommodation_agg.columns = [
    'district', 'num_accommodations', 'total_rooms',
    'avg_rooms_per_hotel', 'rooms_std', 
    'avg_accommodation_grade', 'grade_std',
    'accommodation_diversity'
]
accommodation_agg = accommodation_agg.fillna(0)
print(f"    * Districts with accommodation: {len(accommodation_agg)}")


# Aggregate places data by district
print("Aggregating places data by district...")
places_type_counts = places_df.groupby(['district_clean', 'Type']).size().unstack(fill_value=0).reset_index()
places_type_counts.columns = ['district'] + [
    'places_' + col.lower().replace(' ', '_').replace('&', 'and') 
    for col in places_type_counts.columns[1:]
]

places_agg = places_df.groupby('district_clean').agg({
    'Name': 'count',
    'grade_numeric': ['mean', 'std']
}).reset_index()
places_agg.columns = ['district', 'total_places', 'avg_place_grade', 'place_grade_std']
places_agg = places_agg.merge(places_type_counts, on='district', how='left').fillna(0)
print(f"    * Districts with places: {len(places_agg)}")


# Aggregate Airbnb data [NEW]
print("Aggregating Airbnb data...")
airbnb_agg = airbnb_df.groupby(airbnb_df['roomType']).agg({
    'stars': ['mean', 'count'],
    'numberOfGuests': ['mean', 'sum']
}).reset_index()

# Create national-level Airbnb statistics
airbnb_national = {
    'airbnb_avg_rating': airbnb_df['stars'].mean(),
    'airbnb_total_listings': len(airbnb_df),
    'airbnb_avg_guests': airbnb_df['numberOfGuests'].mean(),
    'airbnb_rating_std': airbnb_df['stars'].std(),
    'airbnb_total_capacity': airbnb_df['numberOfGuests'].sum()
}
print(f"    * Airbnb national stats computed")


# Aggregate Tourist data
print("Aggregating tourist arrival data...")
tourist_agg = tourist_df.groupby('originCountry').agg({
    'totalCount': ['sum', 'mean'],
    'consumerPriceIndex': 'mean'
}).reset_index()
tourist_agg.columns = ['country', 'total_tourists', 'avg_monthly_tourists', 'home_cpi']

# Calculate tourism seasonality
monthly_tourists = tourist_df.groupby('month_num')['totalCount'].sum()
tourism_seasonality = monthly_tourists.std() / monthly_tourists.mean()

# National tourism statistics
tourist_national = {
    'total_annual_tourists': tourist_df['totalCount'].sum(),
    'avg_monthly_tourists': tourist_df.groupby(['year', 'month'])['totalCount'].sum().mean(),
    'tourism_seasonality': tourism_seasonality,
    'top_source_countries': tourist_df.groupby('originCountry')['totalCount'].sum().nlargest(5).to_dict(),
    'avg_dollar_rate': tourist_df['dollarRate'].mean(),
    'avg_temperature': tourist_df['apparent_temperature_mean_celcius'].mean()
}
print(f"    * Tourist arrival stats computed")
print(f"    * Total tourists: {tourist_national['total_annual_tourists']:,}")



# =============================================================================
# MERGE ALL DATASETS
# =============================================================================

print("\n" + "─" * 80)
print("MERGING DATASETS")
print("─" * 80)

# Start with destination data
final_df = destination_agg.copy()

# Merge accommodation data
final_df = final_df.merge(accommodation_agg, on='district', how='left')

# Merge places data
final_df = final_df.merge(places_agg, on='district', how='left')

# Add Airbnb national features (same for all destinations)
for key, value in airbnb_national.items():
    final_df[key] = value

# Add tourist national features
final_df['national_tourism_index'] = tourist_national['avg_monthly_tourists'] / 100000
final_df['tourism_seasonality'] = tourist_national['tourism_seasonality']
final_df['avg_exchange_rate'] = tourist_national['avg_dollar_rate']

# Fill missing numeric values
numeric_cols = final_df.select_dtypes(include=[np.number]).columns
final_df[numeric_cols] = final_df[numeric_cols].fillna(0)

print(f"    * Final merged dataset shape: {final_df.shape}")
print(f"    * Total features: {len(final_df.columns)}")



# =============================================================================
# CREATE TARGET VARIABLE
# =============================================================================

print("\n" + "─" * 80)
print("TARGET VARIABLE CREATION")
print("─" * 80)

# Normalize series to 0-1 range
def normalize(series):
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


# Calculate experience score (weighted combination of factors)
# Target variable for regression
final_df['experience_score'] = (
    0.45 * normalize(final_df['avg_sentiment']) +              # 45% - Sentiment
    0.15 * normalize(final_df['review_count'].clip(upper=500)) + # 15% - Popularity/visibility
    0.15 * normalize(final_df['num_accommodations']) +         # 15% - Infrastructure
    0.10 * normalize(final_df['avg_accommodation_grade']) +    # 10% - Quality
    0.10 * normalize(final_df['total_places']) +               # 10% - Amenities
    0.05 * normalize(final_df['sentiment_consistency'])        # 5% - Consistency
)

# Convert continuous score to category
def categorize_experience(score):
    if score >= 0.65:
        return 'Excellent'
    elif score >= 0.45:
        return 'Good'
    elif score >= 0.25:
        return 'Average'
    else:
        return 'Poor'


final_df['experience_category'] = final_df['experience_score'].apply(categorize_experience)

# Encode for classification
category_mapping = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}
final_df['experience_class'] = final_df['experience_category'].map(category_mapping)

print("    * Target variable created")
print(f"\n   Target Distribution:")
print(f"   " + "-" * 30)
for cat in ['Excellent', 'Good', 'Average', 'Poor']:
    count = (final_df['experience_category'] == cat).sum()
    pct = count / len(final_df) * 100
    print(f"   {cat:<12}: {count:>4} ({pct:>5.1f}%)")



# =============================================================================
# SAVE PROCESSED DATA
# =============================================================================

print("\n" + "─" * 80)
print("SAVING PROCESSED DATA")
print("─" * 80)

# Save main dataset
final_df.to_csv('../data/processed_dataset.csv', index=False)
print(f"    * Saved: ../data/processed_dataset.csv")

# Save reviews with sentiment
reviews_df.to_csv('../data/reviews_with_sentiment.csv', index=False)
print(f"    * Saved: ../data/reviews_with_sentiment.csv")

# Save tourist statistics for plan generator
tourist_df.to_csv('../data/tourist_statistics.csv', index=False)
print(f"    * Saved: ../data/tourist_statistics.csv")

# Save tourist country aggregations
tourist_agg.to_csv('../data/tourist_by_country.csv', index=False)
print(f"    * Saved: ../data/tourist_by_country.csv")

# Save Airbnb data
airbnb_df.to_csv('../data/airbnb_cleaned.csv', index=False)
print(f"    * Saved: ../data/airbnb_cleaned.csv")



# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE!")
print("=" * 80)

feature_cols = [c for c in final_df.columns if c not in 
                ['destination', 'district', 'experience_score', 'experience_category', 'experience_class']]

print(f"""
DATASET SUMMARY
---------------
*  Total destinations:     {len(final_df)}
*  Total reviews:          {len(reviews_df):,}
*  Unique districts:       {final_df['district'].nunique()}
*  Total features:         {len(feature_cols)}

DATA SOURCES USED
-----------------
*  Destination Reviews     ({len(reviews_df):,} records)
*  Accommodation Info      ({len(accommodation_df):,} records)
*  Places & Dining         ({len(places_df):,} records)
*  Airbnb Listings [NEW]   ({len(airbnb_df):,} records)
*  Tourist Statistics [NEW]({len(tourist_df):,} records)

EXPERIENCE SCORE STATISTICS
---------------------------
*  Mean:  {final_df['experience_score'].mean():.4f}
*  Std:   {final_df['experience_score'].std():.4f}
*  Min:   {final_df['experience_score'].min():.4f}
*  Max:   {final_df['experience_score'].max():.4f}

FEATURES ({len(feature_cols)} total):
""")

for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col}")

print("\n" + "=" * 80)