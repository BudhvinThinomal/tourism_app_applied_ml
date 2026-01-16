import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Sri Lanka Tourism Analyzer",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CONSTANTS
# =============================================================================

CATEGORY_COLORS = {
    'Poor': '#e74c3c',
    'Average': '#f39c12',
    'Good': '#3498db',
    'Excellent': '#27ae60'
}

CATEGORY_ORDER = ['Poor', 'Average', 'Good', 'Excellent']

CATEGORY_EMOJI = {
    'Poor': '❌',
    'Average': '⚠️',
    'Good': '✅',
    'Excellent': '🌟'
}


# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================

# Load trained ML models
@st.cache_resource
def load_models():
    try:
        with open('models/classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open('models/regressor.pkl', 'rb') as f:
            reg = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        try:
            with open('models/best_hyperparameters.pkl', 'rb') as f:
                hyperparams = pickle.load(f)
        except:
            hyperparams = None
        return clf, reg, feature_names, hyperparams, True
    except Exception as e:
        return None, None, None, None, False

# Load processed datasets
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed_dataset.csv')
        try:
            importance = pd.read_csv('outputs/feature_importance.csv')
        except:
            importance = None
        try:
            tourist_df = pd.read_csv('data/tourist_statistics.csv')
        except:
            tourist_df = None
        try:
            reviews_df = pd.read_csv('data/reviews_with_sentiment.csv')
        except:
            reviews_df = None
        return df, importance, tourist_df, reviews_df, True
    except Exception as e:
        return None, None, None, None, False


clf, reg, feature_names, hyperparams, models_loaded = load_models()
df, importance_df, tourist_df, reviews_df, data_loaded = load_data()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_category_emoji(category):
    return CATEGORY_EMOJI.get(category, '❓')


# Generate specific improvement recommendations based on data analysis
def get_improvement_recommendations(dest_row, df, feature_names):
    recommendations = []
    
    # Get excellent destinations for comparison
    excellent_df = df[df['experience_category'] == 'Excellent']
    
    if len(excellent_df) == 0:
        return ["Insufficient data for comparison"]
    
    # Calculate means for excellent destinations
    excellent_means = excellent_df[feature_names].mean()
    
    # Compare and generate recommendations
    # 1. Sentiment Analysis
    if dest_row['avg_sentiment'] < excellent_means.get('avg_sentiment', 0):
        gap = excellent_means.get('avg_sentiment', 0) - dest_row['avg_sentiment']
        recommendations.append({
            'area': 'Review Sentiment',
            'priority': 'High' if gap > 0.3 else 'Medium',
            'current': f"{dest_row['avg_sentiment']:.3f}",
            'target': f"{excellent_means.get('avg_sentiment', 0):.3f}",
            'gap': f"{gap:.3f}",
            'actions': [
                "Address common complaints in negative reviews",
                "Improve customer service training",
                "Respond promptly to tourist feedback",
                "Enhance cleanliness and maintenance"
            ]
        })
    
    # 2. Accommodation
    if dest_row.get('num_accommodations', 0) < excellent_means.get('num_accommodations', 0):
        recommendations.append({
            'area': 'Accommodation Infrastructure',
            'priority': 'High',
            'current': f"{int(dest_row.get('num_accommodations', 0))}",
            'target': f"{int(excellent_means.get('num_accommodations', 0))}",
            'gap': f"{int(excellent_means.get('num_accommodations', 0) - dest_row.get('num_accommodations', 0))}",
            'actions': [
                "Encourage new hotel/guesthouse development",
                "Provide incentives for accommodation businesses",
                "Promote homestay programs",
                "Improve existing accommodation quality"
            ]
        })
    
    # 3. Tourist Facilities
    if dest_row.get('total_places', 0) < excellent_means.get('total_places', 0):
        recommendations.append({
            'area': 'Tourist Facilities',
            'priority': 'Medium',
            'current': f"{int(dest_row.get('total_places', 0))}",
            'target': f"{int(excellent_means.get('total_places', 0))}",
            'gap': f"{int(excellent_means.get('total_places', 0) - dest_row.get('total_places', 0))}",
            'actions': [
                "Develop more restaurants and cafes",
                "Add tourist information centers",
                "Create recreational facilities",
                "Establish tour operator offices"
            ]
        })
    
    # 4. Accommodation Quality
    if dest_row.get('avg_accommodation_grade', 0) < excellent_means.get('avg_accommodation_grade', 0):
        recommendations.append({
            'area': 'Accommodation Quality',
            'priority': 'Medium',
            'current': f"{dest_row.get('avg_accommodation_grade', 0):.2f}",
            'target': f"{excellent_means.get('avg_accommodation_grade', 0):.2f}",
            'gap': f"{excellent_means.get('avg_accommodation_grade', 0) - dest_row.get('avg_accommodation_grade', 0):.2f}",
            'actions': [
                "Implement quality certification programs",
                "Provide training for hospitality staff",
                "Enforce tourism board standards",
                "Offer upgrade incentives for lower-grade establishments"
            ]
        })
    
    # 5. Review Volume (Visibility)
    if dest_row.get('review_count', 0) < excellent_means.get('review_count', 0) * 0.5:
        recommendations.append({
            'area': 'Tourist Visibility',
            'priority': 'Medium',
            'current': f"{int(dest_row.get('review_count', 0))}",
            'target': f"{int(excellent_means.get('review_count', 0))}",
            'gap': f"{int(excellent_means.get('review_count', 0) - dest_row.get('review_count', 0))}",
            'actions': [
                "Increase marketing and promotion efforts",
                "Encourage tourists to leave reviews",
                "Partner with travel agencies and tour operators",
                "Improve online presence and social media"
            ]
        })
    
    # 6. Review Consistency
    if dest_row.get('sentiment_std', 1) > excellent_means.get('sentiment_std', 0) + 0.1:
        recommendations.append({
            'area': 'Experience Consistency',
            'priority': 'Low',
            'current': f"{dest_row.get('sentiment_std', 0):.3f} (high variance)",
            'target': f"{excellent_means.get('sentiment_std', 0):.3f}",
            'gap': 'Inconsistent experiences',
            'actions': [
                "Standardize service quality across providers",
                "Create tourism guidelines and best practices",
                "Regular monitoring and quality audits",
                "Address seasonal variation in service quality"
            ]
        })
    
    if not recommendations:
        recommendations.append({
            'area': 'Maintenance',
            'priority': 'Low',
            'current': 'Good',
            'target': 'Excellent',
            'gap': 'Minor',
            'actions': [
                "Maintain current standards",
                "Continue monitoring tourist feedback",
                "Explore unique selling points",
                "Consider sustainable tourism initiatives"
            ]
        })
    
    return recommendations


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.title("🏝️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🔍 Explore Destinations",
        "📊 Make Prediction",
        "🧠 Model Explanation",
        "📈 Analytics Dashboard",
        "🗺️ Tourism Plan Generator",
        "📋 Rating Improvement Advisor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Project Info")
st.sidebar.info("""
**Algorithm:** HistGradientBoosting

**Features:** Hyperparameter Tuned

**Data Sources:** 5 Datasets

**XAI:** SHAP + Permutation
""")


# =============================================================================
# PAGE: HOME
# =============================================================================

if page == "🏠 Home":
    st.title("Sri Lanka Tourism Experience Analyzer")
    st.markdown("### Machine Learning-Powered Destination Analysis & Planning")
    
    if data_loaded:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Destinations", len(df))
        with col2:
            st.metric("Districts Covered", df['district'].nunique())
        with col3:
            excellent = (df['experience_category'] == 'Excellent').sum()
            st.metric("Excellent Destinations", excellent)
        with col4:
            if reviews_df is not None:
                st.metric("Reviews Analyzed", f"{len(reviews_df):,}")
            else:
                st.metric("Model Status", "Ready!!")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 🎯 What This System Does
        
        | Feature | Description |
        |---------|-------------|
        | 📊 **Experience Prediction** | Predict quality rating for destinations |
        | 🗺️ **Tourism Plan Generator** | Create personalized travel itineraries |
        | 📋 **Improvement Advisor** | Actionable recommendations for destinations |
        | 🧠 **Explainable AI** | Understand why predictions are made |
        
        ## 📚 Data Sources Used
        
        - 📝 **35,000+ Tourist Reviews** - Sentiment analyzed
        - 🏨 **2,100+ Accommodations** - Hotels, resorts, guesthouses
        - 🍽️ **1,500+ Places** - Restaurants, spas, attractions
        - 🏡 **5,000+ Airbnb Listings** - Alternative accommodations
        - ✈️ **Tourist Statistics** - Arrival data by country
        """)
    
    with col2:
        st.markdown("""
        ## 🔮 Experience Categories
        
        | Category | Score | Description |
        |----------|-------|-------------|
        | 🌟 **Excellent** | 0.65+ | Top-tier, highly recommended |
        | ✅ **Good** | 0.45-0.65 | Solid choice for tourists |
        | ⚠️ **Average** | 0.25-0.45 | Mixed reviews |
        | ❌ **Poor** | <0.25 | Needs improvement |
        
        ## 🆕 New Features
        
        ### 🗺️ Tourism Plan Generator
        Input your preferences and get a personalized multi-day itinerary!
        
        ### 📋 Rating Improvement Advisor
        For tourism boards: Get specific recommendations to improve ratings.
        """)


# =============================================================================
# PAGE: EXPLORE DESTINATIONS
# =============================================================================

elif page == "🔍 Explore Destinations":
    st.title("🔍 Explore Destinations")
    
    if not data_loaded:
        st.error("Data not loaded.")
    else:
        search_query = st.text_input("🔍 Search destinations", placeholder="Type destination name...")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            districts = ['All'] + sorted(df['district'].unique().tolist())
            selected_district = st.selectbox("District", districts)
        with col2:
            categories = ['All'] + CATEGORY_ORDER
            selected_category = st.selectbox("Category", categories)
        with col3:
            sort_by = st.selectbox("Sort by", ['experience_score', 'avg_sentiment', 'review_count'])
        with col4:
            sort_order = st.selectbox("Order", ['Descending', 'Ascending'])
        
        filtered = df.copy()
        
        if search_query:
            filtered = filtered[filtered['destination'].str.lower().str.contains(search_query.lower(), na=False)]
        if selected_district != 'All':
            filtered = filtered[filtered['district'] == selected_district]
        if selected_category != 'All':
            filtered = filtered[filtered['experience_category'] == selected_category]
        
        filtered = filtered.sort_values(sort_by, ascending=(sort_order == 'Ascending'))
        
        st.markdown(f"**Showing {len(filtered)} destinations**")
        
        for _, row in filtered.head(50).iterrows():
            with st.expander(f"{get_category_emoji(row['experience_category'])} **{row['destination']}** - {row['district'].title()} ({row['experience_score']:.3f})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Category:** {row['experience_category']}\n\n**Score:** {row['experience_score']:.4f}\n\n**Reviews:** {int(row['review_count'])}")
                with col2:
                    st.markdown(f"**Avg Sentiment:** {row['avg_sentiment']:.3f}\n\n**Positive Words:** {row['avg_positive_words']:.1f}")
                with col3:
                    st.markdown(f"**Accommodations:** {int(row.get('num_accommodations', 0))}\n\n**Places:** {int(row.get('total_places', 0))}")


# =============================================================================
# PAGE: MAKE PREDICTION
# =============================================================================

elif page == "📊 Make Prediction":
    st.title("📊 Make Prediction")
    
    if not models_loaded:
        st.error("Models not loaded.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Review Characteristics")
            review_quality = st.select_slider("Review Quality", ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'], value='Positive')
            review_count = st.slider("Number of Reviews", 1, 500, 50)
            review_consistency = st.select_slider("Consistency", ['Very Inconsistent', 'Mixed', 'Consistent', 'Very Consistent'], value='Consistent')
        
        with col2:
            st.markdown("### 🏨 Infrastructure")
            destination_size = st.select_slider("Size", ['Very Small', 'Small', 'Medium', 'Large', 'Very Large'], value='Medium')
            accommodation_quality = st.select_slider("Accommodation Quality", ['Poor', 'Average', 'Good', 'Excellent'], value='Good')
            facilities_level = st.select_slider("Facilities", ['Minimal', 'Basic', 'Moderate', 'Good', 'Comprehensive'], value='Moderate')
        
        if st.button("🔮 Predict Experience", type="primary", use_container_width=True):
            sentiment_map = {'Very Negative': -0.8, 'Negative': -0.4, 'Neutral': 0, 'Positive': 0.4, 'Very Positive': 0.8}
            consistency_map = {'Very Inconsistent': 0.2, 'Mixed': 0.6, 'Consistent': 0.8, 'Very Consistent': 0.95}
            size_map = {'Very Small': 0, 'Small': 5, 'Medium': 20, 'Large': 50, 'Very Large': 100}
            quality_map = {'Poor': 1, 'Average': 2.5, 'Good': 3, 'Excellent': 4}
            facilities_map = {'Minimal': 2, 'Basic': 10, 'Moderate': 25, 'Good': 50, 'Comprehensive': 100}
            
            feature_values = {
                'avg_sentiment': sentiment_map[review_quality],
                'sentiment_std': 1 - consistency_map[review_consistency],
                'min_sentiment': sentiment_map[review_quality] - 0.3,
                'max_sentiment': min(1, sentiment_map[review_quality] + 0.3),
                'review_count': review_count,
                'avg_positive_words': max(0, (sentiment_map[review_quality] + 1) * 3),
                'total_positive_words': max(0, (sentiment_map[review_quality] + 1) * 3 * review_count),
                'avg_negative_words': max(0, (1 - sentiment_map[review_quality]) * 2),
                'total_negative_words': max(0, (1 - sentiment_map[review_quality]) * 2 * review_count),
                'avg_review_length': 150,
                'review_length_std': 50,
                'avg_word_count': 30,
                'num_accommodations': size_map[destination_size],
                'total_rooms': size_map[destination_size] * 20,
                'avg_rooms_per_hotel': 20,
                'avg_accommodation_grade': quality_map[accommodation_quality],
                'total_places': facilities_map[facilities_level],
                'avg_place_grade': quality_map[accommodation_quality],
            }
            
            input_df = pd.DataFrame([feature_values])
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]
            
            pred_class = clf.predict(input_df)[0]
            pred_proba = clf.predict_proba(input_df)[0]
            pred_score = reg.predict(input_df)[0]
            
            class_names = ['Poor', 'Average', 'Good', 'Excellent']
            pred_name = class_names[pred_class]
            
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"### {get_category_emoji(pred_name)} {pred_name}")
            with col2:
                st.metric("Experience Score", f"{pred_score:.3f}")
            with col3:
                st.metric("Confidence", f"{pred_proba[pred_class]*100:.1f}%")
            
            prob_df = pd.DataFrame({'Category': class_names, 'Probability': pred_proba})
            fig = px.bar(prob_df, x='Category', y='Probability', color='Category', color_discrete_map=CATEGORY_COLORS)
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: MODEL EXPLANATION
# =============================================================================

elif page == "🧠 Model Explanation":
    st.title("🧠 Model Explanation (XAI)")
    
    if not data_loaded:
        st.error("Data not loaded.")
    else:
        st.markdown("Understanding **why** the model makes predictions.")
        
        if importance_df is not None:
            st.markdown("### 📊 Feature Importance (SHAP)")
            top_15 = importance_df.head(15)
            fig = px.bar(top_15, x='shap_importance', y='feature', orientation='h', color='shap_importance', color_continuous_scale='Blues')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### 🏗️ Infrastructure Matters
                - `total_places` and `num_accommodations` are top predictors
                - More facilities = better experience
                """)
            with col2:
                st.markdown("""
                #### 📝 Sentiment is Key
                - `avg_sentiment` strongly influences predictions
                - Positive reviews = good ratings
                """)
        else:
            st.warning("Run explainability.py first to generate SHAP values.")
        
        if hyperparams:
            st.markdown("### 🔧 Tuned Hyperparameters")
            hp_df = pd.DataFrame([{'Parameter': k, 'Value': f"{v:.4f}" if isinstance(v, float) else str(v)} for k, v in hyperparams.items()])
            st.table(hp_df)


# =============================================================================
# PAGE: ANALYTICS DASHBOARD
# =============================================================================

elif page == "📈 Analytics Dashboard":
    st.title("📈 Analytics Dashboard")
    
    if not data_loaded:
        st.error("Data not loaded.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Destinations", len(df))
        with col2:
            st.metric("Avg Score", f"{df['experience_score'].mean():.3f}")
        with col3:
            st.metric("Avg Sentiment", f"{df['avg_sentiment'].mean():.3f}")
        with col4:
            st.metric("Total Reviews", f"{df['review_count'].sum():,.0f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Category Distribution")
            cat_counts = df['experience_category'].value_counts()
            fig = px.pie(values=cat_counts.values, names=cat_counts.index, color=cat_counts.index, color_discrete_map=CATEGORY_COLORS)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Score Distribution")
            fig = px.histogram(df, x='experience_score', nbins=25, color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📍 District Performance")
        district_stats = df.groupby('district').agg({'experience_score': 'mean', 'destination': 'count', 'avg_sentiment': 'mean'}).reset_index()
        district_stats.columns = ['District', 'Avg Score', 'Destinations', 'Avg Sentiment']
        fig = px.bar(district_stats.sort_values('Avg Score', ascending=False), x='District', y='Avg Score', color='Avg Sentiment', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: TOURISM PLAN GENERATOR (NEW - IMPROVED)
# =============================================================================

elif page == "🗺️ Tourism Plan Generator":
    st.title("🗺️ Tourism Plan Generator")
    st.markdown("### Create a Personalized Travel Itinerary for Sri Lanka")
    
    if not data_loaded:
        st.error("Data not loaded.")
    else:
        # =================================================================
        # DESTINATION CLASSIFICATION SYSTEM
        # =================================================================
        
        # Keywords to classify destinations by type
        DESTINATION_CATEGORIES = {
            'Beaches': ['beach', 'coast', 'sea', 'island', 'bay', 'lagoon', 'marine', 'coral', 'diving', 'snorkeling', 'surfing', 'whale', 'dolphin', 'turtle', 'mirissa', 'unawatuna', 'hikkaduwa', 'bentota', 'arugam', 'trinco', 'negombo', 'kalpitiya'],
            'Wildlife': ['safari', 'national park', 'wildlife', 'elephant', 'leopard', 'bird', 'sanctuary', 'zoo', 'yala', 'udawalawe', 'wilpattu', 'minneriya', 'bundala', 'kumana', 'sinharaja', 'horton'],
            'Mountains': ['mountain', 'peak', 'hill', 'ella', 'waterfall', 'falls', 'adam', 'knuckles', 'horton', 'nuwara', 'haputale', 'bandarawela', 'ravana', 'diyaluma', 'bambarakanda', 'hiking', 'trekking', 'trail'],
            'Temples & Culture': ['temple', 'viharaya', 'kovil', 'mosque', 'church', 'buddha', 'dagoba', 'stupa', 'ancient', 'heritage', 'museum', 'fort', 'palace', 'sigiriya', 'anuradhapura', 'polonnaruwa', 'dambulla', 'kandy', 'tooth', 'dalada', 'archaeological'],
            'Tea & Plantations': ['tea', 'plantation', 'factory', 'estate', 'ceylon', 'nuwara eliya', 'hatton', 'ella', 'haputale'],
            'Adventure': ['adventure', 'rafting', 'kayak', 'zipline', 'paragliding', 'rock climbing', 'camping', 'cycling', 'kitesurfing', 'water sport'],
            'Relaxation & Spa': ['spa', 'ayurveda', 'wellness', 'resort', 'hot spring', 'botanical garden', 'lake', 'park']
        }
        
        # Geographic regions for route optimization
        REGIONS = {
            'Colombo & Western': ['colombo', 'gampaha', 'kalutara'],
            'Southern Coast': ['galle', 'matara', 'hambantota'],
            'Hill Country': ['kandy', 'matale', 'nuwara eliya', 'badulla', 'hatton'],
            'Cultural Triangle': ['anuradhapura', 'polonnaruwa', 'kurunegala', 'dambulla', 'sigiriya'],
            'East Coast': ['trincomalee', 'batticaloa', 'ampara', 'kalmunai'],
            'North': ['jaffna', 'kilinochchi', 'mannar', 'vavuniya'],
            'Sabaragamuwa': ['ratnapura', 'kegalle']
        }
        
        # Recommended travel routes (optimized paths)
        TRAVEL_ROUTES = {
            'Classic Sri Lanka (7-10 days)': ['colombo', 'kandy', 'nuwara eliya', 'ella', 'yala', 'galle', 'colombo'],
            'Cultural Heritage (5-7 days)': ['colombo', 'anuradhapura', 'polonnaruwa', 'sigiriya', 'kandy', 'colombo'],
            'Beach Paradise (5-7 days)': ['colombo', 'bentota', 'galle', 'mirissa', 'tangalle', 'colombo'],
            'Hill Country Explorer (4-6 days)': ['colombo', 'kandy', 'nuwara eliya', 'ella', 'haputale', 'colombo'],
            'Wildlife Adventure (5-7 days)': ['colombo', 'udawalawe', 'yala', 'bundala', 'mirissa', 'colombo'],
            'Complete Circuit (14 days)': ['colombo', 'anuradhapura', 'polonnaruwa', 'kandy', 'nuwara eliya', 'ella', 'yala', 'mirissa', 'galle', 'colombo']
        }
        
        def classify_destination(dest_name, district):
            """Classify a destination into categories based on name and location."""
            categories = []
            name_lower = dest_name.lower()
            district_lower = district.lower()
            
            for category, keywords in DESTINATION_CATEGORIES.items():
                for keyword in keywords:
                    if keyword in name_lower or keyword in district_lower:
                        categories.append(category)
                        break
            
            # Default category based on district if no match
            if not categories:
                if district_lower in ['galle', 'matara', 'hambantota', 'trincomalee', 'batticaloa']:
                    categories.append('Beaches')
                elif district_lower in ['kandy', 'anuradhapura', 'polonnaruwa']:
                    categories.append('Temples & Culture')
                elif district_lower in ['nuwara eliya', 'badulla', 'hatton']:
                    categories.append('Mountains')
                else:
                    categories.append('General')
            
            return categories
        
        def get_region(district):
            """Get the region for a district."""
            district_lower = district.lower()
            for region, districts in REGIONS.items():
                if any(d in district_lower for d in districts):
                    return region
            return 'Other'
        
        def calculate_route_score(destinations_list):
            """Calculate a route optimization score (lower is better)."""
            if len(destinations_list) < 2:
                return 0
            
            region_changes = 0
            prev_region = None
            for dest in destinations_list:
                region = get_region(dest['district'])
                if prev_region and region != prev_region:
                    region_changes += 1
                prev_region = region
            
            return region_changes
        
        # =================================================================
        # USER INPUT SECTION
        # =================================================================
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Trip Details")
            
            trip_duration = st.slider("Trip Duration (days)", 3, 14, 7)
            
            travel_style = st.selectbox(
                "Travel Style",
                ["Comfort Traveler", "Budget Backpacker", "Luxury Seeker", "Adventure Explorer", "Cultural Enthusiast"]
            )
            
            group_type = st.selectbox(
                "Group Type", 
                ["Couple", "Solo", "Family with Kids", "Friends Group", "Senior Travelers"]
            )
            
            starting_point = st.selectbox(
                "Starting Point",
                ["Colombo (Airport)", "Negombo", "Kandy", "Galle"]
            )
        
        with col2:
            st.markdown("#### 🎯 Preferences")
            
            interests = st.multiselect(
                "Primary Interests (select up to 3)",
                ["Beaches", "Wildlife", "Mountains", "Temples & Culture", "Tea & Plantations", "Adventure", "Relaxation & Spa"],
                default=["Beaches", "Temples & Culture"],
                max_selections=3
            )
            
            pace = st.select_slider(
                "Travel Pace",
                options=['Relaxed (fewer places, more time)', 'Moderate', 'Fast-paced (see more)'],
                value='Moderate'
            )
            
            min_rating = st.select_slider(
                "Minimum Destination Quality",
                options=['Any', 'Average+', 'Good+', 'Excellent Only'],
                value='Good+'
            )
            
            include_hidden_gems = st.checkbox("Include hidden gems (less touristy)", value=True)
        
        # Suggested route based on duration
        st.markdown("#### 🛤️ Suggested Routes")
        
        if trip_duration <= 5:
            suggested_routes = ['Cultural Heritage (5-7 days)', 'Beach Paradise (5-7 days)', 'Hill Country Explorer (4-6 days)']
        elif trip_duration <= 8:
            suggested_routes = ['Classic Sri Lanka (7-10 days)', 'Wildlife Adventure (5-7 days)', 'Cultural Heritage (5-7 days)']
        else:
            suggested_routes = ['Classic Sri Lanka (7-10 days)', 'Complete Circuit (14 days)', 'Wildlife Adventure (5-7 days)']
        
        selected_route = st.selectbox("Choose a route template (or customize below)", ['Custom Plan'] + suggested_routes)
        
        # =================================================================
        # GENERATE PLAN
        # =================================================================
        
        if st.button("🎯 Generate My Travel Plan", type="primary", use_container_width=True):
            
            # Add categories to dataframe
            df_plan = df.copy()
            df_plan['categories'] = df_plan.apply(lambda x: classify_destination(x['destination'], x['district']), axis=1)
            df_plan['region'] = df_plan['district'].apply(get_region)
            
            # Apply rating filter
            if min_rating == 'Average+':
                df_plan = df_plan[df_plan['experience_class'] >= 1]
            elif min_rating == 'Good+':
                df_plan = df_plan[df_plan['experience_class'] >= 2]
            elif min_rating == 'Excellent Only':
                df_plan = df_plan[df_plan['experience_class'] == 3]
            
            # Filter by interests
            if interests:
                def matches_interest(categories):
                    return any(interest in categories for interest in interests)
                df_plan['matches_interest'] = df_plan['categories'].apply(matches_interest)
                df_interest = df_plan[df_plan['matches_interest']]
                if len(df_interest) < 3:
                    df_interest = df_plan  # Fallback if too few matches
            else:
                df_interest = df_plan
            
            # Include hidden gems (lower review count but good rating)
            if include_hidden_gems:
                median_reviews = df_interest['review_count'].median()
                df_interest['is_hidden_gem'] = (df_interest['review_count'] < median_reviews) & (df_interest['experience_class'] >= 2)
            
            # Calculate destinations per day based on pace
            if pace == 'Relaxed (fewer places, more time)':
                destinations_per_day = 0.5  # 1 destination every 2 days
            elif pace == 'Moderate':
                destinations_per_day = 0.75  # 3 destinations every 4 days
            else:
                destinations_per_day = 1  # 1 destination per day
            
            num_destinations = max(3, min(int(trip_duration * destinations_per_day), len(df_interest), 10))
            
            # Smart destination selection
            selected_destinations = []
            used_districts = set()
            used_regions = set()
            
            # Sort by score but prioritize variety
            df_sorted = df_interest.sort_values('experience_score', ascending=False)
            
            # First pass: get top destinations matching interests with region diversity
            for _, row in df_sorted.iterrows():
                if len(selected_destinations) >= num_destinations:
                    break
                
                region = row['region']
                district = row['district']
                
                # Ensure geographic diversity
                if len(used_regions) < 3 or region not in used_regions or len(selected_destinations) > num_destinations // 2:
                    # Avoid too many from same district
                    if district not in used_districts or len(selected_destinations) > num_destinations * 0.7:
                        selected_destinations.append(row.to_dict())
                        used_districts.add(district)
                        used_regions.add(region)
            
            # Add hidden gems if enabled
            if include_hidden_gems and len(selected_destinations) < num_destinations:
                hidden_gems = df_interest[df_interest.get('is_hidden_gem', False)]
                for _, row in hidden_gems.head(2).iterrows():
                    if row['destination'] not in [d['destination'] for d in selected_destinations]:
                        selected_destinations.append(row.to_dict())
                        if len(selected_destinations) >= num_destinations:
                            break
            
            # Sort destinations by geographic proximity (simple region-based sorting)
            region_order = ['Colombo & Western', 'Cultural Triangle', 'Hill Country', 'East Coast', 'Southern Coast', 'Sabaragamuwa', 'North']
            selected_destinations.sort(key=lambda x: region_order.index(x['region']) if x['region'] in region_order else 99)
            
            # =================================================================
            # DISPLAY ITINERARY
            # =================================================================
            
            st.markdown("---")
            st.markdown("## 🗓️ Your Personalized Sri Lanka Itinerary")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{trip_duration} Days")
            with col2:
                st.metric("Destinations", len(selected_destinations))
            with col3:
                avg_score = np.mean([d['experience_score'] for d in selected_destinations])
                st.metric("Avg Quality", f"{avg_score:.2f}")
            with col4:
                regions_covered = len(set(d['region'] for d in selected_destinations))
                st.metric("Regions", regions_covered)
            
            # Travel style tips
            st.markdown("---")
            
            with st.expander("💡 Travel Tips for Your Style", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{travel_style} Tips:**")
                    if travel_style == "Budget Backpacker":
                        st.write("• Stay in guesthouses & hostels (Rs. 2,000-5,000/night)")
                        st.write("• Use public buses (very cheap, ~Rs. 100-500)")
                        st.write("• Eat at local 'hotels' (restaurants)")
                        st.write("• Budget: ~$30-50/day")
                    elif travel_style == "Luxury Seeker":
                        st.write("• Book 5-star resorts (Aman, Chena Huts, etc.)")
                        st.write("• Hire private car with driver (~$50-80/day)")
                        st.write("• Fine dining & boutique experiences")
                        st.write("• Budget: ~$300-500+/day")
                    elif travel_style == "Adventure Explorer":
                        st.write("• Mix camping & eco-lodges")
                        st.write("• Rent a motorcycle or tuk-tuk")
                        st.write("• Join adventure tours (rafting, hiking)")
                        st.write("• Budget: ~$50-100/day")
                    else:
                        st.write("• Mid-range hotels & boutique stays")
                        st.write("• Hire car with driver (~$40-60/day)")
                        st.write("• Mix of local & tourist restaurants")
                        st.write("• Budget: ~$80-150/day")
                
                with col2:
                    st.markdown(f"**{group_type} Tips:**")
                    if group_type == "Family with Kids":
                        st.write("• Choose family-friendly hotels with pools")
                        st.write("• Plan shorter travel days (max 3-4 hours)")
                        st.write("• Include wildlife experiences (kids love elephants!)")
                        st.write("• Carry snacks & entertainment for drives")
                    elif group_type == "Couple":
                        st.write("• Consider romantic boutique hotels")
                        st.write("• Book couples spa treatments")
                        st.write("• Sunset spots: Galle Fort, Ella, Mirissa")
                        st.write("• Private tours for flexibility")
                    elif group_type == "Solo":
                        st.write("• Stay in social hostels to meet others")
                        st.write("• Join group tours for activities")
                        st.write("• Use ride-sharing apps (PickMe)")
                        st.write("• Keep emergency contacts handy")
                    else:
                        st.write("• Book accommodations together for discounts")
                        st.write("• Hire a van for group travel")
                        st.write("• Plan group activities (rafting, safari)")
                        st.write("• Split costs for guides & drivers")
            
            st.markdown("---")
            st.markdown("### 📍 Day-by-Day Itinerary")
            
            # Calculate days per destination
            days_per_dest = trip_duration / len(selected_destinations)
            current_day = 1
            
            for idx, dest in enumerate(selected_destinations):
                # Calculate day range
                end_day = min(round(current_day + days_per_dest - 0.5), trip_duration)
                if idx == len(selected_destinations) - 1:
                    end_day = trip_duration  # Last destination gets remaining days
                
                day_label = f"Day {int(current_day)}" if int(current_day) == end_day else f"Days {int(current_day)}-{end_day}"
                nights = max(1, end_day - int(current_day) + 1)
                
                # Determine best activities for this destination based on categories
                dest_categories = dest.get('categories', ['General'])
                
                with st.expander(f"📍 {day_label}: **{dest['destination']}** | {dest['district'].title()} | {nights} night(s)", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Rating and reviews
                        st.markdown(f"""
                        **Quality Rating:** {get_category_emoji(dest['experience_category'])} **{dest['experience_category']}** ({dest['experience_score']:.3f})
                        
                        **Based on:** {int(dest['review_count'])} tourist reviews | **Sentiment:** {'😊 Positive' if dest['avg_sentiment'] > 0.2 else '😐 Mixed' if dest['avg_sentiment'] > -0.1 else '😟 Needs attention'}
                        
                        **Region:** {dest['region']}
                        """)
                        
                        # Category-specific activities
                        st.markdown("**🎯 Recommended Activities:**")
                        
                        activities = []
                        if 'Beaches' in dest_categories:
                            activities.extend([
                                "🏖️ Beach relaxation & swimming",
                                "🐢 Turtle watching (seasonal)",
                                "🏄 Water sports (surfing, snorkeling)"
                            ])
                        if 'Wildlife' in dest_categories:
                            activities.extend([
                                "🐘 Wildlife safari (early morning best)",
                                "🦜 Bird watching",
                                "📸 Photography tours"
                            ])
                        if 'Mountains' in dest_categories:
                            activities.extend([
                                "🥾 Hiking & trekking",
                                "🌄 Sunrise/sunset viewpoints",
                                "💧 Waterfall visits"
                            ])
                        if 'Temples & Culture' in dest_categories:
                            activities.extend([
                                "🛕 Temple visits (dress modestly)",
                                "📜 Historical site exploration",
                                "🎭 Cultural performances"
                            ])
                        if 'Tea & Plantations' in dest_categories:
                            activities.extend([
                                "🍃 Tea factory tour",
                                "☕ Tea tasting experience",
                                "📸 Scenic train rides"
                            ])
                        if 'Adventure' in dest_categories:
                            activities.extend([
                                "🚣 White water rafting",
                                "🧗 Rock climbing",
                                "🪂 Adventure sports"
                            ])
                        if 'Relaxation & Spa' in dest_categories:
                            activities.extend([
                                "💆 Ayurvedic spa treatment",
                                "🧘 Yoga & meditation",
                                "🌿 Nature walks"
                            ])
                        
                        # Default activities if none matched
                        if not activities:
                            activities = [
                                "🚶 Explore local area",
                                "🍛 Try local cuisine",
                                "📸 Photography"
                            ]
                        
                        for activity in activities[:4]:  # Show top 4 activities
                            st.write(f"  • {activity}")
                    
                    with col2:
                        st.markdown("**📊 Destination Stats:**")
                        st.write(f"🏨 Accommodations: {int(dest.get('num_accommodations', 0))}")
                        st.write(f"🍽️ Restaurants/Places: {int(dest.get('total_places', 0))}")
                        if dest.get('avg_accommodation_grade', 0) > 0:
                            grade_label = {4: 'A (Excellent)', 3: 'B (Good)', 2: 'C (Average)', 1: 'D (Basic)'}.get(round(dest.get('avg_accommodation_grade', 2)), 'N/A')
                            st.write(f"⭐ Avg Hotel Grade: {grade_label}")
                        
                        # Travel time to next destination (estimate)
                        if idx < len(selected_destinations) - 1:
                            next_dest = selected_destinations[idx + 1]
                            if dest['region'] == next_dest['region']:
                                travel_time = "1-2 hours"
                            elif dest['region'] in ['Colombo & Western'] or next_dest['region'] in ['Colombo & Western']:
                                travel_time = "3-4 hours"
                            else:
                                travel_time = "4-6 hours"
                            st.markdown(f"**🚗 To next stop:** ~{travel_time}")
                
                current_day = end_day + 1
            
            # =================================================================
            # TRIP SUMMARY
            # =================================================================
            
            st.markdown("---")
            st.markdown("### 📋 Trip Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Destinations by Category:**")
                category_counts = {}
                for dest in selected_destinations:
                    for cat in dest.get('categories', ['General']):
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                
                for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                    st.write(f"• {cat}: {count} destination(s)")
            
            with col2:
                st.markdown("**Regions Covered:**")
                region_counts = {}
                for dest in selected_destinations:
                    region = dest['region']
                    region_counts[region] = region_counts.get(region, 0) + 1
                
                for region, count in sorted(region_counts.items(), key=lambda x: -x[1]):
                    st.write(f"• {region}: {count} destination(s)")
            
            # Budget estimate
            st.markdown("---")
            st.markdown("### 💰 Estimated Budget")
            
            if travel_style == "Budget Backpacker":
                daily_cost = (30, 50)
            elif travel_style == "Luxury Seeker":
                daily_cost = (300, 500)
            elif travel_style == "Adventure Explorer":
                daily_cost = (50, 100)
            else:
                daily_cost = (80, 150)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Budget", f"${daily_cost[0] * trip_duration}")
            with col2:
                st.metric("Max Budget", f"${daily_cost[1] * trip_duration}")
            with col3:
                st.metric("Per Day", f"${daily_cost[0]}-{daily_cost[1]}")
            
            st.info("💡 **Tip:** Budget excludes international flights. Book accommodations and drivers in advance during peak season (Dec-Mar).")


# =============================================================================
# PAGE: RATING IMPROVEMENT ADVISOR (NEW)
# =============================================================================

elif page == "📋 Rating Improvement Advisor":
    st.title("📋 Rating Improvement Advisor")
    st.markdown("### Data-Driven Recommendations for Tourism Development")
    
    if not data_loaded or feature_names is None:
        st.error("Data or models not loaded.")
    else:
        st.markdown("""
        This tool analyzes destination/district data and provides **specific, actionable recommendations** 
        to improve tourism ratings. Ideal for **Tourism Boards** and **Destination Authorities**.
        """)
        
        analysis_level = st.radio("Analysis Level", ["🏝️ Single Destination", "📍 Entire District"], horizontal=True)
        
        if analysis_level == "🏝️ Single Destination":
            destinations = sorted(df['destination'].unique().tolist())
            selected_dest = st.selectbox("Select Destination", destinations)
            
            if st.button("📊 Analyze & Get Recommendations", type="primary", use_container_width=True):
                dest_row = df[df['destination'] == selected_dest].iloc[0]
                
                st.markdown("---")
                st.markdown(f"## 📊 Analysis: {selected_dest}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Rating", dest_row['experience_category'])
                with col2:
                    st.metric("Score", f"{dest_row['experience_score']:.3f}")
                with col3:
                    st.metric("Sentiment", f"{dest_row['avg_sentiment']:.3f}")
                with col4:
                    st.metric("Reviews", int(dest_row['review_count']))
                
                st.markdown("---")
                st.markdown("## 📋 Improvement Recommendations")
                
                recommendations = get_improvement_recommendations(dest_row, df, feature_names)
                
                for i, rec in enumerate(recommendations, 1):
                    priority_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(rec['priority'], '⚪')
                    
                    with st.expander(f"{priority_color} **{rec['area']}** (Priority: {rec['priority']})", expanded=(rec['priority'] == 'High')):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Current:** {rec['current']}")
                        with col2:
                            st.markdown(f"**Target:** {rec['target']}")
                        with col3:
                            st.markdown(f"**Gap:** {rec['gap']}")
                        
                        st.markdown("**Recommended Actions:**")
                        for action in rec['actions']:
                            st.write(f"• {action}")
        
        else:  # District analysis
            districts = sorted(df['district'].unique().tolist())
            selected_district = st.selectbox("Select District", districts)
            
            if st.button("📊 Analyze District", type="primary", use_container_width=True):
                district_df = df[df['district'] == selected_district]
                
                st.markdown("---")
                st.markdown(f"## 📊 District Analysis: {selected_district.title()}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Destinations", len(district_df))
                with col2:
                    st.metric("Avg Score", f"{district_df['experience_score'].mean():.3f}")
                with col3:
                    excellent_pct = (district_df['experience_category'] == 'Excellent').mean() * 100
                    st.metric("Excellent %", f"{excellent_pct:.1f}%")
                with col4:
                    poor_count = (district_df['experience_category'] == 'Poor').sum()
                    st.metric("Poor Rated", poor_count)
                
                st.markdown("### 📊 Category Distribution")
                cat_counts = district_df['experience_category'].value_counts()
                fig = px.pie(values=cat_counts.values, names=cat_counts.index, color=cat_counts.index, color_discrete_map=CATEGORY_COLORS)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### ⚠️ Destinations Needing Attention")
                poor_avg = district_df[district_df['experience_class'] <= 1].sort_values('experience_score')
                
                if len(poor_avg) > 0:
                    for _, dest in poor_avg.head(5).iterrows():
                        with st.expander(f"{get_category_emoji(dest['experience_category'])} {dest['destination']} ({dest['experience_score']:.3f})"):
                            recommendations = get_improvement_recommendations(dest, df, feature_names)
                            for rec in recommendations[:2]:
                                st.markdown(f"**{rec['area']}:** {', '.join(rec['actions'][:2])}")
                else:
                    st.success("All destinations in this district are rated Good or Excellent!")
                
                st.markdown("### 📋 District-Wide Recommendations")
                
                district_means = district_df.mean(numeric_only=True)
                excellent_means = df[df['experience_category'] == 'Excellent'].mean(numeric_only=True)
                
                if district_means.get('avg_sentiment', 0) < excellent_means.get('avg_sentiment', 0) - 0.1:
                    st.warning("**Sentiment Gap:** Focus on improving tourist satisfaction across the district through service quality training.")
                
                if district_means.get('num_accommodations', 0) < excellent_means.get('num_accommodations', 0) * 0.7:
                    st.warning("**Infrastructure Gap:** The district needs more accommodation options. Consider development incentives.")
                
                if district_means.get('total_places', 0) < excellent_means.get('total_places', 0) * 0.7:
                    st.warning("**Facilities Gap:** More tourist facilities (restaurants, attractions) would improve ratings.")


# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("*Sri Lanka Tourism Analyzer*")