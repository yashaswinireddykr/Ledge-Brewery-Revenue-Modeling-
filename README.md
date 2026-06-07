# Pour Decisions: How Live Music and Weather Shape Brewery Revenue in New Hampshire

*A comprehensive data analysis project examining the impact of live music events and weather conditions on craft brewery revenue using machine learning and statistical modeling.*

---

## Project Overview

This project analyzes transaction records from Ledge Brewing Co., a craft brewery in North Conway, New Hampshire, to understand how live music events and weather conditions influence daily revenue. Using multiple machine learning models and statistical analysis, we quantify the financial impact of entertainment programming and weather patterns on customer behavior.

### Key Research Questions

**Live Music Analysis:**
- Does live music significantly increase daily revenue?
- How do weekends (Friday-Sunday) influence revenue independently of music events?
- What is the optimal band-day combination for maximum ROI?
- Do events attract new customers or shift existing visit timing?

**Weather Impact Analysis:**
- How do snow, rain, and temperature affect daily sales?
- Does Boston weather vs. local NH weather better predict customer behavior?
- Can weather data improve revenue forecasting accuracy?

---

## Key Findings

### Live Music Impact

- **Live music adds $879.74 to daily revenue** on average
- **Weekend days (Fri-Sun) boost revenue by $2,623.09** independently
- **Combined effect**: Weekend + Music = $3,504 more than weekday without music
- **Statistical significance**: p < 0.001 for both predictors (t-statistic = 8.18 for music effect)

### Weather Effects

- **Boston weather data** provides better predictive power than local North Conway weather
- **Rainy conditions** correlate with higher revenue (customers prefer indoor activities)
- **Snow conditions** significantly decrease revenue (-$80.41 per hour)
- **Sunny weather** shows modest positive impact (+$16.49 per hour)

### Artist Performance Analysis

- **Band M017**: Most active performer (24 Friday shows) — reliable Friday draw
- **Saturday dominance**: Multiple bands (M004, M038, M011, M022) show strong Saturday performance
- **Sunday performers**: M026, M002, M010 demonstrate consistent Sunday programming
- **Cross-day flexibility**: Band M022 successfully performs on both Thursday and Saturday

### Genre Analysis

- **Blues**: Highest median revenue (~$5,500) with consistent performance
- **Rock**: Highest revenue ceiling (up to $15,500) but more variable
- **Solo acts**: Most predictable revenue stream around $4,200

---

## Revenue Breakdown

| Category | Mean Revenue ($) | Median Revenue ($) | Event Count |
|---|---|---|---|
| Weekday without Music | $1,282.71 | $1,010.49 | 277 |
| Weekday with Music | $2,144.80 | $1,696.37 | 7 |
| Weekend without Music | $3,904.40 | $3,096.13 | 126 |
| Weekend with Music | $4,786.38 | $4,299.99 | 94 |

**Overall Music Impact:**
- **Days with Music**: $4,603.30 average (101 events)
- **Days without Music**: $2,102.39 average (403 events)
- **Revenue Difference**: $2,500.91 per music day

---

## Technical Implementation

### Data Sources

- **Transaction Data**: 49,866 rows covering June 2023 – November 2024
- **Live Music Data**: Event schedules, artist details, genres, performance rates
- **Weather Data**: Hourly data from both Boston and North Conway sources
- **Features**: Precipitation, temperature, visibility, cloud cover

### Machine Learning Models

**Live Music Analysis:**
- **Multiple Linear Regression**: Quantify independent effects of music and weekends
- **Statistical Testing**: Welch's t-test for significance validation
- **Feature Engineering**: Binary indicators for music presence and weekend status

**Weather Analysis:**
- **Linear Regression**: Interpretable coefficients for business insights
- **Random Forest Regressor**: Feature importance and non-linear relationships
- **Decision Tree Regressor**: Clear decision paths for operational planning

### Model Performance Comparison

| Model | R² (North Conway) | R² (Boston) | MAE (Boston) | RMSE (Boston) |
|---|---|---|---|---|
| Linear Regression | 0.407 | **0.428** | $211.97 | $321.36 |
| Decision Tree | 0.385 | **0.433** | $211.75 | $319.48 |
| Random Forest | 0.328 | 0.357 | $224.80 | $340.74 |

*Boston weather data consistently outperformed local weather in predictive accuracy*

---

## Analysis Workflow

**1. Data Preprocessing**
- Transaction aggregation: hourly and daily revenue summaries
- Weather feature engineering: binary flags for rain, snow, sunny conditions
- Time series preparation: lagged revenue variables for momentum analysis
- Data integration: merging transaction, weather, and event data

**2. Exploratory Data Analysis**
- Revenue patterns by day of week and music presence
- Weather impact visualization across different conditions
- Genre performance analysis and artist effectiveness
- Seasonal trend identification and customer behavior patterns

**3. Statistical Modeling**
- Multiple regression analysis for live music impact quantification
- Cross-validation for model reliability assessment
- Feature importance analysis using Random Forest
- Residual analysis for model assumption validation

**4. Business Intelligence**
- ROI calculations for different music genres and artists
- Weather-based operational recommendations
- Revenue forecasting for planning and budgeting

---

## Business Recommendations

### Live Music Strategy
1. **Prioritize weekend events (Friday-Sunday)** — Weekends show 3x higher revenue than weekdays
2. **Focus on Friday performances** — Band M017 performs 24 Friday shows, indicating strong demand
3. **Expand Saturday programming** — Multiple successful bands show consistent Saturday performance
4. **Consider Thursday shows** — Band M022 shows success on Thursday, potential for expansion
5. **Leverage high-performing artists** — M017, M004, and M038 show consistent booking patterns

### Weather-Informed Operations
1. **Use Boston weather forecasts** for staffing and event planning
2. **Schedule events during mild rain** — Higher indoor customer activity
3. **Avoid snow day events** — Significant negative impact on turnout
4. **Deploy weather-triggered promotions** — "Rainy Day Specials"
5. **Optimize inventory** based on weather-driven demand patterns

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightgrey?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-green)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-teal)

---

*This project demonstrates the power of data science in small business optimization, showing how statistical analysis and machine learning can provide actionable insights for revenue growth in the hospitality industry.*
