### INTRODUCTION TO BIG DATA ###
## NAME: RUHANIKA Alex  
## ID: 26627


### **Project Title: Global Smoking Trend Analysis and Cluster Identification**
<img width="938" height="422" alt="Dashboard" src="https://github.com/user-attachments/assets/e6705e6f-35e8-4d07-8a79-157d2ae1cbc0" />
# Dashboard

### **1. Introduction**

This capstone project provides a comprehensive data-driven analysis of global smoking trends using a public dataset from the OECD. The core objective is to move beyond simple descriptive statistics and identify distinct patterns in smoking rates across different countries over time. This is achieved by implementing a complete data analysis pipeline, from raw data processing to advanced machine learning and interactive data visualization.

The project follows a standard data science pipeline:
* **Data Preprocessing:** Cleaning and preparing the raw data for analysis.
* **Exploratory Data Analysis (EDA):** Visualizing trends and distributions to understand the data's characteristics.
* **Clustering:** Applying the K-Means machine learning algorithm to group countries with similar smoking trend trajectories.
* **Interactive Visualization:** Creating an interactive dashboard in Power BI to present the findings and allow for in-depth data exploration.

### **2. Project Structure and Deliverables**

The project is organized to provide a clear and reproducible workflow. Key deliverables and files include:

* **Raw Data:** `OECD.ELS.HD,DSD_HEALTH_LVNG@DF_HEALTH_LVNG_TC,1.0+.A......csv` - The original dataset on daily smoking percentages.
* **Analysis Scripts:** `cap.py` and `Tobacco.ipynb` - The Python script and its Jupyter Notebook counterpart that perform all data cleaning, analysis, and clustering.
* **Processed Data:**
    * `cleaned_data.csv`: A CSV file containing the raw data after initial cleaning and preprocessing.
    * `clustered_data.csv`: The final, enriched dataset that includes a new `Cluster` column, assigning each country to a group based on its smoking trends. This file is the primary source for the Power BI dashboard.
* **Visualizations:**
    * `daily_smokers_trends.jpg`: A line plot illustrating smoking trends for all countries over time.
    * `daily_smokers_distribution.png`: A box plot showing the distribution of smoking rates across all data points.
    * `elbow_method_for_k.png`: A plot generated during the K-Means analysis to determine the optimal number of clusters.
    * `clustered_trends.png`: A plot visualizing the smoking trends, with colors representing the identified clusters, highlighting the groups' distinct behaviors.
* **Dashboard:** `Dashboard.jpg` - A screenshot of the final interactive Power BI dashboard, which allows for dynamic filtering and exploration of the analysis results.

### **3. Methodology**

The project's analytical approach is divided into four main parts, with key code snippets from `cap.py` provided below.

#### **3.1 Data Cleaning and Preprocessing**
This step focused on transforming the raw data into a clean, usable format. Irrelevant columns were dropped, meaningful columns were renamed, and data types were corrected. The processed data was then saved to a new file, `cleaned_data.csv`.

**Code Snippet: Data Cleaning**
```python
def clean_data(df):
    cols_to_drop = [
        'STRUCTURE', 'STRUCTURE_ID', 'STRUCTURE_NAME', 'ACTION', 'FREQ',
        'MEASURE', 'UNIT_MEASURE', 'METHODOLOGY', 'OBS_STATUS', 'UNIT_MULT',
        'DECIMALS'
    ]
    
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop, errors='ignore')

    rename_dict = {
        'REF_AREA': 'Country',
        'TIME_PERIOD': 'Year',
        'OBS_VALUE': 'Daily_Smokers_Percentage'
    }
    df = df.rename(columns=rename_dict)
    
    df = df.dropna(subset=['Year', 'Daily_Smokers_Percentage'])
    df.to_csv('cleaned_data.csv', index=False)
    return df
```
## **3.2 Exploratory Data Analysis (EDA)**
EDA involved visualizing trends and distributions to uncover initial insights. A line plot was used to show smoking trends over time by country, and a box plot was used to visualize the distribution of smoking rates.

Code Snippet: Visualization of Trends
```python
def perform_eda(df):
    # ... (code for descriptive statistics)
    
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df, x='Year', y='Daily_Smokers_Percentage', hue='Country', marker='o')
    plt.title('Daily Smokers Percentage Over Time by Country')
    plt.xlabel('Year')
    plt.ylabel('Daily Smokers Percentage')
    plt.savefig('daily_smokers_trends.png')
    plt.close()

    # ... (code for box plot)
```

## **3.3 K-Means Clustering**
K-Means clustering was applied as an innovative method to group countries with similar smoking trend patterns. The Elbow Method was used to determine the optimal number of clusters.

Code Snippet: Elbow Method and Clustering
```python
def apply_model(df):
    df_pivot = df.pivot_table(index='Country', columns='Year', values='Daily_Smokers_Percentage')
    df_pivot_filled = df_pivot.apply(lambda row: row.interpolate(method='linear', limit_direction='both'), axis=1)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_pivot_filled)
    
    inertias = []
    if scaled_features.shape[0] > 1:
        for k in range(2, min(11, scaled_features.shape[0] + 1)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            inertias.append(kmeans.inertia_)
        plt.plot(range(2, min(11, scaled_features.shape[0] + 1)), inertias, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.savefig('elbow_method_for_k.png')
        plt.close()

    if scaled_features.shape[0] >= 3:
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        df_pivot_filled['Cluster'] = clusters
        df_pivot_with_clusters = df_pivot_filled.reset_index()[['Country', 'Cluster']]
        
        df_clustered = df.merge(df_pivot_with_clusters, on='Country', how='left')
        df_clustered.to_csv('clustered_data.csv', index=False)
        
        return df_clustered
```
## **4. Key Insights**
Three Distinct Clusters: The K-Means algorithm successfully partitioned countries into three groups with different smoking rate trajectories over the years, demonstrating a powerful way to categorize countries with similar public health challenges and successes.

Overall Downward Trend: Most countries in the dataset show a general decline in daily smoking percentages, indicating a positive public health trend.

Clustering reveals nuanced patterns: The clusters reveal that while a downward trend is common, the rate and shape of that decline differ significantly by country, providing a deeper level of insight than a simple average.

## **5. Tools and Technologies**
Python: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Environment: Jupyter Notebook, VS Code

Business Intelligence: Power BI
