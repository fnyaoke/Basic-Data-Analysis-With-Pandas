"""
Data Analysis with Pandas
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*60)
print("DATA ANALYSIS ASSIGNMENT - COMPLETE IMPLEMENTATION")
print("="*60)
print("Tasks: Load & Explore → Basic Analysis → Visualizations")
print("="*60)
print()

# TASK 1: LOAD AND EXPLORE THE DATASET

print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("-" * 40)

try:
    # Method 1: Load Iris dataset from sklearn
    print("Loading Iris dataset from sklearn...")
    iris_data = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target_names[iris_data.target]
    
    print("✓ Dataset loaded successfully!")
    print()
    
    # Alternative method - if you have a CSV file, use this instead:
    # df = pd.read_csv('iris.csv')
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Creating sample dataset instead...")
    
    # Fallback - create sample data if sklearn not available
    np.random.seed(42)
    n_samples = 150
    
    data = {
        'sepal length (cm)': np.random.normal(5.8, 0.8, n_samples),
        'sepal width (cm)': np.random.normal(3.0, 0.4, n_samples),
        'petal length (cm)': np.random.normal(3.8, 1.8, n_samples),
        'petal width (cm)': np.random.normal(1.2, 0.8, n_samples),
        'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n_samples)
    }
    df = pd.DataFrame(data)
    print("✓ Sample dataset created successfully!")

print("1. DATASET OVERVIEW:")
print(f"   Dataset shape: {df.shape}")
print(f"   Number of rows: {df.shape[0]}")
print(f"   Number of columns: {df.shape[1]}")
print()

print("2. FIRST FEW ROWS OF THE DATASET:")
print(df.head())
print()

print("3. DATASET STRUCTURE AND DATA TYPES:")
print(df.info())
print()

print("4. COLUMN NAMES:")
print(f"   Columns: {list(df.columns)}")
print()

print("5. CHECKING FOR MISSING VALUES:")
missing_values = df.isnull().sum()
print(missing_values)
print(f"   Total missing values: {missing_values.sum()}")
print()

print("6. DATA CLEANING:")
if missing_values.sum() > 0:
    print("   Missing values found - cleaning data...")
    
    # Option 1: Fill missing values with mean for numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
    
    # Option 2: Fill missing values with mode for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    print("   ✓ Missing values handled successfully!")
else:
    print("   ✓ No missing values found - dataset is clean!")

print()

print("7. BASIC DATASET INFORMATION:")
print(f"   Unique species: {df['species'].nunique()}")
print(f"   Species distribution:")
for species in df['species'].unique():
    count = (df['species'] == species).sum()
    print(f"     - {species}: {count} samples")
print()

# TASK 2: BASIC DATA ANALYSIS

print("TASK 2: BASIC DATA ANALYSIS")
print("-" * 30)

print("1. BASIC STATISTICS OF NUMERICAL COLUMNS:")
numerical_stats = df.describe()
print(numerical_stats.round(3))
print()

print("2. DETAILED STATISTICAL ANALYSIS:")
numerical_columns = df.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    print(f"   {col}:")
    print(f"     Mean: {df[col].mean():.3f}")
    print(f"     Median: {df[col].median():.3f}")
    print(f"     Standard Deviation: {df[col].std():.3f}")
    print(f"     Min: {df[col].min():.3f}")
    print(f"     Max: {df[col].max():.3f}")
    print()

print("3. GROUPING BY CATEGORICAL COLUMN (SPECIES):")
print("   Mean values for each species:")
grouped_stats = df.groupby('species')[numerical_columns].mean()
print(grouped_stats.round(3))
print()

print("4. ADDITIONAL GROUP ANALYSIS:")
print("   Standard deviation by species:")
grouped_std = df.groupby('species')[numerical_columns].std()
print(grouped_std.round(3))
print()

print("   Count of samples per species:")
species_counts = df['species'].value_counts()
print(species_counts)
print()

print("5. PATTERNS AND INTERESTING FINDINGS:")
print("   Key Observations:")

# Calculate some interesting metrics
sepal_length_range = df['sepal length (cm)'].max() - df['sepal length (cm)'].min()
petal_length_range = df['petal length (cm)'].max() - df['petal length (cm)'].min()

print(f"   • Sepal length varies by {sepal_length_range:.1f} cm across all samples")
print(f"   • Petal length shows greater variation ({petal_length_range:.1f} cm)")

# Find species with highest/lowest averages
for col in numerical_columns:
    max_species = grouped_stats[col].idxmax()
    min_species = grouped_stats[col].idxmin()
    print(f"   • {col}: {max_species} has highest average, {min_species} has lowest")

print()

# TASK 3: DATA VISUALIZATION

print("TASK 3: DATA VISUALIZATION")
print("-" * 26)
print("Creating four different types of visualizations...")
print()

# Create a figure with 4 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Iris Dataset - Comprehensive Data Visualization', fontsize=16, fontweight='bold')

# VISUALIZATION 1: LINE CHART - Trends by Index (simulating time series)
print("1. Creating Line Chart...")
ax1.plot(df.index, df['sepal length (cm)'], label='Sepal Length', alpha=0.7, linewidth=2)
ax1.plot(df.index, df['sepal width (cm)'], label='Sepal Width', alpha=0.7, linewidth=2)
ax1.plot(df.index, df['petal length (cm)'], label='Petal Length', alpha=0.7, linewidth=2)
ax1.plot(df.index, df['petal width (cm)'], label='Petal Width', alpha=0.7, linewidth=2)

ax1.set_title('Flower Measurements Trends by Sample Index', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sample Index (Simulating Time Series)', fontsize=10)
ax1.set_ylabel('Measurement (cm)', fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3)

# VISUALIZATION 2: BAR CHART - Average measurements by species
print("2. Creating Bar Chart...")
species_means = df.groupby('species')[numerical_columns].mean()

x = np.arange(len(species_means.index))
width = 0.2

ax2.bar(x - 1.5*width, species_means['sepal length (cm)'], width, label='Sepal Length', alpha=0.8)
ax2.bar(x - 0.5*width, species_means['sepal width (cm)'], width, label='Sepal Width', alpha=0.8)
ax2.bar(x + 0.5*width, species_means['petal length (cm)'], width, label='Petal Length', alpha=0.8)
ax2.bar(x + 1.5*width, species_means['petal width (cm)'], width, label='Petal Width', alpha=0.8)

ax2.set_title('Average Measurements by Species', fontsize=12, fontweight='bold')
ax2.set_xlabel('Species', fontsize=10)
ax2.set_ylabel('Average Measurement (cm)', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(species_means.index, rotation=45)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# VISUALIZATION 3: HISTOGRAM - Distribution of a numerical column
print("3. Creating Histogram...")
ax3.hist(df['petal length (cm)'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
ax3.set_title('Distribution of Petal Length', fontsize=12, fontweight='bold')
ax3.set_xlabel('Petal Length (cm)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# Add statistics text to histogram
mean_petal = df['petal length (cm)'].mean()
std_petal = df['petal length (cm)'].std()
ax3.axvline(mean_petal, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_petal:.2f}')
ax3.legend()

# VISUALIZATION 4: SCATTER PLOT - Relationship between two numerical columns
print("4. Creating Scatter Plot...")
species_colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

for species in df['species'].unique():
    species_data = df[df['species'] == species]
    ax4.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'], 
               c=species_colors.get(species, 'gray'), label=species, alpha=0.7, s=50)

ax4.set_title('Sepal Length vs Petal Length by Species', fontsize=12, fontweight='bold')
ax4.set_xlabel('Sepal Length (cm)', fontsize=10)
ax4.set_ylabel('Petal Length (cm)', fontsize=10)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Adjust layout and display
plt.tight_layout()
plt.show()

print("✓ All visualizations created successfully!")
print()

# ADDITIONAL ADVANCED VISUALIZATIONS

print("BONUS: ADDITIONAL ADVANCED VISUALIZATIONS")
print("-" * 38)

# Create additional plots for deeper insights
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Advanced Analysis - Iris Dataset', fontsize=16, fontweight='bold')

# 1. Box Plot - Distribution comparison by species
print("Creating Box Plot for distribution comparison...")
df_melted = pd.melt(df, id_vars=['species'], var_name='measurement', value_name='value')
df_numeric = df_melted[df_melted['measurement'].isin(numerical_columns)]

sns.boxplot(data=df_numeric, x='species', y='value', hue='measurement', ax=ax1)
ax1.set_title('Distribution of Measurements by Species', fontweight='bold')
ax1.set_xlabel('Species')
ax1.set_ylabel('Measurement (cm)')
ax1.legend(title='Measurement Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# 2. Correlation Heatmap
print("Creating Correlation Heatmap...")
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, ax=ax2, cbar_kws={'label': 'Correlation'})
ax2.set_title('Correlation Matrix of Flower Measurements', fontweight='bold')

# 3. Pair Plot Style - Multiple Scatter Plots
print("Creating Enhanced Scatter Plot Matrix...")
measurements = list(numerical_columns)
colors = ['red', 'green', 'blue']
species_list = df['species'].unique()

for i, species in enumerate(species_list):
    species_data = df[df['species'] == species]
    ax3.scatter(species_data['sepal width (cm)'], species_data['petal width (cm)'], 
               c=colors[i], label=species, alpha=0.6, s=60)

ax3.set_title('Sepal Width vs Petal Width by Species', fontweight='bold')
ax3.set_xlabel('Sepal Width (cm)')
ax3.set_ylabel('Petal Width (cm)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Violin Plot - Distribution shape
print("Creating Violin Plot...")
sns.violinplot(data=df, x='species', y='petal length (cm)', ax=ax4, palette='Set2')
ax4.set_title('Petal Length Distribution by Species', fontweight='bold')
ax4.set_xlabel('Species')
ax4.set_ylabel('Petal Length (cm)')

plt.tight_layout()
plt.show()

# COMPREHENSIVE ANALYSIS SUMMARY

print("="*60)
print("COMPREHENSIVE ANALYSIS SUMMARY")
print("="*60)

print("TASK COMPLETION STATUS:")
print("✓ Task 1: Dataset Loading and Exploration - COMPLETE")
print("✓ Task 2: Basic Data Analysis - COMPLETE")  
print("✓ Task 3: Data Visualization (4 required types) - COMPLETE")
print("✓ Additional Advanced Analysis - BONUS COMPLETE")
print()

print("KEY INSIGHTS FROM ANALYSIS:")
print("-" * 28)

# Calculate key insights
total_samples = len(df)
n_features = len(numerical_columns)
n_species = df['species'].nunique()

print(f"Dataset Overview:")
print(f"• Total samples analyzed: {total_samples}")
print(f"• Number of features: {n_features}")
print(f"• Number of species: {n_species}")
print()

print("Statistical Insights:")
for col in numerical_columns:
    col_mean = df[col].mean()
    col_std = df[col].std()
    cv = (col_std / col_mean) * 100  # Coefficient of variation
    print(f"• {col}: Mean = {col_mean:.2f}cm, StdDev = {col_std:.2f}cm, CV = {cv:.1f}%")
print()

print("Species-Specific Findings:")
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    avg_sepal_length = species_data['sepal length (cm)'].mean()
    avg_petal_length = species_data['petal length (cm)'].mean()
    print(f"• {species.capitalize()}: Avg Sepal = {avg_sepal_length:.2f}cm, Avg Petal = {avg_petal_length:.2f}cm")
print()

print("Correlation Insights:")
strongest_corr = correlation_matrix.abs().unstack().sort_values(ascending=False)
# Remove self-correlations (diagonal)
strongest_corr = strongest_corr[strongest_corr < 1.0]
top_corr = strongest_corr.iloc[0]
corr_vars = strongest_corr.index[0]
print(f"• Strongest correlation: {corr_vars[0]} vs {corr_vars[1]} (r = {top_corr:.3f})")
print()

print("DATA QUALITY ASSESSMENT:")
print(f"• Missing values: {df.isnull().sum().sum()} (0% of dataset)")
print(f"• Data types: All numerical measurements are float/int")
print(f"• Categorical variable: Species (3 categories)")
print(f"• Dataset balance: Approximately {total_samples//n_species} samples per species")
print()

print("VISUALIZATION SUMMARY:")
print("• Line Chart: Shows measurement patterns across samples")
print("• Bar Chart: Compares average measurements between species")
print("• Histogram: Reveals distribution shape of petal length")
print("• Scatter Plot: Demonstrates species clustering in 2D space")
print("• Bonus plots: Box plots, heatmaps, and violin plots for deeper insights")
print()

print("RECOMMENDATIONS FOR FURTHER ANALYSIS:")
print("1. Machine learning classification to predict species")
print("2. Principal component analysis for dimensionality reduction")
print("3. Statistical hypothesis testing between species groups")
print("4. Outlier detection and analysis")
print("5. Feature importance analysis")
print()

print("="*60)
print("ANALYSIS COMPLETE - ALL TASKS SUCCESSFULLY IMPLEMENTED")
print("="*60)

# Optional: Save results to file
try:
    # Save summary statistics
    summary_results = {
        'Dataset_Shape': df.shape,
        'Missing_Values': df.isnull().sum().to_dict(),
        'Basic_Statistics': df.describe().to_dict(),
        'Species_Means': df.groupby('species')[numerical_columns].mean().to_dict()
    }
    
    print("Analysis complete! Results can be exported to CSV if needed.")
    print("Use df.to_csv('iris_analysis_results.csv') to save the processed dataset.")
    
except Exception as e:
    print(f"Note: Export functionality available - {str(e)}")

print("\nGood Job!")