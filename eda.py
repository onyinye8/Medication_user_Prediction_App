# Importing libraries
import pandas as pd     # To read our dataset
import numpy as np      # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns   # For statistical data visualization
import joblib           # For saving and loading models
from collections import Counter
#loading the dataset
Med_Dataset = pd.read_csv("C:/Users/owner/PycharmProjects/MedProject/Medicine_Details.csv")

#Displaying the first five rows
print(Med_Dataset.head)

#General Info about the dataset
print(Med_Dataset.info())

#statistical summary of the dataset
print(Med_Dataset.describe())

#Checking for missing data
print(Med_Dataset.isnull().sum())

#displaying the columns name of the DataFrame
print(Med_Dataset.columns)

#Visualization to understand our dataset

Med_Dataset_Columns = ['Medicine Name', 'Composition', 'Uses', 'Side_effects', 'Image URL', 'Manufacturer']

for c in Med_Dataset_Columns:
    value_counts = Med_Dataset[c].value_counts()

    # Print the value counts along with the column name
    print(f"Value counts for column '{c}':")
    print(value_counts)
    print("\n")  # Add a newline for better readability between output

# Identify the most common side effects and their associated medicines.
def get_most_common_side_effects(Med_Dataset):  # Function to extract and count side effects
    side_effects = []

    for index, row in Med_Dataset.iterrows():  # Loop through each row and split the side effects
        effects = str(row['Side_effects']).split(',')  # Split side effects by comma
        effects = [effect.strip() for effect in effects]
        side_effects.extend(effects)  # Add these effects to the main list

    side_effect_counts = Counter(side_effects)  # Count the occurrences of each side effect
    return side_effect_counts


def get_medicine_for_side_effects(Med_Dataset,
                                  side_effect_counts):  # Function to find the associated medicines for each side effect
    side_effect_medicines = {}

    for index, row in Med_Dataset.iterrows():  # to find which medicines correspond to each side effect in each row
        medicine_name = row['Medicine Name']
        effects = str(row['Side_effects']).split(',')  # Split side effects by comma
        effects = [effect.strip() for effect in effects]  # Clean side effects

        for effect in effects:
            if effect in side_effect_medicines:
                side_effect_medicines[effect].append(medicine_name)
            else:
                side_effect_medicines[effect] = [medicine_name]

    return side_effect_medicines


side_effect_counts = get_most_common_side_effects(Med_Dataset)  # Get the most common side effects
side_effect_medicines = get_medicine_for_side_effects(Med_Dataset,
                                                      side_effect_counts)  # Get the medicines associated with each side effect

print("Most Common Side Effects and Associated Medicines:")
for effect, count in side_effect_counts.most_common(10):  # Show top 10 side effects
    medicines = ", ".join(side_effect_medicines.get(effect, []))
    print(f"Side Effect: {effect}")
    print(f"Frequency: {count}")
    print(f"Medicines: {medicines}")
    print("-" * 50)

Num_Columns = ['Excellent Review %', 'Average Review %', 'Poor Review %']

# Loop through each numerical column
for n in Num_Columns:
    # Sort the dataset by the current column in descending order and take the top 50
    top_10 = Med_Dataset.sort_values(by=n, ascending=False).head(50)

    # Create a bar plot for the top 50 values
    sns.barplot(x='Medicine Name', y=n, data=top_10)

    # Set the title of the plot
    plt.title(f'Top 50 Medicines by {n}')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=90)

    # Display the plot
    plt.show()





