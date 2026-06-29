import pandas as pd

df = pd.read_csv(
    r"C:\Users\RBRap\OneDrive\Desktop\Project\Backend\Social_view\Data\social_features.csv"
)

print("Total players:", df.shape[0])
print("Sample player IDs:")
print(df["player_id"].head(20).tolist())
