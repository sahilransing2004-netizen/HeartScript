from ucimlrepo import fetch_ucirepo

heart = fetch_ucirepo(id=45)
df = heart.data.features.copy()
df["target"] = (heart.data.targets["num"] > 0).astype(int)

print("Shape:", df.shape)
print("Missing values:")
print(df.isna().sum()[df.isna().sum() > 0])
print("Target counts (0 = no disease, 1 = disease):")
print(df["target"].value_counts())

df.to_csv("heart.csv", index=False)
print("Saved heart.csv")
