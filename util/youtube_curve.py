import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("data.csv")

# --- Apply your filters ---
df = df[df["subscriber_count"] >= 10]
df = df[df["view_count"] >= 10]
df = df[df["hours_since_upload"] <= 40 * 24 + 23]

# Convert hours to days
df["day"] = df["hours_since_upload"] / 24

# Create day bins (integer days)
df["day_bin"] = df["day"].astype(int)

# Compute average view count per day
avg_curve = df.groupby("day_bin")["view_count"].mean().reset_index()

# Plot
plt.figure(figsize=(8, 5))
plt.plot(avg_curve["day_bin"], avg_curve["view_count"], marker="o")
plt.xlabel("Day Since Upload")
plt.ylabel("Average Views")
plt.title("Average YouTube View Curve")
plt.grid(True)
plt.savefig("average_view_curve.png", dpi=300, bbox_inches="tight")
plt.show()
