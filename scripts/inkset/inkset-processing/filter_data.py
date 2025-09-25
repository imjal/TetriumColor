import os
import pandas as pd

ansari_path = "../../data/inksets/ansari/ansari-inks.csv"
fp_path = "../../data/inksets/fp_inks/all_inks.csv"

ansari_df = pd.read_csv(ansari_path)
fp_df = pd.read_csv(fp_path)

# Ensure both have an 'Index' column; if not, add one
if 'Index' not in ansari_df.columns:
    ansari_df.insert(0, 'Index', range(len(ansari_df)))
if 'Index' not in fp_df.columns:
    fp_df.insert(0, 'Index', range(len(fp_df)))

# Offset the FP indices by the length of Ansari
fp_df['Index'] = fp_df['Index'] + len(ansari_df)

# Concatenate the two DataFrames
combined_df = pd.concat([ansari_df, fp_df], ignore_index=True)

# Save to a new CSV
output_csv_path = "../../data/inksets/combined/combined_inks.csv"
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
combined_df.to_csv(output_csv_path, index=False)


# Read the list of top FP inks from the text file
top_fp_inks_path = "./results/FP_k4/top_50_inks_FP.txt"
top_fp_inks = []
with open(top_fp_inks_path, "r") as f:
    for line in f:
        line = line.strip()
        # Skip empty lines and lines that are headers or separators
        if not line or line.startswith("Top") or set(line) == {"="}:
            continue
        # Parse lines like " 1. inkname"
        parts = line.split(".", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            inkname = parts[1].strip()
            if inkname:
                top_fp_inks.append(inkname)

# Filter fp_df to only keep rows whose 'Name' is in top_fp_inks
filtered_fp_df = fp_df[fp_df['Name'].isin(top_fp_inks)].copy()

# Recompute the Index for filtered_fp_df to be consecutive after ansari_df
filtered_fp_df['Index'] = range(len(ansari_df), len(ansari_df) + len(filtered_fp_df))

# Concatenate ansari_df and filtered_fp_df
filtered_combined_df = pd.concat([ansari_df, filtered_fp_df], ignore_index=True)

# Save to a new CSV
filtered_combined_no_index = filtered_combined_df.drop(columns=['ID'])
filtered_output_no_index_csv_path = "../../data/inksets/combined/combined_inks_filtered.csv"
# Add a 'clogged' column with all 0s to the filtered_combined_no_index DataFrame
filtered_combined_no_index['clogged'] = 0
# Save again with the clogged column
filtered_combined_no_index.to_csv(filtered_output_no_index_csv_path, index=False)
print(f"Filtered combined CSV with 'clogged' column saved to {filtered_output_no_index_csv_path}")
