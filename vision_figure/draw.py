import pandas as pd
import matplotlib.pyplot as plt

# # Data
# data = {
#     "Methods": ["Our", "CNN-ER", "Co-Attention", "Gray et.al", "Peng2023", 
#              "AlexNet", "Resnet-18", "ResNeXt-50", "GPNet", "HMTNet"],
#     "PC": [0.9072, 0.913, 0.8927, 0.7194, 0.7977, 0.8298, 0.8513, 0.8777, 0.9034, 0.8783],
#     "Parameter": [0.36, 52.29, 7.4, 0.08, 0.28, 2.53, 11.18, 22.98, 321.63, 101.78],
#     "TestTime":[1.77, 130.89, 5.21, 4, 1.35, 4.05, 3.98, 3.89, 10.79, 46.46 ]
# }

# df = pd.DataFrame(data)

# # Plotting Parameter vs PC (First figure)
# fig_param, ax_param = plt.subplots(figsize=(10, 6))

# # Scatter plot
# ax_param.scatter(df["Parameter"], df["PC"], label=df["Methods"], marker='o', color='blue')
# ax_param.scatter(df["Parameter"][0], df["PC"][0], color='red', s=100, zorder=5)  # Larger red point

# # Annotate each point with the model name
# for i, row in df.iterrows():
#     # Adjust annotation position to the right or left of the point
#     if row["Methods"] == "Gray et.al":
#         ax_param.text(row["Parameter"] + 0.01 , row["PC"], row["Methods"], fontsize=12, ha='left', va='center')
#     elif row["Methods"] == "Peng2023":
#         ax_param.text(row["Parameter"] - 0.02 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
#     elif row["Methods"] == "Our":
#         ax_param.text(row["Parameter"] - 0.02 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
#     elif row["Methods"] == "CNN-ER":
#         ax_param.text(row["Parameter"] - 1 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
#     elif row["Methods"] == "HMTNet":
#         ax_param.text(row["Parameter"] - 5 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
#     elif row["Methods"] == "GPNet":
#         ax_param.text(row["Parameter"] - 15 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
#     else:
#         ax_param.text(row["Parameter"] - 0.5, row["PC"], row["Methods"], fontsize=12, ha='right', va='center')

# ax_param.set_xlabel("Parameter (M)")
# ax_param.set_ylabel("PC")
# ax_param.set_title("Model Comparison: PC vs Parameter Size")
# ax_param.set_xscale('log')  # Log scale for parameter size
# ax_param.set_yscale('linear')

# # Save Parameter vs PC plot
# fig_param.tight_layout()
# fig_param.savefig('PC_vs_Parameter_Size.png')

# # Plotting Test Time vs PC (Second figure)
# fig_test_time, ax_test_time = plt.subplots(figsize=(10, 6))

# # Scatter plot for Test Time vs PC
# ax_test_time.scatter(df["TestTime"], df["PC"], label=df["Methods"], marker='o', color='blue')
# ax_test_time.scatter(df["TestTime"][0], df["PC"][0], color='red', s=100, zorder=5)  # Larger red point

# # Annotate each point with the model name
# for i, row in df.iterrows():
#     # Check if the current method is "Peng2023", adjust annotation to the right
#     if row["Methods"] == "Peng2023":
#         ax_test_time.text(row["TestTime"] + 0.1, row["PC"], row["Methods"], fontsize=12, ha='left', va='center')
#     # elif row["Methods"] == "CNN-ER":
#     #     ax_param.text(row["TestTime"] - 15 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
#     # elif row["Methods"] == "HMTNet":
#     #     ax_param.text(row["TestTime"] - 5 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
#     # elif row["Methods"] == "GPNet":
#     #     ax_param.text(row["TestTime"] - 1 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
#     else:
#         ax_test_time.text(row["TestTime"] - 0.1, row["PC"], row["Methods"], fontsize=12, ha='right', va='center')

# ax_test_time.set_xlabel("Test Time (ms)")
# ax_test_time.set_ylabel("PC")
# ax_test_time.set_title("Model Comparison: PC vs Test Time")
# ax_test_time.set_xscale('log')  # Log scale for test time
# ax_test_time.set_yscale('linear')

# # Save Test Time vs PC plot
# fig_test_time.tight_layout()
# fig_test_time.savefig('PC_vs_Test_Time.png')


######################R2
# Data

data = {
    "Methods": ["Our", "AlexNet", "Resnet-18", "ResNeXt-50", "R3CNN", 
             "GPNet", "CNN-ER", "Co-attention", "Gray et.al"],
    "PC": [0.9145, 0.8634, 0.8900, 0.8997, 0.9096, 0.9113, 0.9028, 0.9066, 0.7623],
    "Parameter": [0.44, 2.53, 11.18, 22.98, 23.05, 321.63, 52.29, 7.40, 0.08],
    "TestTime": [2.33, 4.88, 5.32, 5.32, 64.19, 29.54, 47.47, 5.54, 5.21]
}

df = pd.DataFrame(data)

# Plotting Parameter vs PC (First figure)
fig_param, ax_param = plt.subplots(figsize=(10, 6))

# Scatter plot
ax_param.scatter(df["Parameter"], df["PC"], label=df["Methods"], marker='o', color='blue')
ax_param.scatter(df["Parameter"][0], df["PC"][0], color='red', s=100, zorder=5)  # Larger red point

# Annotate each point with the model name
for i, row in df.iterrows():
    # Adjust annotation position to the right or left of the point
    if row["Methods"] == "Gray et.al":
        ax_param.text(row["Parameter"] + 0.01 , row["PC"], row["Methods"], fontsize=12, ha='left', va='center')
    elif row["Methods"] == "Peng2023":
        ax_param.text(row["Parameter"] - 0.02 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
    elif row["Methods"] == "Our":
        ax_param.text(row["Parameter"] - 0.02 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
    elif row["Methods"] == "CNN-ER":
        ax_param.text(row["Parameter"] - 1 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
    elif row["Methods"] == "GPNet":
        ax_param.text(row["Parameter"] - 15 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
    else:
        ax_param.text(row["Parameter"] - 0.5, row["PC"], row["Methods"], fontsize=12, ha='right', va='center')

ax_param.set_xlabel("Parameter (M)")
ax_param.set_ylabel("PC")
ax_param.set_title("Model Comparison: PC vs Parameter Size")
ax_param.set_xscale('log')  # Log scale for parameter size
ax_param.set_yscale('linear')

# Save Parameter vs PC plot
fig_param.tight_layout()
fig_param.savefig('r2_PC_vs_Parameter_Size.png')

# Plotting Test Time vs PC (Second figure)
fig_test_time, ax_test_time = plt.subplots(figsize=(10, 6))

# Scatter plot for Test Time vs PC
ax_test_time.scatter(df["TestTime"], df["PC"], label=df["Methods"], marker='o', color='blue')
ax_test_time.scatter(df["TestTime"][0], df["PC"][0], color='red', s=100, zorder=5)  # Larger red point

# Annotate each point with the model name
for i, row in df.iterrows():
    # Check if the current method is "Peng2023", adjust annotation to the right
    if row["Methods"] == "Peng2023":
        ax_test_time.text(row["TestTime"] + 0.1, row["PC"], row["Methods"], fontsize=12, ha='left', va='center')
    # elif row["Methods"] == "CNN-ER":
    #     ax_param.text(row["TestTime"] - 15 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
    # elif row["Methods"] == "HMTNet":
    #     ax_param.text(row["TestTime"] - 5 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
    # elif row["Methods"] == "GPNet":
    #     ax_param.text(row["TestTime"] - 1 , row["PC"], row["Methods"], fontsize=12, ha='right', va='center')
    else:
        ax_test_time.text(row["TestTime"] - 0.1, row["PC"], row["Methods"], fontsize=12, ha='right', va='center')

ax_test_time.set_xlabel("Test Time (ms)")
ax_test_time.set_ylabel("PC")
ax_test_time.set_title("Model Comparison: PC vs Test Time")
ax_test_time.set_xscale('log')  # Log scale for test time
ax_test_time.set_yscale('linear')

# Save Test Time vs PC plot
fig_test_time.tight_layout()
fig_test_time.savefig('r2_PC_vs_Test_Time.png')
