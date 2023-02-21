import pandas as pd
import numpy as np

patent_df = pd.read_csv("data/apat63_99.txt", dtype={"PATENT": np.int64})
patent_df = patent_df.fillna("")


cite_df = pd.read_csv("data/cite75_99.txt")
cite_df = cite_df.fillna("")

patent_num_to_ids = {}
edge_rows = set()
id_num = 0
cap = 300000

seed_connection = cite_df.iloc[1000000]
citing = seed_connection["CITING"]
cited = seed_connection["CITED"]
patent_num_to_ids[citing] = id_num
id_num += 1
patent_num_to_ids[cited] = id_num
id_num += 1

while True:
    if id_num >= cap:
        break
    for i, row in cite_df.iterrows():
        citing = row["CITING"]
        cited = row["CITED"]
        if id_num >= cap:
            break
        if citing not in patent_num_to_ids and cited not in patent_num_to_ids:
            # Only take connections that are fully connected to the seeded graph
            continue
        if citing not in patent_num_to_ids:
            patent_num_to_ids[citing] = id_num
            print(id_num)
            id_num += 1
        if cited not in patent_num_to_ids:
            patent_num_to_ids[cited] = id_num
            print(id_num)
            id_num += 1

        edge_rows.add((patent_num_to_ids[citing], patent_num_to_ids[cited]))

# Final pass to make sure all edges make it
for i, row in cite_df.iterrows():
    citing = row["CITING"]
    cited = row["CITED"]

    if citing not in patent_num_to_ids or cited not in patent_num_to_ids:
        # Only take connections that are fully connected to the seeded graph
        continue

    edge_rows.add((patent_num_to_ids[citing], patent_num_to_ids[cited]))

print("Saving edges...")
print(len(edge_rows))
save_df = pd.DataFrame(edge_rows, columns=['node_1', 'node_2'])
save_df.to_csv("patent_edges.csv", index=False)

patent_df["id"] = -1
print("Filling in ID numbers...")
for i, row in patent_df.iterrows():
    patent_num = row["PATENT"]
    if patent_num in patent_num_to_ids:
        patent_df.at[i, "id"] = patent_num_to_ids[patent_num]

patent_df.to_csv("patent_data_with_ids.csv", index=False)

