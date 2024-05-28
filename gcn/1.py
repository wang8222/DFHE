
import re

import re

import re

def extract_embeddings(embeddings_file, output_file_users, output_file_businesses):
    with open(embeddings_file, 'r') as f:
        embeddings = f.readlines()

    user_embeddings = {}
    business_embeddings = {}
    other_embeddings = {}

    for embedding in embeddings:
        elements = embedding.strip().split(' ')
        node_id = elements[0]
        values = ' '.join(elements[1:])  # 提取节点嵌入

        if node_id.startswith('u'):
            user_embeddings[node_id] = values
        elif node_id.startswith('b'):
            business_embeddings[node_id] = values
        else:
            other_embeddings[node_id] = values

    with open(output_file_users, 'w') as f:
        for node_id, values in sorted(user_embeddings.items(), key=lambda x: int(x[0][1:])):
            f.write(f"{node_id[1:]} {values}\n")

    with open(output_file_businesses, 'w') as f:
        for node_id, values in sorted(business_embeddings.items(), key=lambda x: int(x[0][1:])):
            f.write(f"{node_id[1:]} {values}\n")

    # with open('other_embeddings.txt', 'w') as f:
    #     for node_id, values in sorted(other_embeddings.items()):
    #         f.write(f"{node_id} {values}\n")


extract_embeddings('D:/.embedding', 'D:/.txt', 'D:/.txt')
