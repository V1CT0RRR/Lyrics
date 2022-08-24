import numpy as np

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if(denom==0.0):
        return 0.0
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

def get_distance_cosine(sent_embedding, candidate_embedding, elmo_layers_weight = [0.0, 1.0, 0.0]):
    result = 0.0
    for i in range(0, 3):
        a, b = sent_embedding[i], candidate_embedding[i]
        result += cos_sim(a, b) * elmo_layers_weight[i]
    return result