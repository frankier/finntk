def avg_vec_from_vecs(vecs):
    word_count = 0
    vec_sum = 0
    for vec in vecs:
        vec_sum += vec
        word_count += 1
    if word_count == 0:
        return None
    return vec_sum / word_count


def avg_vec(space, stream):

    def gen():
        for word in stream:
            try:
                yield space.get_vector(word)
            except KeyError:
                continue

    return avg_vec_from_vecs(gen())


def cosine_sim(u, v):
    from scipy.spatial.distance import cosine

    return 1 - cosine(u, v)
