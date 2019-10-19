import numpy as np

def score_function(replicate):
    result = np.sum(replicate, axis=1)
    result = result.reshape((replicate.shape[0], 1))
    return result


def selection(candidates, scores, M):
    N = candidates.shape[0]
    probs = scores/np.sum(scores)
    idx = np.random.choice(np.arange(0, N), size=M, replace=False, p=probs)
    return candidates[idx,:]

def parent_selection(candidates, scores, M):
    M = int(M)
    N = candidates.shape[0]
    probs = scores/np.sum(scores)
    idx = np.random.choice(np.arange(0, N), size=M, replace=False, p=probs)
    return candidates[idx,:]

def mating(candidates):
    # TODO : parameterize proportion to crossover
    N = candidates.shape[0]
    C = candidates.shape[1]
    midpoint = int(C/2)
    partition_a = candidates[0:int(N/2),:]
    partition_b = candidates[int(N/2):,:]
    np.random.shuffle(partition_a)
    np.random.shuffle(partition_b)
    result_top = np.concatenate((partition_a[:,:midpoint], partition_b[:,midpoint:]), axis=1)
    result_bot = np.concatenate((partition_b[:,:midpoint], partition_a[:,midpoint:]), axis=1)
    result = np.concatenate((result_top, result_bot), axis=0)
    return result

def mutate(candidates):
    N = candidates.shape[0]
    C = candidates.shape[1]
    indicators = np.random.binomial(1, 1/C, C*N).reshape(N,C)
    result = np.logical_xor(candidates,indicators)
    return result.astype(int)

def fitness(scores):
    return np.mean(scores)

def elite(candidates, scores, pop_size, elitism_size):
    tmp = np.concatenate((candidates, scores), axis=1)
    idx = np.argsort(-tmp[:, -1])
    sorted_candidates = tmp[idx]
    elite_candidates = sorted_candidates[:elitism_size,:-1]
    remaining = sorted_candidates[elitism_size:,:-1]
    weights = sorted_candidates[elitism_size:,-1]
    selected_children = selection(remaining, weights, pop_size - elitism_size)
    return (selected_children, elite_candidates)



def optimize(n_iter, candidates, scores, pop_size, elitism_size):
    remaining, elite_candidates = elite(candidates, scores, pop_size, elitism_size)

    for iter in np.arange(n_iter):
        weights = score_function(remaining)
        parents = parent_selection(remaining, weights, pop_size/2)
        children = mutate(mating(parents))
        # plot avg, plot best score per gen
        children_score = score_function(children)
        selected_children = selection(children, children_score, pop_size-elitism_size)
        next_gen = np.concatenate((elite_candidates, selected_children), axis=0)
        remaining, elite_candidates = elite(next_gen, score_function(next_gen), pop_size, elitism_size)

    return remaining, elite_candidates









if __name__ == '__main__':
    K = 10
    candidates = np.random.binomial(1, 1/10, K*K).reshape(K,K)
    scores = score_function(candidates)
    pop_size = 8
    parents = selection(candidates, scores, pop_size)

    optimize(10, candidates, scores, 4, 4)

