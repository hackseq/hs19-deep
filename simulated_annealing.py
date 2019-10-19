import pandas as pd
import numpy as np
import logging
import time
import argparse


###########
# How to run 
# python simulated_annealing.py -i hackseq_19/23/ETHAMBUTOL_X.csv -l hackseq_19/23/ETHAMBUTOL_y.csv -s 2000 -n 10 -o test
##########






def load_data(data_file, label_file):
    with open(data_file, 'r') as fp:
        counter = 0
        name_list = []
        X = []
        for line in fp:
            counter += 1
            if counter < 5:
                continue
            linelist = line.strip().split(',')
            name, x_i = linelist[0], np.array(list(map(int, linelist[1:])))
            name_list += [name]
            X += [x_i]
            print(counter, end = '\r')
    
    X = np.array(X)
    
    Y = pd.read_csv(label_file, index_col=0, header = None, names=['label'])
    return (name_list, X, Y)

# we want to minimize energy
def energy(labels, activation_scores):
    return ((labels - activation_scores)**2).sum()


def acceptance_probability(energy_old, energy_new, i, c=1):
    return np.exp(i*float((energy_old-energy_new))/float(energy_old))


def init_feature_set(tot_features, num_features=100):
    return np.random.choice(tot_features, num_features)


# Call @Sasha's function to get activation  here
def get_activation_score(feature_set, X, Y):
    return [np.random.uniform(0,1)]*len(X)


def purturb_set(tot_feature_set, feature_set):
    to_replace = to_add = np.ceil(len(feature_set)*0.05).astype(int)
    #print(to_replace, to_add)
    feature_indices_to_replace = np.random.choice(
        len(feature_set), 
        to_replace
    )
    remaining_set = np.setdiff1d(tot_feature_set, feature_set)
                         
    feature_indices_to_add = np.random.choice(len(remaining_set),to_add)
    for i in range(to_add):
            feature_set[feature_indices_to_replace[i]] = (
                remaining_set[feature_indices_to_add[i]]
            )
        
        
    return feature_set
    

def anneal(X, Y, num_steps, num_features):
    _, tot_features = X.shape
    Y = Y.label.values
    feature_set = init_feature_set(tot_features, num_features)
    activation_scores = get_activation_score(feature_set, X, Y)
    old_energy = energy(Y, activation_scores)
    
    feature_sets = []
    steps  = 0
    time_start = time.time()
    while steps <= num_steps:
        feature_set_new = purturb_set(np.arange(tot_features), feature_set)
        activation_scores = get_activation_score(feature_set_new, X, Y)
        new_energy = energy(Y, activation_scores)
        # print(old_energy, new_energy)
        if new_energy < old_energy:
            # accept 
            feature_sets += [(feature_set_new.copy(), new_energy)]
            feature_set = feature_set_new
            old_energy = new_energy
        else:
            prob = acceptance_probability(old_energy, new_energy, steps)
            rand_draw = np.random.uniform(0,1)
            # print('--',rand_draw, prob)
            if rand_draw > prob:
                # accept 
                feature_sets += [(feature_set_new.copy(), new_energy)]
                feature_set = feature_set_new
                old_energy = new_energy
                
        steps += 1
        if(steps%1000 == 0):
            print('Steps {} Elapsed time ... {}'.format(steps, time.time() - time_start))
                
    return feature_sets

def main():
    parser = argparse.ArgumentParser(
        description='Performs simple simulated annealing'
        )
    parser.add_argument(
        '-i', '--input', required=True, type=str, help='file for input data')
    parser.add_argument(
        '-l', '--label', required=True, type=str, help='file with output labels')
    parser.add_argument(
        '-s', '--steps', required=True, type=int, help='number of steps to continue')
    parser.add_argument(
        '-n', '--num_features', required=False, type=int, help='number of features to sample')
    parser.add_argument(
        '-o', '--outfile', required=True, type=str, help='output file dump the progress')

    args = parser.parse_args()

    data_file, label_file, steps, num_features, outfile = (
        args.input,
        args.label,
        args.steps,
        args.num_features,
        args.outfile
    )

    name_list, X, Y = load_data(data_file, label_file)
    Y = Y.reindex(name_list)
    feature_set = anneal(X,Y, steps, num_features)
    feature_set_df = pd.DataFrame(feature_set)
    feature_set_df.to_csv(outfile)

if __name__ == "__main__":
    main()
