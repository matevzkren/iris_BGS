import numpy as np
import pickle
from collections import defaultdict

# BGS algorithm: http://ieeexplore.ieee.org/document/4483672/


def calc_hamming_distance(t1, m1, t2, m2):
    """Calculate hamming distance between two normalized iris templates.
    Also take into account noise masking, so that only significant bits
    are used in calculating the Hamming distance.

    The use of a single filter is presumed. Shifts are taken into
    consideration to minimize the Hamming distance and to outweigh iris
    rotations during the Acquisition step of the recognition pipeline.

    :param t1: np.ndarray
        Iris template.
    :param m1: np.ndarray
        Iris mask.
    :param t2: np.ndarray
        Iris template.
    :param m2: np.ndarray
        Iris mask.
    :return:
        hd: float
            Normalized hamming distance.
    """

    assert t1.size == t2.size and m1.size == m2.size

    hd = None
    for shift in np.arange(-8, 9):
        shift *= 2
        # shift the first iris cmp item
        np_t_template1 = np.roll(t1, shift, axis=1)
        np_t_mask1 = np.roll(m1, shift, axis=1)
        # calculate the combined mask (insignificant bits)
        combined_mask = np.logical_or(np_t_mask1, m2)
        # how many significant bits remain?
        num_masked_bits = np.sum(combined_mask)
        num_bits_to_compare = np_t_template1.size - num_masked_bits

        # where are the templates different?
        diff = np.logical_xor(np_t_template1, t2)
        # intersect the result with significant bits (non-masked)
        diff = np.logical_and(diff, np.logical_not(combined_mask))
        num_different_bits = np.sum(diff)

        temp = num_different_bits * 1.0 / num_bits_to_compare
        hd = temp if temp < hd or hd is None else hd

    return hd


def compute_beacon_values(t, num_inner_rows=7):
    """Compute beacon values for a given iris template. These values
    help to uniquely position the iris in each beacon space. This part
    of the algorithm may be called locality sensitive hashing.

    :param t: np.ndarray
        Iris template.
    :param num_inner_rows: int
        In each block, beacon value is obtained by concatenating this
        many least significant bits of both bytes. The selected bits
        correspond to the iris region near the pupil. Altogether, there
        are 2^m unique beacon values possible in each beacon space,
        where m = 2 * num_inner_rows.
    :return:
        beacon_values: list of str
    """

    # prepare indices for byte interleaving
    s1, s2 = range(t.shape[1]/2), range(t.shape[1]/2, t.shape[1])
    indices = [j for i in zip(s1, s2) for j in i]

    # permute the iris code and rotate bit columns (each ring by a
    # different angle) to separate the neighborhood bits
    prep_t = np.column_stack((t[:, i] for i in indices))
    for i in range(1, num_inner_rows):
        prep_t[t.shape[0]-i] = np.roll(prep_t[t.shape[0]-i], -i)

    beacon_values = []
    for i in range(0, t.shape[1], 2):
        temp = prep_t[t.shape[0]-num_inner_rows:, i:i+2]
        temp = ''.join([str(bit) for smh in temp for bit in smh])
        beacon_values.append(temp)

    return beacon_values


def query_iris_code(t, m, iris_templates, beacon_spaces, c=3,
                    hamming_threshold=0.35):
    """Given an iris template and its masking matrix, find and return
    its nearest neighbor from the database.

    :param t: np.ndarray
        Iris template.
    :param m: np.ndarray
        Iris mask.
    :param iris_templates: dict, key=name, value=(t,m)
        Iris templates database.
    :param beacon_spaces: list of defaultdict(list)
        Populated beacon spaces structure.
    :param c: int
        Number of collisions required to test a pair of irises.
    :param hamming_threshold: float
        Similarity threshold to predict a match of irises.
    :return:
        cmp_name: str
            Identifier of the nearest neighbor iris.
    """

    b_v = compute_beacon_values(t)
    shifts = [-3, -2, -1, 0, 1, 2, 3]
    counter_through_shifts = {k: defaultdict(lambda: 0) for k in shifts}

    for i in range(len(b_v)):
        for k in shifts:
            temp_ind = i+k
            invert = False
            if temp_ind < 0:
                temp_ind += len(b_v)
                invert = True
            elif temp_ind >= len(b_v):
                invert = True
                temp_ind -= len(b_v)

            beacon = b_v[temp_ind]
            if invert:
                beacon = beacon[len(beacon)/2:] + beacon[:len(beacon)/2]

            names_to_check = beacon_spaces[i][beacon]
            for cmp_name in names_to_check:
                counter_through_shifts[k][cmp_name] += 1
                if counter_through_shifts[k][cmp_name] == c:
                    cmp_t, cmp_m = iris_templates[cmp_name]
                    distance = calc_hamming_distance(t, m, cmp_t, cmp_m)
                    if distance <= hamming_threshold:
                        return cmp_name

if __name__ == '__main__':

    with open('iris_templates.dat', 'rb') as f:
        iris_templates = pickle.load(f)

    # calculate the number of beacon spaces (2 columns per block)
    n = list(iris_templates.values())[0][0].shape[1]/2

    # populate beacon spaces
    beacon_spaces = [defaultdict(list) for i in range(n)]
    for name, (t, m) in list(iris_templates.items()):
        b_v = compute_beacon_values(t)
        for i, beacon in enumerate(b_v):
            beacon_spaces[i][beacon].append(name)

    t, m = iris_templates['001_1_1.jpg']
    output_name = query_iris_code(t, m, iris_templates, beacon_spaces)
