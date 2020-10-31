import copy
import csv
import math
import sys
import matplotlib.pyplot

CLASSES_CNT = 3


def uniform(u):
    return 0.5 if abs(u) < 1 else 0


def triangular(u):
    return 1 - abs(u) if abs(u) < 1. else 0


def epanechnikov(u):
    return 3 * (1. - u * u) / 4 if abs(u) < 1 else 0


def quartic(u):
    return 15 * ((1 - u * u) ** 2) / 16. if abs(u) < 1 else 0


def triweight(u):
    return 35 * ((1 - u * u) ** 3) / 32 if abs(u) < 1 else 0


def tricube(u):
    return 70 * ((1 - (abs(u) ** 3)) ** 3) / 81 if abs(u) < 1 else 0


def gaussian(u):
    return (math.e ** (-(u * u) / 2)) / math.sqrt(2 * math.pi)


def cosine(u):
    return math.pi / 4 * math.cos(math.pi * u / 2) if abs(u) < 1 else 0


def logistic(u):
    return 1 / ((math.e ** u) + 2 + (math.e ** -u))


def sigmoid(u):
    return 2 / (math.pi * ((math.e ** u) + (math.e ** -u)))


def manhattan(row1, row2):
    dist = 0.0
    for i in range(len(row1)):
        if i == 0:
            continue
        dist += abs(row1[i] - row2[i])
    return dist


def euclidean(row1, row2):
    dist = 0.0
    for i in range(len(row1)):
        if i == 0:
            continue
        dist += (row1[i] - row2[i]) ** 2
    return math.sqrt(dist)


def chebyshev(row1, row2):
    dist = abs(row1[1] - row2[1])
    for i in range(len(row1)):
        if i == 0:
            continue
        dist = max(dist, abs(row1[i] - row2[i]))
    return dist


def find_min(dataset):
    result = []
    for i in range(len(dataset[0])):
        if i == 0:
            continue
        result.append(dataset[0][i])
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if j == 0:
                continue
            result[j - 1] = min(result[j - 1], dataset[i][j])
    return result


def find_max(dataset):
    result = []
    for i in range(len(dataset[0])):
        if i == 0:
            continue
        result.append(dataset[0][i])
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if j == 0:
                continue
            result[j - 1] = max(result[j - 1], dataset[i][j])
    return result


def normalize(dataset, mins, maxs):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if j == 0:
                continue
            dataset[i][j] = (dataset[i][j] - mins[j - 1]) / (maxs[j - 1] - mins[j - 1])


def get_ans_if_0(dataset, ind_to_skip):
    count = 0
    sum = 0.
    for i in range(len(dataset)):
        if i == ind_to_skip:
            continue
        if dataset[i][1:] == dataset[ind_to_skip][1:]:
            sum += dataset[i][0]
            count += 1
    if count == 0:
        for i in range(len(dataset)):
            if i == ind_to_skip:
                continue
            count += 1
            sum += dataset[i][0]
    return sum / count


def calc_dist(type, row1, row2):
    if type == "manhattan":
        return manhattan(row1, row2)
    if type == "euclidean":
        return euclidean(row1, row2)
    if type == "chebyshev":
        return chebyshev(row1, row2)


def calc_kernel(type, u):
    if type == "uniform":
        return uniform(u)
    if type == "triangular":
        return triangular(u)
    if type == "epanechnikov":
        return epanechnikov(u)
    if type == "quartic":
        return quartic(u)
    if type == "triweight":
        return triweight(u)
    if type == "tricube":
        return tricube(u)
    if type == "gaussian":
        return gaussian(u)
    if type == "cosine":
        return cosine(u)
    if type == "logistic":
        return logistic(u)
    if type == "sigmoid":
        return sigmoid(u)


def calc_window(dataset, kernel, dist_func, ind_to_skip, h):
    if h == 0:
        return get_ans_if_0(dataset, ind_to_skip)
    else:
        sum1 = 0.0
        sum2 = 0.0
        for i in range(len(dataset)):
            if i == ind_to_skip:
                continue
            dist = calc_dist(dist_func, dataset[i], dataset[ind_to_skip])
            value = calc_kernel(kernel, dist / h)
            sum1 += dataset[i][0] * value
            sum2 += value
        if sum2 == 0.0:
            return get_ans_if_0(dataset, ind_to_skip)
        else:
            return sum1 / sum2


def calc_h(dataset, dist_func, ind_to_skip, k):
    array = []
    for i in range(len(dataset)):
        if i == ind_to_skip:
            continue
        array.append(calc_dist(dist_func, dataset[i], dataset[ind_to_skip]))
    array.sort()
    return array[k]


def calc_miss_table_variable_window(dataset, dist_func, kernel, k):
    table = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for ind_to_skip in range(len(dataset)):
        my_ans = calc_window(dataset, kernel, dist_func, ind_to_skip, calc_h(dataset, dist_func, ind_to_skip, k))
        ans = dataset[ind_to_skip][0]
        table[round(my_ans - 1)][round(ans - 1)] += 1
    return table


def calc_miss_table_fixed_window(dataset, dist_func, kernel, h):
    table = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for ind_to_skip in range(len(dataset)):
        my_ans = calc_window(dataset, kernel, dist_func, ind_to_skip, h)
        ans = dataset[ind_to_skip][0]
        table[round(my_ans - 1)][round(ans - 1)] += 1
    return table


def calc_miss_table_variable_window_one_hot(dataset, dist_func, kernel, k):
    table = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dataset1 = copy.deepcopy(dataset)
    dataset2 = copy.deepcopy(dataset)
    dataset3 = copy.deepcopy(dataset)
    for i in range(len(dataset1)):
        if dataset[i][0] == 1.0:
            dataset1[i][0] = 1.0
        else:
            dataset1[i][0] = 0.0
    for i in range(len(dataset2)):
        if dataset[i][0] == 2.0:
            dataset2[i][0] = 1.0
        else:
            dataset2[i][0] = 0.0
    for i in range(len(dataset3)):
        if dataset[i][0] == 3.0:
            dataset3[i][0] = 1.0
        else:
            dataset3[i][0] = 0.0
    for ind_to_skip in range(len(dataset)):
        anses = [0.0, 0.0, 0.0]
        anses[0] = calc_window(dataset1, kernel, dist_func, ind_to_skip, calc_h(dataset1, dist_func, ind_to_skip, k))
        anses[1] = calc_window(dataset2, kernel, dist_func, ind_to_skip, calc_h(dataset2, dist_func, ind_to_skip, k))
        anses[2] = calc_window(dataset3, kernel, dist_func, ind_to_skip, calc_h(dataset3, dist_func, ind_to_skip, k))
        actually_ans = max(range(len(anses)), key=lambda j: anses[j])  # argmax
        table[round(actually_ans)][round(dataset[ind_to_skip][0] - 1)] += 1
    return table


def calc_miss_table_fixed_window_one_hot(dataset, dist_func, kernel, h):
    table = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dataset1 = copy.deepcopy(dataset)
    dataset2 = copy.deepcopy(dataset)
    dataset3 = copy.deepcopy(dataset)
    for i in range(len(dataset1)):
        if dataset[i][0] == 1.0:
            dataset1[i][0] = 1.0
        else:
            dataset1[i][0] = 0.0
    for i in range(len(dataset2)):
        if dataset[i][0] == 2.0:
            dataset2[i][0] = 1.0
        else:
            dataset2[i][0] = 0.0
    for i in range(len(dataset3)):
        if dataset[i][0] == 3.0:
            dataset3[i][0] = 1.0
        else:
            dataset3[i][0] = 0.0
    for ind_to_skip in range(len(dataset)):
        anses = [0.0, 0.0, 0.0]
        anses[0] = calc_window(dataset1, kernel, dist_func, ind_to_skip, h)
        anses[1] = calc_window(dataset2, kernel, dist_func, ind_to_skip, h)
        anses[2] = calc_window(dataset3, kernel, dist_func, ind_to_skip, h)
        actually_ans = max(range(len(anses)), key=lambda j: anses[j])  # argmax
        table[round(actually_ans)][round(dataset[ind_to_skip][0] - 1)] += 1
    return table


def get_info_from_miss_table(table):
    sum = 0
    for i in range(CLASSES_CNT):
        for j in range(CLASSES_CNT):
            sum += table[i][j]
    return "{0} out of {1}".format(str(table[0][0] + table[1][1] + table[2][2]), str(sum))


def get_info_file_about_variable_window(filename):
    original_stdout = sys.stdout
    with open(filename, 'w') as file:
        sys.stdout = file
        for kernel in kernels:
            for dist_func in dists:
                for k in range(20):
                    print(kernel + ', ' + dist_func + ", k = " + str(k))
                    print(get_info_from_miss_table(calc_miss_table_variable_window(rows, dist_func, kernel, k)))
        sys.stdout = original_stdout


def get_info_file_about_fixed_window(filename):
    original_stdout = sys.stdout
    with open(filename, 'w') as file:
        sys.stdout = file
        for kernel in kernels:
            for dist_func in dists:
                for h in range(20):
                    print(kernel + ', ' + dist_func + ", h = " + str(h))
                    print(get_info_from_miss_table(calc_miss_table_fixed_window(rows, dist_func, kernel, h)))
        sys.stdout = original_stdout


def get_info_file_about_variable_window_one_hot(filename):
    original_stdout = sys.stdout
    with open(filename, 'w') as file:
        sys.stdout = file
        for kernel in kernels:
            for dist_func in dists:
                for k in range(20):
                    print(kernel + ', ' + dist_func + ", k = " + str(k))
                    print(get_info_from_miss_table(calc_miss_table_variable_window_one_hot(rows, dist_func, kernel, k)))
        sys.stdout = original_stdout


def get_info_file_about_fixed_window_one_hot(filename):
    original_stdout = sys.stdout
    with open(filename, 'w') as file:
        sys.stdout = file
        for kernel in kernels:
            for dist_func in dists:
                for h in range(20):
                    print(kernel + ', ' + dist_func + ", h = " + str(h))
                    print(get_info_from_miss_table(calc_miss_table_fixed_window_one_hot(rows, dist_func, kernel, h)))
        sys.stdout = original_stdout


def har(x, y):
    if x == 0 or y == 0:
        return 0
    return 2 * (x * y) / (x + y)


def get_micro_and_macro_f(table):
    cnt = 0.0
    precisions = [0.0 for i in range(CLASSES_CNT)]
    recalls = [0.0 for i in range(CLASSES_CNT)]
    f1 = [0.0 for i in range(CLASSES_CNT)]
    counts = [0.0 for i in range(CLASSES_CNT)]
    for i in range(CLASSES_CNT):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(CLASSES_CNT):
            sum1 += table[i][j]
            sum2 += table[j][i]
        counts[i] = sum1
        cnt += sum2
        precisions[i] = 0 if sum1 == 0.0 else table[i][i] / sum1
        recalls[i] = 0 if sum2 == 0.0 else table[i][i] / sum2
        f1[i] = har(precisions[i], recalls[i])
    micro = 0.0
    for i in range(CLASSES_CNT):
        micro += counts[i] * f1[i]
    micro /= cnt
    precision = 0.0
    recall = 0.0
    for i in range(CLASSES_CNT):
        precision += counts[i] * precisions[i]
        recall += counts[i] * recalls[i]
    macro = har(precision / cnt, recall / cnt)
    return [micro, macro]


def build_graphics_dummy_variable():
    for k in range(20):
        if k == 0:
            continue
        fs = get_micro_and_macro_f(calc_miss_table_variable_window(rows, variable_best_dist, variable_best_kernel, k))
        matplotlib.pyplot.plot(float(k), fs[0], 'bo')
        matplotlib.pyplot.plot(float(k), fs[1], 'go')
    matplotlib.pyplot.savefig("variable_dummy_graphics.png")


def build_graphics_one_hot_variable():
    for k in range(20):
        if k == 0:
            continue
        fs = get_micro_and_macro_f(
            calc_miss_table_variable_window_one_hot(rows, variable_best_dist_one_hot, variable_best_kernel_one_hot, k))
        matplotlib.pyplot.plot(float(k), fs[0], 'bo')
        matplotlib.pyplot.plot(float(k), fs[1], 'go')
    matplotlib.pyplot.savefig("variable_one_hot_graphics.png")


def build_graphics_dummy_fixed():
    for h in range(20):
        if h == 0:
            continue
        fs = get_micro_and_macro_f(
            calc_miss_table_fixed_window(rows, fixed_best_dist_one_hot, fixed_best_kernel_one_hot, float(h) - 0.5))
        matplotlib.pyplot.plot(float(h), fs[0], 'bo')
        matplotlib.pyplot.plot(float(h), fs[1], 'go')
        fs = get_micro_and_macro_f(calc_miss_table_fixed_window(rows, fixed_best_dist_one_hot, fixed_best_kernel_one_hot, float(h)))
        matplotlib.pyplot.plot(float(h), fs[0], 'bo')
        matplotlib.pyplot.plot(float(h), fs[1], 'go')
    matplotlib.pyplot.savefig("fixed_dummy_graphics.png")


def build_graphics_one_hot_fixed():
    for h in range(20):
        if h == 0:
            continue
        fs = get_micro_and_macro_f(
            calc_miss_table_fixed_window(rows, fixed_best_dist, fixed_best_kernel, float(h) - 0.5))
        matplotlib.pyplot.plot(float(h), fs[0], 'bo')
        matplotlib.pyplot.plot(float(h), fs[1], 'go')
        fs = get_micro_and_macro_f(calc_miss_table_fixed_window(rows, fixed_best_dist, fixed_best_kernel, float(h)))
        matplotlib.pyplot.plot(float(h), fs[0], 'bo')
        matplotlib.pyplot.plot(float(h), fs[1], 'go')
    matplotlib.pyplot.savefig("fixed_one_hot_graphics.png")


filename = 'dataset_191_wine.csv'

rows = []
with open(filename) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        rows.append(row)

rows.pop(0)
for i in range(len(rows)):
    for j in range(len(rows[i])):
        rows[i][j] = float(rows[i][j])

mins = find_min(rows)
maxs = find_max(rows)
normalize(rows, mins, maxs)
kernels = ["uniform", "triangular", "epanechnikov", "quartic", "triweight", "tricube", "gaussian", "cosine", "logistic",
           "sigmoid"]
dists = ["manhattan", "euclidean", "chebyshev"]

variable_best_kernel = "uniform"
variable_best_dist = "manhattan"

fixed_best_kernel = "triangular"
fixed_best_dist = "manhattan"

variable_best_kernel_one_hot = "gaussian"
variable_best_dist_one_hot = "euclidean"

fixed_best_kernel_one_hot = "triangular"
fixed_best_dist_one_hot = "manhattan"

