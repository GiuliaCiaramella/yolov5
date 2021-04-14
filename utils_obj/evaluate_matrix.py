# b is overfit
a = np.matrix([[0.22, 0.01, 0.05, 0.01, 0.35], [0.02, 0.42, 0, 0.02, 0.41], [0.02, 0.01, 0.53, 0, 0.21],
               [0.01, 0, 0.02, 0.8, 0.02], [0.73, 0.56, 0.41, 0.17, 0]])

b = np.matrix([[0.3, 0.01, 0.01, 0, 0.34], [0.01, 0.4, 0.01, 0, 0.44], [0.01, 0.01, 0.49, 0, 0.2],
               [0.01, 0, 0.01, 0.84, 0.02], [0.66, 0.59, 0.47, 0.15, 0]])

import seaborn as sns
import matplotlib.pyplot as plt


# ideal matrix
np.random.seed(15)
diagonal = np.random.randint(low = 85, high=99, size=len(a))
ideal = np.random.randint(low=1,high=3, size = a.shape)
np.fill_diagonal(ideal, diagonal)
ideal[-1, -1] = 0
ideal = ideal /100
# p = sns.heatmap(ideal, annot=True, cmap="YlGnBu")
# plt.show()
ideal = np.reshape(ideal, (ideal.size, -1))
best = {'old':0, 'new':0}
# compare_matrix(a, b)
def distance_ideal(ideal,mat):
    a = np.reshape(mat, (mat.size, -1))
    da = 1-distance.cosine(a, ideal)
    return da

def prepare_series(mat):
    wrong_det = mat[-1:]  # background False positive
    missed_det = mat[:, -1]  # background false negative
    mat_small = np.delete(mat, -1, 0)
    mat_small = np.delete(mat_small, -1, 1)
    # prepare df
    data = {'diag': np.average(np.diagonal(mat_small)), 'bgFP': np.average(wrong_det),
            'bgFN': np.average(missed_det),
            # 'summ_off_diag': (np.sum(mat_small) - np.sum(mat_small.diagonal()))/(mat_small.size-np.sqrt(mat_small.size)),
            'b_cells': len(np.where(mat_small == 0)[0]),
            'd_from_ideal':distance_ideal(ideal, mat)}
    serie = pd.Series(data=data, index=list(data.keys()))
    return (serie)


def compare_metrics(sa, sb, metric, results, optimal='min'):
    if metric == 'diag':
        optimal = 'max'

    best_model = 'new'
    if optimal == 'max':
        if sa.loc[metric] > sb.loc[metric]:
            results['diff'][metric] = sa.loc[metric] - sb.loc[metric]
            best_model = 'old'
        elif sa.loc[metric] < sb.loc[metric]:
            results['diff'][metric] = sb.loc[metric] - sa.loc[metric]
        else:
            pass

    elif optimal == 'min':
        if sa.loc[metric] > sb.loc[metric]:
            results['diff'][metric] = sa.loc[metric] - sb.loc[metric]
        elif sa.loc[metric] < sb.loc[metric]:
            results['diff'][metric]= sb.loc[metric] - sa.loc[metric]
            best_model = 'old'
        else:
            pass


    if math.isnan(results['diff'][metric]):
        point = 0
    else:
        point = results['diff'][metric]
        results['best'][metric] = best_model

    best[best_model] += point

    return results


def compare_matrix(old, new):
    sa = prepare_series(old)
    sb = prepare_series(new)

    results = pd.DataFrame(columns=['old', 'new', 'diff', 'best'], index=sa.index)
    results['old'] = sa
    results['new'] = sb

    for metric in sa.index:
        results = compare_metrics(sa, sb, metric, results)

    return(results)


def comment_results(old, new):
    results = compare_matrix(old, new)

    print('MORE IDEAL: ')
    best_model = results['best']['d_from_ideal']
    if type(best_model)== str:
        print('\t', best_model)
    else:
        print('\tMatrices are similar.')

    # look at overfitting signs
    print('OVERFITTING: ')
    best_model = results['best']['b_cells']
    if type(best_model)== str:
        print('\t', best_model, ' model seems to overfit less. ')
    else:
        if results['new']['b_cells'] == 0:
            print('\tModels do not show overfitting')
        elif results['new']['b_cells'] != 0:
            print('\tModels behave the same with overfitting')

    # looking at wrong detection and missed detection
    print('MISSED DETECTION: ')
    best_model = results['best']['bgFN']
    if type(best_model)== str:
        print('\t', best_model, ' model has less missed detection.')
    else:
        print('\tModels have same miss detection.')

    print('WRONG DETECTION: ')
    best_model = results['best']['bgFP']
    if type(best_model)== str:
        print('\t', best_model, ' model has less background wrong detection.')
    else:
        print('\tModels have same background wrong detection.')

    # print(best)
    max_best = max(best)
    if best[max_best] != 0:
        print('********************************************************')
        print('---------------BEST OVERALL MODEL: %s ---------------' %max_best)
        print('********************************************************')
    else:
        print('*******************************************************')
        print('----BEST OVERALL MODEL: models are perfectly equal ----')
        print('*******************************************************')

    return(results)


res = comment_results(a,b)
print(res)
print('**************************')
print('SCORES: ', best)
print('**************************')