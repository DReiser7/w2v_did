from scipy import stats
import numpy as np


def calc_and_print_ttest(experiment1, experiment2):
    significance_level = 0.05

    t_statistic, p_value = stats.ttest_rel(experiment1, experiment2)

    print("t-statistic: " + str(round(t_statistic, 4)))
    print("p-value: " + str(round(p_value, 4)))
    print("averages are signifcantly different: " + str(p_value <= significance_level))

if __name__ == "__main__": #https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/one-sample-t-test/

    run_1000 = np.array([1.072	,1.053	,1.105])
    run_2000 = np.array([1.004	,1.034	,0.9756])
    run_3000 = np.array([1.01	,0.9988	,1.009])
    run_4000 = np.array([1.01	,1.009	,0.9259])

    print("4000 - 1000")
    calc_and_print_ttest(run_4000, run_1000)

    print("4000 - 2000")
    calc_and_print_ttest(run_4000, run_2000)

    print("4000 - 3000")
    calc_and_print_ttest(run_4000, run_3000)
