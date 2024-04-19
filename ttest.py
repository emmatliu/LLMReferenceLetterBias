import scipy.stats as stats
import pandas as pd

inference_map = {
    "per_pos":"Positivity",
    "per_for":"Formality",
    "per_ac":"Agency"
}

def compute_ttest(df_m, df_f, hallucination=False):
    results = []
    for inference in ["per_pos", "per_for", "per_ac"]:
        if not hallucination:
            per_f = df_f[inference].tolist()    
            per_m = df_m[inference].tolist()

            res = stats.ttest_ind(a=per_m, b=per_f, equal_var=True, alternative='greater')
            statistic, pvalue = res[0], res[1]
            results.append([inference_map[inference], statistic, pvalue])

        if hallucination:
            hal_f = df_f[inference].tolist()   
            ori_f = df_f['{}_1'.format(inference)].tolist()
            hal_m = df_m[inference].tolist()
            ori_m = df_m['{}_1'.format(inference)].tolist()

            res1 = stats.ttest_ind(a=hal_m, b=ori_m, equal_var=True, alternative='greater')
            statistic1, pvalue1 = res1[0], res1[1]
            results.append(["Male", inference, statistic1, pvalue1])

            res2 = stats.ttest_ind(a=ori_f, b=hal_f, equal_var=True, alternative='greater')
            statistic2, pvalue2 = res2[0], res2[1]
            results.append(["Female", inference, statistic2, pvalue2])
    
    if not hallucination:
        results_df = pd.DataFrame(results, columns=["Inference", "Statistic", "P-Value"])
    else:
        results_df = pd.DataFrame(results, columns=["Gender", "Inference", "Statistic", "P-Value"])

    return results_df