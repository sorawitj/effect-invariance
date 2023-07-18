import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main_effec_str(x):
    if x == 'lin-lin':
        return r"Linear | $\tau$: Linear"
    elif x == 'nonlin-lin':
        return r"Nonlinear | $\tau$: Linear"
    elif x == 'nonlin-nonlin':
        return r"Nonlinear | $\tau$: Nonlinear"
    else:
        raise Exception('main_effect invalid string')


df = pd.read_csv("results/wald_dr_compare.csv")
df['Main Effect'] = df['model'].map(lambda x: main_effec_str(x))

sns.set(font_scale=1.1, style='white', palette=sns.set_palette("tab10"))

g = sns.relplot(
    data=df, x="sample size", y="Rejection Rate", row='Main Effect',
    col="test", hue="set", kind="line", marker='o', markersize=6,
    height=2.5, aspect=1.8, alpha=.6
)
for ax in g.axes.flatten():
    ax.axhline(0.05, ls='--', color='black', label='5% level', linewidth=0.85, alpha=0.7)
plt.legend(bbox_to_anchor=(1.35, 2.5))

plt.xscale('log', base=2)
g.set(xticks=[1000, 2000, 4000, 8000], xticklabels=[1000, 2000, 4000, 8000])
g.set_titles(r'Test: {col_name} | Main effect: {row_name}')
g.set_xlabels("Sample Size")
plt.savefig('results/wald_dr_compare.pdf', bbox_inches="tight")
plt.close()