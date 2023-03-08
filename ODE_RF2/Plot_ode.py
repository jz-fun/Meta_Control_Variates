import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set(font_scale=1.)

# Extract Saved Outputs
my_ODE_DF = pd.read_pickle("ODE_RF2/ODE_DF2.pkl")
df2 = my_ODE_DF[my_ODE_DF["sample_size"] != 'm=2']
df2['method'] = df2['method'].replace('MCV', 'Meta-CVs')
df2['method'] = df2['method'].replace('NCV', 'Neural-CVs')
my_pal = {"Meta-CVs": "blue", "Neural-CVs": "red", "MC":"black"}
sns.set_style(style='white')
g=sns.catplot(x="sample_size",
              y="est_abserr",
              hue="method",
              data=df2,
              kind="box",
              height=3,
              aspect=5/3,
              palette=my_pal,
              medianprops={'color':'white'},
              showmeans=True,
              meanprops={"marker": "+",
                         "linestyle": "--",
                         "color": "grey",
                         "markeredgecolor": "grey",
                         "markersize": "10"})


(g.set_axis_labels("Sample Size", "Absolute Error")
  .set_titles("{col_name}")
  .despine(left=True)
 )
g.set(ylim=(None, 7))
g._legend.remove()
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.0, 1), loc='best', borderaxespad=0.)
# plt.show()
plt.savefig('Bound_value_ode.pdf')