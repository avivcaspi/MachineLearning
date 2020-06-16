from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from operator import mul,add
from graphviz import Source
features = ['Yearly_IncomeK', 'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score', 'Avg_Satisfaction_with_previous_vote', 'Avg_monthly_income_all_years', 'Overall_happiness_score', 'Avg_size_per_room', 'Weighted_education_rank', 'Most_Important_Issue_Education', 'Most_Important_Issue_Environment', 'Most_Important_Issue_Financial', 'Most_Important_Issue_Foreign_Affairs', 'Most_Important_Issue_Healthcare', 'Most_Important_Issue_Military', 'Most_Important_Issue_Other', 'Most_Important_Issue_Social']

classes = ['Blues', 'Browns', 'Greens', 'Greys', 'Khakis', 'Oranges', 'Pinks', 'Purples', 'Reds', 'Turquoises', 'Violets', 'Whites', 'Yellows']


def export_graph_tree(decision_tree, class_names, file_name):
    graph = Source(export_graphviz(decision_tree, out_file=None, feature_names=features, class_names=class_names,
                        filled=True, max_depth=3))
    png_bytes = graph.pipe(format='png')
    with open(file_name + '.png', 'wb') as f:
        f.write(png_bytes)


def check_factors(clf, X_test):
    factors = {'Political_interest_Total_Score': (0.5, add), 'Overall_happiness_score': (2.3, mul), 'Avg_size_per_room': (0.5, mul), 'Weighted_education_rank': (2, add)}
    for f, (w, op) in factors.items():
        X_test_temp = X_test.copy()
        X_test_temp.loc[:, f] = op(X_test_temp.loc[:, f], w)
        y_pred = clf.predict(X_test_temp)
        division_of_voters = {classes[party]: list(y_pred).count(party) for party in set(y_pred)}
        n_voters = len(X_test_temp)
        division_of_voters.update((key, round(value * 100 / n_voters, 3)) for key, value in division_of_voters.items())
        explode = (0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        plt.pie(division_of_voters.values(), explode=explode, labels=division_of_voters.keys(),autopct='%1.1f%%', shadow=True, startangle=0)
        plt.title('Division of voters, factor: ' + f)
        plt.axis('equal')
        plt.show()


"""
data_original = XY_test
f1 = 'Vote'
f2 = 'Weighted_education_rank'
df = data_original.loc[:, [f1, f2]]
plt.figure(figsize=(8, 8))
plt.scatter(x=df[f1], y=df[f2], c=df['Vote'])

plt.xlabel(f1)
plt.ylabel(f2)
plt.colorbar()
plt.show()

data_original = insert_label_to_data(x,y_test)
f1 = 'Vote'
f2 = 'Weighted_education_rank'
df = data_original.loc[:, [f1, f2]]
plt.figure(figsize=(8, 8))
plt.scatter(x=df[f1], y=df[f2], c=df['Vote'])

plt.xlabel(f1)
plt.ylabel(f2)
plt.colorbar()
plt.show()"""
