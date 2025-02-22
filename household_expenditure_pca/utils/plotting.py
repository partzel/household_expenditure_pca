import matplotlib.pyplot as plt
import seaborn as sns

def distribution_plot(df,column):
    plt.figure()
    plot=sns.histplot(df[column], kde=True)
    plt.title(f'Boxplot of {column}')
    return plot
def outlier_check(df,column):
    plt.figure()
    plot=sns.boxplot(data=df[column])
    plt.title(f'Boxplot of {column}')
    return plot

def correlation_analytics(df):
    plot=sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    return plot
