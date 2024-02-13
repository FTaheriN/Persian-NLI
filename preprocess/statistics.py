import numpy as np
import matplotlib.pyplot as plt


def stats(df, plot=False):
    print('\033[1m'+ "Number of rows with NULL values: ", sum(df.isnull().any(axis=1)))
    print("Number of Duplicates: ", df.duplicated().sum())
    df['p_lengths'] = df.iloc[:,0].apply(lambda t: len(t.split()))
    df['h_lengths'] = df.iloc[:,1].apply(lambda t: len(t.split()))

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist(df['p_lengths'], bins=np.arange(min(df['p_lengths']), max(df['p_lengths']) + 1, 1))
        ax1.set_title('Premise Length')
        ax2.hist(df['h_lengths'], bins=np.arange(min(df['h_lengths']), max(df['h_lengths']) + 1, 1))
        ax2.set_title('Hypothesis Length');

    print('\033[1m'+"     Premise      ")
    print("Min: {}, Max: {}".format(df['p_lengths'].min(), df['p_lengths'].max()))
    print()
    print('\033[1m'+"     Hypothesis      ")
    print("Min: {}, Max: {}".format(df['h_lengths'].min(), df['h_lengths'].max()))

    dfg = df.groupby(['label'])['premise'].count()

    plt.figure()
    ax = dfg.plot(kind='bar', title='Label Frequency', ylabel='count',
            xlabel='labels', figsize=(4, 3));
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


def data_gl_than(data, col, less_than=100.0, greater_than=0.0):
    data_length = data[col].values
    data_glt = sum([1 for length in data_length if greater_than < length <= less_than])
    data_glt_rate = (data_glt / len(data_length)) * 100
    print(f'Premise with word length of greater than {greater_than} and less than {less_than} includes {data_glt_rate:.2f}% of the whole!')



def plot_statistics(train_df, valid_df, test_df):
    stats(train_df, True)
    stats(valid_df, False)
    stats(test_df,  False)

    print("Train data: ")
    data_gl_than(train_df, 'p_lengths', 80)
    print("Validation data: ")
    data_gl_than(valid_df, 'p_lengths', 80)
    print("Test data: ")
    data_gl_than(test_df, 'p_lengths', 80)

    return 


