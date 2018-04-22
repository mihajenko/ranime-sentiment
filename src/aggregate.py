import os

import pandas
import seaborn as sns

from visualize import plot
from visualize import PROJECT_DIR
from visualize import save_visualization

series_label_color = [
    ('3boshi', 'bubble gum pink'),
    ('deathmarch', 'midnight'),
    ('overlord', 'moss green'),
    ('popepic', 'red'),
    ('violet', 'blue'),
    ('yorimoi', 'bright orange'),
    ('yurucamp', 'grey/blue')
]


def aggregated_plot():
    avg_sentiments, colors = [], []
    for label, color in series_label_color:
        print(f'processing "{label}"')
        # prepare series dataframe
        comments_path = os.path.join(PROJECT_DIR, f'comments_{label}')
        avg_sent = plot(comments_path, color=sns.xkcd_rgb[color])
        # add episode label
        avg_sent['title'] = label
        avg_sentiments.append(avg_sent)
        colors.append(color)

    # concatenate series dataframes
    df = pandas.concat(avg_sentiments)

    # plot the linear regression lines for each series
    ml = sns.lmplot(x='episode', y='sentiment', hue='title',
                    data=df.reset_index(), truncate=True, size=10,
                    legend_out=True, palette=(sns.xkcd_palette(colors)))
    # set figure title
    ax = ml.fig.gca()
    ax.set_title('Episodic averages of user sentiment extracted from /r/anime '
                 'threads & series sentiment trend lines', fontsize=12, pad=0)
    # fix ticks
    ax.set_xticks(range(1, 14, 1))


if __name__ == '__main__':
    # calculate data
    aggregated_plot()
    # save to file
    save_visualization('aggregated.png')
