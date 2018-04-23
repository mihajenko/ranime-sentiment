# monkey-patch FacetGrid

import os

import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from visualize import plot
from visualize import PROJECT_DIR
from visualize import save_visualization


class MyFacetGrid(sns.axisgrid.FacetGrid):
    def __init__(self, *args, **kwargs):
        self.reg_plot = None
        super(MyFacetGrid, self).__init__(*args, **kwargs)

    def _facet_plot(self, func, ax, plot_args, plot_kwargs):
        # Draw the plot
        x, y = plot_args
        self.reg_plot = func(x, y, **plot_kwargs)

        # Sort out the supporting information
        self._update_legend_data(ax)
        self._clean_axis(ax)

    def get_reg_plot(self):
        return self.reg_plot


def my_lmplot(x, y, data, hue=None, col=None, row=None, palette=None,
              col_wrap=None, size=5, aspect=1, markers="o", sharex=True,
              sharey=True, hue_order=None, col_order=None, row_order=None,
              legend=True, legend_out=True, x_estimator=None, x_bins=None,
              x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
              units=None, order=1, logistic=False, lowess=False, robust=False,
              logx=False, x_partial=None, y_partial=None, truncate=False,
              x_jitter=None, y_jitter=None, scatter_kws=None, line_kws=None):

    # Reduce the dataframe to only needed columns
    need_cols = [x, y, hue, col, row, units, x_partial, y_partial]
    cols = np.unique([a for a in need_cols if a is not None]).tolist()
    data = data[cols]

    # Initialize the grid
    facets = MyFacetGrid(data, row, col, hue, palette=palette,
                         row_order=row_order, col_order=col_order,
                         hue_order=hue_order, size=size, aspect=aspect,
                         col_wrap=col_wrap, sharex=sharex, sharey=sharey,
                         legend_out=legend_out)

    # Add the markers here as FacetGrid has figured out how many levels of the
    # hue variable are needed and we don't want to duplicate that process
    if facets.hue_names is None:
        n_markers = 1
    else:
        n_markers = len(facets.hue_names)
    if not isinstance(markers, list):
        markers = [markers] * n_markers
    if len(markers) != n_markers:
        raise ValueError(("markers must be a singeton or a list of markers "
                          "for each level of the hue variable"))
    facets.hue_kws = {"marker": markers}

    # Hack to set the x limits properly, which needs to happen here
    # because the extent of the regression estimate is determined
    # by the limits of the plot
    if sharex:
        for ax in facets.axes.flat:
            ax.scatter(data[x], np.ones(len(data)) * data[y].mean()).remove()

    # Draw the regression plot on each facet
    regplot_kws = dict(
        x_estimator=x_estimator, x_bins=x_bins, x_ci=x_ci,
        scatter=scatter, fit_reg=fit_reg, ci=ci, n_boot=n_boot, units=units,
        order=order, logistic=logistic, lowess=lowess, robust=robust,
        logx=logx, x_partial=x_partial, y_partial=y_partial, truncate=truncate,
        x_jitter=x_jitter, y_jitter=y_jitter,
        scatter_kws=scatter_kws, line_kws=line_kws,
        )
    facets.map_dataframe(sns.regplot, x, y, **regplot_kws)

    # Add a legend
    if legend and (hue is not None) and (hue not in [col, row]):
        facets.add_legend()
    return facets


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

    legend_patches = []
    avg_sentiments, colors = [], []
    for label, color in series_label_color:
        print(f'processing "{label}"')

        # get color code
        color = sns.xkcd_rgb[color]

        # prepare series dataframe
        comments_path = os.path.join(PROJECT_DIR, f'comments_{label}')
        avg_sent = plot(comments_path, color=color)
        # add episode label
        avg_sent['title'] = label
        # plot
        ax = sns.regplot(x='episode', y='sentiment',
                         data=avg_sent.reset_index(),
                         fit_reg=True, truncate=True, color=color)

        line = ax.get_lines()[-1]
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        x_idx = list(range(int(xdata[0]), int(xdata[-1]) + 1))
        # interpolate data at "episode number" values
        y = np.interp(x_idx, xdata, ydata)
        fy = avg_sent['sentiment'].as_matrix()
        print(y)
        print(np.corrcoef(y, y=fy)[1, 0])
        legend_patches.append(mpatches.Patch(color=color, label=label))

    ax = plt.gca()
    ax.set_xlabel('Episode number')
    ax.set_ylabel('Sentiment score')
    plt.legend(handles=legend_patches,
               bbox_to_anchor=(1, 0.5), loc="center right", fontsize=10,
               bbox_transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.45)

    # ax.annotate(f'{sent_val:.2f}', (index, row['sentiment']))
    #
    #
    #     avg_sentiments.append(avg_sent)
    #     colors.append(color)
    #
    # # concatenate series dataframes
    # df = pandas.concat(avg_sentiments)
    # print('mean', df['sentiment'].mean())
    #
    # # plot the linear regression lines for each series
    # ml = my_lmplot(x='episode', y='sentiment', hue='title',
    #                data=df.reset_index(), truncate=True, size=10,
    #                fit_reg=True, logistic=False, legend_out=True,
    #                palette=(sns.xkcd_palette(colors)))
    #
    # # TODO: calculate r (correlation coefficients) for regression lines
    # for line in ml.get_reg_plot().get_lines():
    #     xdata = line.get_xdata()
    #     ydata = line.get_ydata()
    #     x_idx = list(range(int(xdata[0]), int(xdata[-1]) + 1))
    #     # interpolate data at "episode number" values
    #     x_lin = np.interp(x_idx, xdata, ydata)
    #     print(x_lin)

    # # set figure title
    # ax = ml.fig.gca()
    # ax.set_title('Episodic averages of user sentiment extracted from /r/anime '
    #              'threads & series sentiment trend lines', fontsize=12, pad=0)
    # # fix ticks
    # ax.set_xticks(range(1, 14, 1))

    # for index, row in df.iterrows():
    #     sent_val = row['sentiment']
    #     ax.annotate(f'{sent_val:.2f}', (index, row['sentiment']))


if __name__ == '__main__':
    # calculate data
    aggregated_plot()
    # save to file
    save_visualization('aggregated.png')
