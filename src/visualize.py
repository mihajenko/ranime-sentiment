import json
import os
from collections import Counter
from collections import defaultdict

import lxml
import lxml.html
from matplotlib import pyplot as plt
import mistune
import nltk.data
import pandas
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# file paths
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LOAD_DIR = os.path.join(PROJECT_DIR, 'comments')

# init plotter
sns.set()
# Markdown parser
markdown = mistune.Markdown()
# init sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
try:  # load language tokenizer
    tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
except LookupError:
    raise SystemError('Could not load any tokenizer package')


def get_sentiment(text):
    """Calculate sentiment from text."""
    global analyzer

    # tokenize text into sentences
    sentences = tokenizer.tokenize(text)

    taken_sents = []
    cumulate_sentiment = 0
    for sent in sentences:
        if len(sent) < 15:
            # ignore too short sentences
            curr_sentiment = 0.0
        else:
            # calculate sentiment
            vs = analyzer.polarity_scores(sent)
            curr_sentiment = vs['compound']
        # append
        cumulate_sentiment += curr_sentiment
        taken_sents.append(curr_sentiment)

    if not taken_sents:
        return None

    # calculate global sentiment
    return cumulate_sentiment / len(taken_sents)


def parse_markdown(md_text):
    """Parse Reddit Markdown text."""
    html = markdown(md_text)
    dom = lxml.html.fromstring(html)
    return '\n'.join([  # get text from paragraphs, must not be blockquotes
        el.text_content() for el in dom.xpath('//p[not(ancestor::blockquote)]')
        if el.text_content()
    ])


def plot(path, color='black', top_users=False, avg_sentiment=False):
    """Main plotting procedure."""

    # list of JSON comment files to open
    fns = os.listdir(path)

    # count user posts
    post_counts = Counter()
    for fn in fns:
        # split username from filename
        user = fn.split('__', 1)[0]
        post_counts[user] += 1

    # per-episode store (users can post several replies per episode thread)
    episode_store = defaultdict(list)
    # go through comment JSON files
    for fn in fns:
        # split username from filename
        user = fn.split('__', 1)[0]
        # open comment info in json
        fn = os.path.join(path, fn)
        with open(fn, 'r') as fp:
            json_data = json.load(fp)

        # convert markdown to regular text -- skip, if empty
        md_text = json_data['body']
        if not md_text:
            continue
        text = parse_markdown(md_text)
        if not text:
            continue

        # commit to episode store
        ep_num = json_data['episodeNumber']
        episode_store[(ep_num, user)].append(text)

    # collect filtered data
    data = []
    for composite, texts in episode_store.items():
        ep_num, user = composite

        # join texts and calculate sentiment on it
        text = '\n\n'.join(texts)
        sent = get_sentiment(text)
        if not sent:
            continue

        # save sentiment, episode, user
        data.append({
            'sentiment': sent,
            'episode': ep_num,
            'user': user
        })

    # construct DataFrame from the collected data
    df = pandas.DataFrame(data)

    if top_users:
        # generate user post counts
        post_counts = df['user'].value_counts(sort=True, ascending=True)
        # get 15 most active users
        users = list(post_counts.iloc[-15:].index)
        # filter DataFrame by users with most posts
        df_most_active = df.loc[df['user'].isin(users)]
        # plot the linear regression lines for each user
        sns.lmplot(x='episode', y='sentiment', hue='user', data=df_most_active,
                   truncate=True, size=10, legend_out=True)

    # calculate average sentiment (per episode) of all users
    avg_sent_df = df.groupby('episode').mean()
    if avg_sentiment:
        sns.tsplot(data=avg_sent_df['sentiment'], ci="sd", color=color)

    return avg_sent_df


def save_visualization(path, clear_plot=False):
    """Save the plot to an image file."""
    if clear_plot:
        plt.clf()
    plt.savefig(path, dpi=300)


if __name__ == '__main__':
    plot(LOAD_DIR, top_users=True, avg_sentiment=True)
    save_visualization('viz.png')
