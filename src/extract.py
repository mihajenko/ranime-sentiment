import json
import os
import re

import praw

# episode number pattern
_num_p = re.compile(r'episode_([0-9]+)_', flags=re.I | re.U)


# import Reddit credentials (app & user)
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESOURCES_DIR = os.path.join(PROJECT_DIR, 'resources')

config_fn = os.path.join(RESOURCES_DIR, 'config.json')
if not os.path.exists(config_fn):
    raise SystemExit('"config.json" file not found in "resources" folder')
try:  # open the config file
    with open(config_fn) as fp:
        config = json.load(fp)
except (IOError, EOFError, OSError):
    raise SystemExit('error opening "config.json" file')

# load the article IDs
article_urls = config.pop('articleURLs', None)
if not article_urls:
    raise SystemExit('no article URLs under config key "articleList"')
credentials = config


# configure the reddit instance
reddit = praw.Reddit(**credentials)
# define directory for saving
SAVE_DIR = os.path.join(PROJECT_DIR, 'comments')
# build a list of saved comments to ignore (by ID)
saved_comments = set(os.listdir(SAVE_DIR))


# get the comment forest
for url in article_urls:
    # get article object
    article = reddit.submission(url=url)
    # get episode number
    ep_num = int(_num_p.search(url).group(1))

    # load comments
    comments = article.comments
    comments.replace_more(limit=0)

    # get first-level comments only
    for comment in comments[:]:
        # check if valid comment
        author = comment.author
        if not author:
            continue
        author_name = author.name

        # check if comment already crawled
        fn = f'{author_name}__{comment.subreddit_id}__{comment.id}'
        if fn in saved_comments:
            print(f'ignore "{fn}"')
            continue

        d = {
            'author': author_name,
            'body': comment.body,
            'score': comment.score,
            'flair': comment.author_flair_text,
            'episodeNumber': ep_num
        }
        # save the comment
        with open(os.path.join(SAVE_DIR, fn), 'w') as fp:
            json.dump(d, fp)
        print(f'saved "{fn}"')
