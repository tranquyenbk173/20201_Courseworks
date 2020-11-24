import os
from numpy.lib.npyio import save
from pandas import DataFrame, read_csv
from sklearn.preprocessing import OneHotEncoder


def remove_post_not_in_user_history(user_history_link, post_info_link, save_to=None):
    """
    Use:
        Remove post that title = null and is not in user_history
    Args:
        user_history_link: path to user_history.csv
        post_info_link: path to post_info.csv - new version
        save_to: path to save return eg. '/home/ducnv/new_post_info.csv'
    Return:
        data frame of new_post_info 
    """

    user_history = read_csv(
        user_history_link,
        usecols=[1, 2, 3, 4],
        encoding='utf-8',
        dtype={'userID': str, 'postId': str, 'eventId': int, 'serverTime': int}
    )

    post_info = read_csv(
        post_info_link,
        usecols=[2, 3, 4, 5, 6, 7],
        encoding='utf-8',
        dtype={'link_share': str, 'title': str, 'card_type': int, 'post_id': str, 'user_owner': str, 'board_id': float}
    )

    # remove post that title = null
    new_post_info = post_info[ [not it for it in post_info.title.isnull()] ]

    # remove post that is not in user_history
    post_in_user_history = user_history.postId.unique()
    new_post_info = new_post_info[ [post in post_in_user_history for post in new_post_info.post_id] ]

    if save_to is not None:
        new_post_info.to_csv(index=False, encoding='utf-8', path_or_buf=save_to)

    return new_post_info



def calculate_score(user_post_event, map_score):
    """
    Use: 
        Calculate total score for user-post
    Args:
        user_post_event: DataFrame: columns=['user_id', 'post_id', 'event_id']
        map_score: map event_id -> score : {2: 3, 1004: 7, ...}
    Returns:
        DataFrame: columns = ['user_id', 'post_id', 'score']
    """
    columns = list(user_post_event.columns) # user, post, event
    scores = [  map_score[event_id] for event_id in user_post_event[columns[2]] ]
    df = user_post_event[columns[0:2]]
    df['score'] = scores
    return df.groupby(columns[0:2]).sum().reset_index()


def train_test_split(user_post_score, threshold=10, split_ratio=0.7, save_dir=None):
    """
    Use:
        Split data for training and testing
    Args:
        user_post_score: DataFrame, columns = ['user_id', 'post_id', 'score'] (output of calculate_score function)
        threshold: number of posts for each particular user
        split_ratio: train test split
        save_dir: directory for save train.csv, test.csv
    Returns:
        df_train, df_test
    """

    columns = user_post_score.columns.to_list()
    user_n_posts = user_post_score[columns[0:2]].groupby(columns[0]).count()
    user_post_score = user_post_score.set_index(columns[0])

    df_train = DataFrame(columns=columns).set_index(columns[0])
    df_test = DataFrame(columns=columns).set_index(columns[0])
    for it in user_post_score.index.unique():
        n_posts = int(user_n_posts.loc[it])
        if  n_posts >= threshold:
            n_train = int(split_ratio*n_posts)
            df_train = df_train.append(user_post_score.loc[it][:n_train])
            df_test = df_test.append(user_post_score.loc[it][n_train:])
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        df_train.to_csv(os.path.join(save_dir, 'train.csv'))
        df_test.to_csv(os.path.join(save_dir, 'test.csv'))

    return df_train, df_test


def get_onehotencoder(X):
    """
    Use:
        create one hot encoder - scikit-learn
    Args:
        X: array like (n_samples, n_features) to fit encoder
    Return:
        one hot encoder object of scikit-learn
    How to use:
        X = [ ['user_id1'], ['user_id2'], ['user_id3'], ['user_id1']]
        encoder = get_onehotencoder(X)
        one_hot_user1_user2 = encoder.transform([['user_id1'], ['user_id2']]).toarray()
    """
    encoder = OneHotEncoder()
    encoder.fit(X)
    return encoder


