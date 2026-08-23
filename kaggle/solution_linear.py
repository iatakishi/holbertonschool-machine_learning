"""
TED Talks Views Prediction - Linear Models
Metric: MSLE
Models: Ridge, Lasso, ElasticNet + ensemble
Key addition: StandardScaler (linear models need scaled features)
             + TF-IDF on transcript (linear models handle sparse well)
"""

import pandas as pd
import numpy as np
import ast
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv('/mnt/user-data/uploads/train.csv', index_col=0)
test  = pd.read_csv('/mnt/user-data/uploads/test.csv',  index_col=0)
trans = pd.read_csv('/mnt/user-data/uploads/transcripts.csv')

train['url'] = train['url'].str.strip()
test['url']  = test['url'].str.strip()
trans['url'] = trans['url'].str.strip()

train = train.merge(trans, on='url', how='left')
test  = test.merge(trans,  on='url', how='left')

# Fix negative views
train.loc[train['views'] < 0, 'views'] = np.nan
train['views'] = train['views'].fillna(train['views'].median())

print(f"Train: {train.shape}, Test: {test.shape}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING (same as before)
# ─────────────────────────────────────────────

def parse_ratings(ratings_str):
    try:
        ratings = ast.literal_eval(ratings_str)
        total = sum(r['count'] for r in ratings)
        positive_names = {'Inspiring','Fascinating','Jaw-dropping','Ingenious',
                          'Beautiful','Courageous','Funny','Persuasive','Informative'}
        negative_names = {'Obnoxious','Unconvincing','Confusing','Longwinded','OK'}
        pos = sum(r['count'] for r in ratings if r['name'] in positive_names)
        neg = sum(r['count'] for r in ratings if r['name'] in negative_names)
        top = max(ratings, key=lambda x: x['count'])
        return {'ratings_total': total, 'ratings_n_categories': len(ratings),
                'ratings_positive': pos, 'ratings_negative': neg,
                'ratings_pos_ratio': pos / (total + 1),
                'ratings_neg_ratio': neg / (total + 1),
                'top_rating_name': top['name'], 'top_rating_count': top['count']}
    except:
        return {'ratings_total': 0, 'ratings_n_categories': 0,
                'ratings_positive': 0, 'ratings_negative': 0,
                'ratings_pos_ratio': 0.5, 'ratings_neg_ratio': 0.5,
                'top_rating_name': 'Unknown', 'top_rating_count': 0}

def parse_related_talks(rel_str):
    try:
        related = ast.literal_eval(rel_str)
        counts = [r.get('viewed_count', 0) for r in related]
        return {'related_n': len(counts),
                'related_views_mean': np.mean(counts) if counts else 0,
                'related_views_max': max(counts) if counts else 0,
                'related_views_min': min(counts) if counts else 0}
    except:
        return {'related_n': 0, 'related_views_mean': 0,
                'related_views_max': 0, 'related_views_min': 0}

def parse_tags(tags_str):
    try:
        tags = ast.literal_eval(tags_str)
        popular = {'technology','science','health','education','business',
                   'culture','society','mind','brain','psychology'}
        return {'tags_count': len(tags),
                'has_popular_tag': int(any(t in popular for t in tags))}
    except:
        return {'tags_count': 0, 'has_popular_tag': 0}

def transcript_features(text):
    if pd.isna(text) or text == '':
        return {'transcript_len': 0, 'transcript_words': 0,
                'transcript_sentences': 0, 'transcript_avg_word_len': 0,
                'transcript_has': 0, 'laughter_count': 0,
                'applause_count': 0, 'question_count': 0, 'exclamation_count': 0}
    text = str(text)
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    return {'transcript_len': len(text), 'transcript_words': len(words),
            'transcript_sentences': len(sentences),
            'transcript_avg_word_len': np.mean([len(w) for w in words]) if words else 0,
            'transcript_has': 1,
            'laughter_count': text.lower().count('laughter'),
            'applause_count': text.lower().count('applause'),
            'question_count': text.count('?'),
            'exclamation_count': text.count('!')}

def engineer_features(df):
    df = df.copy()
    df['film_date']      = pd.to_datetime(df['film_date'], unit='s')
    df['published_date'] = pd.to_datetime(df['published_date'], unit='s')
    df['film_year']      = df['film_date'].dt.year
    df['film_month']     = df['film_date'].dt.month
    df['pub_year']       = df['published_date'].dt.year
    df['pub_month']      = df['published_date'].dt.month
    df['days_to_publish'] = (df['published_date'] - df['film_date']).dt.days
    df['duration_min']   = df['duration'] / 60

    ratings_feats = df['ratings'].apply(parse_ratings).apply(pd.Series)
    df = pd.concat([df, ratings_feats], axis=1)
    related_feats = df['related_talks'].apply(parse_related_talks).apply(pd.Series)
    df = pd.concat([df, related_feats], axis=1)
    tags_feats = df['tags'].apply(parse_tags).apply(pd.Series)
    df = pd.concat([df, tags_feats], axis=1)
    trans_feats = df['transcript'].apply(transcript_features).apply(pd.Series)
    df = pd.concat([df, trans_feats], axis=1)

    df['title_len']   = df['title'].str.len()
    df['title_words'] = df['title'].str.split().str.len()
    df['desc_len']    = df['description'].str.len()
    df['desc_words']  = df['description'].str.split().str.len()

    spk_counts = df['main_speaker'].value_counts()
    df['speaker_freq'] = df['main_speaker'].map(spk_counts).fillna(1)
    ev_counts = df['event'].value_counts()
    df['event_freq'] = df['event'].map(ev_counts).fillna(1)

    for col in ['top_rating_name', 'event', 'main_speaker', 'speaker_occupation']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].fillna('Unknown').astype(str))

    return df

print("Engineering features...")
train = engineer_features(train)
test  = engineer_features(test)

# ─────────────────────────────────────────────
# 3. TF-IDF ON TRANSCRIPT
# Linear models handle high-dimensional sparse text very well —
# this is actually where they shine vs tree models
# ─────────────────────────────────────────────
print("Fitting TF-IDF on transcripts...")
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),     # single words + pairs of words
    min_df=3,               # ignore very rare terms
    sublinear_tf=True,      # apply log to term frequencies
    stop_words='english'
)

train_text = train['transcript'].fillna('').astype(str)
test_text  = test['transcript'].fillna('').astype(str)

tfidf_train = tfidf.fit_transform(train_text)   # fit only on train!
tfidf_test  = tfidf.transform(test_text)

# ─────────────────────────────────────────────
# 4. COMBINE NUMERIC + TF-IDF FEATURES
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'film_year', 'film_month', 'pub_year', 'pub_month', 'days_to_publish',
    'duration', 'duration_min', 'languages', 'num_speaker',
    'ratings_total', 'ratings_n_categories', 'ratings_positive', 'ratings_negative',
    'ratings_pos_ratio', 'ratings_neg_ratio', 'top_rating_count', 'top_rating_name_enc',
    'related_n', 'related_views_mean', 'related_views_max', 'related_views_min',
    'tags_count', 'has_popular_tag',
    'transcript_len', 'transcript_words', 'transcript_sentences',
    'transcript_avg_word_len', 'transcript_has',
    'laughter_count', 'applause_count', 'question_count', 'exclamation_count',
    'title_len', 'title_words', 'desc_len', 'desc_words',
    'speaker_freq', 'event_freq', 'event_enc', 'main_speaker_enc', 'speaker_occupation_enc',
]

X_num      = train[FEATURE_COLS].fillna(0).values
X_num_test = test[FEATURE_COLS].fillna(0).values

# Scale numeric features — critical for linear models
scaler    = StandardScaler()
X_num     = scaler.fit_transform(X_num)
X_num_test = scaler.transform(X_num_test)

# Combine numeric (dense) + TF-IDF (sparse) into one sparse matrix
X_all      = hstack([csr_matrix(X_num),      tfidf_train])
X_all_test = hstack([csr_matrix(X_num_test), tfidf_test])

y     = train['views'].clip(lower=1)
log_y = np.log1p(y)

print(f"Total features: {X_all.shape[1]} (numeric: {len(FEATURE_COLS)}, TF-IDF: {tfidf_train.shape[1]})")

# ─────────────────────────────────────────────
# 5. CROSS-VALIDATED TRAINING — 3 LINEAR MODELS
# ─────────────────────────────────────────────
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

models = {
    'Ridge':      Ridge(alpha=10.0),           # L2 penalty — shrinks all weights
    'Lasso':      Lasso(alpha=0.001),           # L1 penalty — zeros out weak features
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5),  # mix of both
}

oof_preds  = {name: np.zeros(len(X_all.toarray())) for name in models}
test_preds = {name: np.zeros(X_all_test.shape[0])  for name in models}

print("\nTraining with 5-fold CV...\n")

for name, model in models.items():
    print(f"── {name}")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all.toarray())):
        X_tr  = X_all[tr_idx]
        X_val = X_all[val_idx]
        y_tr  = log_y.iloc[tr_idx]
        y_val = log_y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        oof_preds[name][val_idx] = val_pred

        fold_views = np.expm1(val_pred).clip(min=1)
        true_views = y.iloc[val_idx].values
        fold_msle  = mean_squared_log_error(true_views, fold_views)
        print(f"   Fold {fold+1} | RMSLE: {np.sqrt(fold_msle):.5f}")

        test_preds[name] += model.predict(X_all_test) / N_FOLDS

    oof_views = np.expm1(oof_preds[name]).clip(min=1)
    oof_msle  = mean_squared_log_error(y, oof_views)
    print(f"   OOF RMSLE: {np.sqrt(oof_msle):.5f}\n")

# ─────────────────────────────────────────────
# 6. ENSEMBLE: AVERAGE ALL 3 MODELS
# ─────────────────────────────────────────────
ens_log   = np.mean([test_preds[n] for n in models], axis=0)
ens_views = np.expm1(ens_log).clip(min=1)

ens_oof_log   = np.mean([oof_preds[n] for n in models], axis=0)
ens_oof_views = np.expm1(ens_oof_log).clip(min=1)
ens_msle      = mean_squared_log_error(y, ens_oof_views)
print(f"Ensemble OOF RMSLE: {np.sqrt(ens_msle):.5f}")

# ─────────────────────────────────────────────
# 7. SUBMISSION
# ─────────────────────────────────────────────
submission = pd.DataFrame({
    'id': test['id'],
    'views': ens_views.astype(int)
})
submission = submission.drop_duplicates(subset='id', keep='first')
submission.to_csv('/mnt/user-data/outputs/submission_linear.csv', index=False)
print(f"\nSubmission saved → submission_linear.csv")
print(f"Rows: {len(submission)}")
print(submission.head(10))
print(f"\nViews range: {submission['views'].min():,} – {submission['views'].max():,}")
print(f"Views median: {submission['views'].median():,.0f}")
