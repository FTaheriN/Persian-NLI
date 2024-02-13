from parsinorm import General_normalization
import hazm
import re



normalizer = hazm.Normalizer()
def hazm_preprocessing(txt):
    txt = txt.strip()
    txt = normalizer.normalize(txt)
    extra_characters = re.compile("["
    u"\u200c"
    u"\u200d"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\u3030"
    u"\ufe0f"
    u"\u2069"
    u"\u2066"
    u"\u2068"
    u"\u2067"
    "]+", flags=re.UNICODE)

    txt = extra_characters.sub(r'', txt)
    return txt


def preprocess_data(df):
    general_normalization = General_normalization()

    new_sent1 = [general_normalization.alphabet_correction(snt) for snt in df['premise']]
    new_sent1 = [general_normalization.semi_space_correction(s) for s in new_sent1]
    new_sent2 = [general_normalization.alphabet_correction(snt) for snt in df['hypothesis']]
    new_sent2 = [general_normalization.semi_space_correction(s) for s in new_sent2]
    df['premise'] = new_sent1
    df['hypothesis'] = new_sent2

    df['premise'] = (df['premise']).apply(hazm_preprocessing)
    df['hypothesis'] = (df['hypothesis']).apply(hazm_preprocessing)

    y = np.array(pd.get_dummies(df['label']), dtype=np.int32)

    return df, y