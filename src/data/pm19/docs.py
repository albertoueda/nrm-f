import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Set, Dict

project_dir = Path(__file__).resolve().parents[3]


def load_raw_docs() -> Dict:
    docs = defaultdict(dict)

    with open(f'{project_dir}/data/raw/docs.json') as file:
        keys = [
            'abstracttext',
            'abstracttext_orig',
            'articletitle',
            'docno',
            'nlmcategorybackground',
            'nlmcategoryconclusions',
            'nlmcategorymethods',
            'nlmcategoryresults',
        ]
        raw = json.load(file)
        for docno in raw:
            for key in keys:
                docs[int(docno)][key] = raw[docno][key]

    return docs