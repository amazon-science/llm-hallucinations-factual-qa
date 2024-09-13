import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool


N_PROCESSES=8  # Num CPU cores for processing TREX files in parallel
TREX_PATH = Path('./trex_uncompressed/')  # Location of the uncompressed TREX dir
WIKI_CODINGS = {
    'P19': 'place_of_birth', # surface form: place of birth & born in
    'P112': 'founders',    # surface form: founder & founded by
    'P36': 'capitals',        # surface form: capital & capital city -- P1376 doesn't add much on top so removed
}
MAX_PER_RELATION = 10000  # Save no more than these many entries per Wiki relationship


def collect_entry(triplet):
    triplet_data = {}
    triplet_data['predicate_surfaceform'] = triplet['predicate']['surfaceform']
    triplet_data['predicate_uri'] = triplet['predicate']['uri']
    triplet_data['predicate_wikidata_code'] = Path(triplet['predicate']['uri']).name
    triplet_data['predicate_annotator'] = triplet['predicate']['annotator']
    triplet_data['object_surfaceform'] = triplet['object']['surfaceform']
    triplet_data['object_uri'] = triplet['object']['uri']
    triplet_data['object_wikidata_code'] = Path(triplet['object']['uri']).name
    triplet_data['object_annotator'] = triplet['object']['annotator']
    triplet_data['subject_surfaceform'] = triplet['subject']['surfaceform']
    triplet_data['subject_uri'] = triplet['subject']['uri']
    triplet_data['subject_wikidata_code'] = Path(triplet['subject']['uri']).name
    triplet_data['subject_annotator'] = triplet['subject']['annotator']
    return triplet_data


def extract_all_triples(t_rex_file):
    triples_data = {'predicate_surfaceform': [],
                    'predicate_uri': [],
                    'predicate_wikidata_code': [],
                    'predicate_annotator': [],
                    'object_surfaceform': [],
                    'object_uri': [],
                    'object_wikidata_code': [],
                    'object_annotator': [],
                    'subject_surfaceform': [],
                    'subject_uri': [],
                    'subject_wikidata_code': [],
                    'subject_annotator': [],}
    with open(t_rex_file, 'r') as infile:
        json_contents = json.loads(infile.read())
    for contents in json_contents:
        if contents['triples']:
            for triple in contents['triples']:
                triplet_data = collect_entry(triple)
                for key, value in triplet_data.items():
                    triples_data[key].append(value)
    return pd.DataFrame(triples_data)


def filter_subj_obj(df: pd.DataFrame) -> pd.DataFrame:
    pronouns = ["i", "you", "he", "she", "it", "we", "they"]
    df = df[~(df["subject_surfaceform"].isna() | df["object_surfaceform"].isna())]
    df = df[~(df["subject_surfaceform"].str.lower().isin(pronouns) | df["object_surfaceform"].str.lower().isin(pronouns))]
    return df


def extract_selected_relation(df: pd.DataFrame, relationship_id: str) -> pd.DataFrame:
    df_selected_predicate = df[df["predicate_wikidata_code"] == relationship_id][
        ["subject_surfaceform", "object_surfaceform"]].drop_duplicates()
    subj_to_obj_options = defaultdict(list)
    for _, row in df_selected_predicate.iterrows():
        subj, obj = row["subject_surfaceform"], row["object_surfaceform"]
        # Convert subject to title case (New south Wales = New South Wales) but leave out acronyms like USA
        if len(subj.split()) > 1 or (len(subj.split()) == 1 and subj.upper() != subj):
            subj = subj.title()
        # For place of birth, filter out single names like "Chen"
        if relationship_id == "P19" and len(subj.split()) == 1:
            continue
        subj_to_obj_options[subj].append(obj)
        # For place of birth, sometimes we are given info like "Paris, France"
        # Let us consider both Paris and France as correct answers
        if relationship_id == "P19" and "," in obj:
            subj_to_obj_options[subj].extend(map(str.strip, obj.split(",")))
        # For capitals, both "Keosauqua, Iowa" and "Keosauqua" are correct answers
        if relationship_id == "P36" and "," in obj:
            subj_to_obj_options[subj].append(obj.split(",")[0])

    subj_to_obj_options = dict(subj_to_obj_options)
    for k, v in subj_to_obj_options.items():
        subj_to_obj_options[k] = set(v)
    for k, v in subj_to_obj_options.items():
        subj_to_obj_options[k] = "<OR>".join(v)
    return pd.DataFrame(subj_to_obj_options.items(), columns=["subject", "object"])


def main():
    all_triples_file = Path("all_triples.csv")
    if not all_triples_file.is_file():
        with Pool(8) as pool:
            if not TREX_PATH.is_dir():
                raise Exception(
                    f"{TREX_PATH} is not a directory. Download and uncompress the trex data "
                    f"from: https://doi.org/10.6084/m9.figshare.5146864.v1"
                )
            t_rex_files = list(TREX_PATH.rglob('*.json*'))
            triples_data = list(tqdm(pool.imap(extract_all_triples, t_rex_files), total=len(t_rex_files)))
        df = pd.concat(triples_data)
        df.to_csv(all_triples_file, index=False)
    else:
        df = pd.read_csv(all_triples_file)
    df = filter_subj_obj(df)
    for relationship_id, file_name in WIKI_CODINGS.items():
        df_relation = extract_selected_relation(df, relationship_id)
        df_relation.sample(min(len(df_relation), MAX_PER_RELATION), random_state=122333).to_csv(f"{file_name}.csv", index=False)


if __name__ == '__main__':
    main()
