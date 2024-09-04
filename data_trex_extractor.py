import random
from string import Template
import logging
from logging import Logger
import json
import pickle
from pathlib import Path
import zipfile
import tempfile
from collections import defaultdict
import itertools
from typing import List, Optional, Dict, NamedTuple, Tuple

import pandas as pd
import tqdm
import urllib.request

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def download_if_not_exists(file_name: str, url: str, logger: Logger) -> None:
    """
    Given a file name, check if it is found locally. If not, download it from
    the given URL.

    :param file_name: Local name of the file
    :param url: URL where the file can be retrieved from
    :param logger: For logging
    :return: None
    """
    data_file = Path(file_name)
    if not data_file.is_file():
        if not url.startswith("https://"):
            raise ValueError("Unsecure url. Make sure to use https URLs.")
        logger.info(f"File {file_name} not found. Retrieving from {url}.")
        Path("/".join(file_name.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, file_name)
        logger.info("Downloaded.")
    else:
        logger.info(f"File {file_name} found locally.")
    return

# A single knowledge category could be covered by multiple TREx predicates, lets construct the
# mapping here
KNOWLEDGE_CATEGORY_TO_TREX_PREDICATES: Dict[str, List[str]] = {
    "Capitals": ["capital of", "capital", "capital city"],
    "Founders": ["founder"],
    "BirthPlace": ["born in"]
}

KNOWLEDGE_CATEGORY_TO_PROMPT_TEMPLATE = {
    "Capitals": Template("$inp is the capital of"),
    "Founders": Template("$inp was founded by"),
    "BirthPlace": Template("$inp was born in"),
}

KNOWLEDGE_CATEGORY_TO_FILENAME = {
    "Capitals": "capitals.csv",
    "Founders": "founders.csv",
    "BirthPlace": "place_of_birth.csv",
}

TREX_PREDICATES_TO_KNOWLEDGE_CATEGORY = {}
for category, predicates in KNOWLEDGE_CATEGORY_TO_TREX_PREDICATES.items():
    for predicate in predicates:
        TREX_PREDICATES_TO_KNOWLEDGE_CATEGORY[predicate] = category


class Triplet(NamedTuple):
    """Holds the subject, object and the predicate as seen in TREx corpus."""

    subj: str
    obj: str
    predicate: str  # Specified how subject and object are related.
    # For instance "capital of" means [Subject] is the capital of [Object]


# Stores the mapping from predicate to list of triplets with this predicate.
TripletDict = Dict[str, List[Triplet]]


def convert_triplet_to_prompt_and_answer(triplet: Triplet) -> Tuple[str, str, str, str]:
    """
    Convert TREx triplets into prompts and respective answers.

    The prompt construction consists of two steps:
    1. A knowledge category can consist of multiple TREx predicates, for instance, to load information
    about category "Capitals" we might need to gather triplets from predicates "capital of",
    "capital" and "capital city". Here we merge predicates into categories.
    2. Depending on the predicate, sometimes the subject and other times
    the object constitutes the answer to the prompt. For instance,
    "Ankara [Subject] is the capital of Turkey [Object]" and "Libya's [Subject] capital is
    Tripoli [Object]" need to be processed differently to form the prompt "X is the capital of Y".
    We handle these substitutions on a case to case basis here.

    :param triplet: A single triplet from TREx.
    :return: The triplet formed into a prompt and the expected answer.
    """
    predicate = triplet.predicate
    prompt_template = KNOWLEDGE_CATEGORY_TO_PROMPT_TEMPLATE[TREX_PREDICATES_TO_KNOWLEDGE_CATEGORY[predicate]]

    all_valid_predicates = itertools.chain(*KNOWLEDGE_CATEGORY_TO_TREX_PREDICATES.values())

    # In most cases, the prompt is formed by [subject] [predicate] [object].
    # For instance Ankara [Subject] is the capital of Turkey [Object]
    # However, for some rare cases, the subject/object order is reversed, like [object] [predicate] [subject]
    # For instance Libya's [Subject] capital is Tripoli [Object]. Lets handle these cases separately.
    if predicate == "capital":
        # Libya's [Subject] capital is Tripoli [Object]
        question = triplet.obj
        answer = triplet.subj
    elif predicate == "capital city":
        # The capital city of Qatar [Subject] is Doha [Object].
        question = triplet.obj
        answer = triplet.subj
    elif predicate in all_valid_predicates:
        # All remaining cases are where the subject is the question and the object is the answer
        question = triplet.subj
        answer = f"{triplet.obj}"
    else:
        raise ValueError(f"Unknown predicate {triplet.predicate}.")
    prompt = prompt_template.substitute(inp=question)
    return (prompt, answer, question, predicate)


def load_prompts_from_trex(
    trex_raw_file: Path,
    categories: List[str],
    triplets_cache_file: Path,
    *,
    answer_delimiter: str = "<OR>",
    max_prompts_per_category: Optional[int] = None,
    seed: int = 42,
    save_by_category = True
) -> pd.DataFrame:
    """
    Load knowledge triplets (subject, object, predicate) from TREx and convert them to prompts
    and expected answers. For instance, the triplet (Berlin, Germany, Capital of) will get
    coverted into the prompt "Berlin is the capital of" with the expected answer being "Germany".

    :param trex_raw_file: The path to the raw trex data file downloaded from the source https://doi.org/10.6084/m9.figshare.5146864.v1
        If the file is not found, we will download it.
    :param categories: Knowledge categories that should be loaded, e.g., "Capitals", "Founders".
    :param triplets_cache_file: The file containing the cleaned version of the TREX data, that is,
        (subject, object, predicate) relationship triplets which is faster to process.
        TREX is a large corpus of several GBs. Downloading, unzipping and cleaning the data into
        relationship triplets takes several minutes. If this file is found, we directly load cleaned
        triplets from here. Else, we load from scratch and store in this file.
    :param answer_delimiter: Some prompts can have multiple answers. We combine all the answers into a single
        string and use the delimiter to separate them. For instance, if the answers are ["UK", "England"]
        and the delimiter="<OR>", then the combined answer will be represented as "UK<OR>England".
    :param max_prompts_per_category: Max. number of prompts to load per category.
    :param seed: Random seed for storing the prompts.
    :returns: A ray dataset with prompts and answers stored in columns called "prompt" and
        "expected_answer". "prompt" is a string whereas "expected_answer" is a list of strings,
        that is, the list of acceptable answers.
    """

    predicate_to_triplets = load_trex_triplets(trex_raw_file, triplets_cache_file)
    data_items = []
    by_category = {}
    
    if save_by_category:
        for category in categories:
            by_category[category] = []
        
    for category in categories:
        predicates = KNOWLEDGE_CATEGORY_TO_TREX_PREDICATES[category]
        triplets = itertools.chain(*[predicate_to_triplets[predicate] for predicate in predicates])

        # Prompts can have multiple correct answers, combine them first
        prompt_to_answers = defaultdict(list)  # Same prompt could have many answers
        for prompt_answer_question_predicate in map(convert_triplet_to_prompt_and_answer, triplets):
            prompt, answer, question, predicate = prompt_answer_question_predicate
            prompt_to_answers[prompt].append(answer)
            if save_by_category: 
                if category == "Capitals":
                    # Because of the way we'll formulate the question
                    by_category[category].append(
                        {
                        "subject": answer,
                        "answer": question
                        }
                    )
                else:
                     by_category[category].append(
                        {
                        "subject": question,
                        "answer": answer
                        }
                    )
        # Subsample if needed
        if max_prompts_per_category is not None and len(prompt_to_answers) > max_prompts_per_category:
            all_keys = list(prompt_to_answers.keys())
            random.seed(seed)
            random.shuffle(all_keys)
            subset = all_keys[:max_prompts_per_category]
            prompt_to_answers = {k: prompt_to_answers.get(k) for k in subset}

        # Store the prompts to be formed into pandas dataframe
        data_items.extend(
            [
                {
                    "prompt": prompt,
                    "answers": answer_delimiter.join(set(answers)),  # Same answers might have repeated in TREx corpus
                    "knowledge_category": category,
                }
                for prompt, answers in prompt_to_answers.items()
            ]
        )

    if save_by_category: 
        for category in categories:
            filename = KNOWLEDGE_CATEGORY_TO_FILENAME[category]
            df = pd.DataFrame(by_category[category])
            print (f"Category: {category} has df of len {len(df.index)}")
            print (df.head(10))
            df.to_csv(filename, index=True, header=True)
                
            df = pd.read_csv(filename)
            print (df.head(10))
    
    return pd.DataFrame(data_items)


def load_trex_triplets(trex_raw_file: Path, triplets_cache_file: Path) -> TripletDict:
    """
    Load the (subject, object, predicate) triplets from the TREx corpus.
    If not found in the cache file, download and process the TREx corpus to form triplets, and store to the cache file.

    :param trex_raw_file: Same as in `load_prompts_from_trex`.
    :param triplets_cache_file: Same as in `load_prompts_from_trex`.
    :returns: A dict containing the mapping from predicates to list of triplets with that predicate.
    """

    if triplets_cache_file.is_file():
        with open(triplets_cache_file, "rb") as f:
            predicate_to_triplets = pickle.load(f)
    else:
        with tempfile.TemporaryDirectory() as raw_extraction_dir:
            raw_extraction_dir = Path(raw_extraction_dir)
            download_and_extract_trex_raw_data(trex_raw_file, raw_extraction_dir=raw_extraction_dir)
            predicate_to_triplets = extract_triplets_from_raw_files(raw_extraction_dir)

        predicate_to_triplets = cleanup_triplets(predicate_to_triplets)
        with open(triplets_cache_file, "wb") as f:
            pickle.dump(predicate_to_triplets, f)
    logger.info(f"Loaded a total of {len(predicate_to_triplets)} predicates.")
    return predicate_to_triplets


def extract_triplets_from_raw_files(raw_extraction_dir: Path) -> TripletDict:
    """
    Extract the subject/object/predicate triplets from the TREx data and store separately for
    quicker processing later.

    :param raw_extraction_dir: A directory where the raw trex.zip file is extracted. This directory
        will contains a number of jsons like `re_nlg_*.json`.
    :returns: A dict containing the mapping from predicates to list of triplets with that predicate.
    """

    predicate_to_triplets = defaultdict(list)
    raw_files = raw_extraction_dir.glob("*.json")
    for file_name in tqdm.tqdm(raw_files):
        with open(file_name, "r") as f:
            data = json.load(f)
        for document in data:
            for triple in document["triples"]:
                # Sometimes the predicate is not visible in the sentence and we have to
                # follow a link. Let us ignore these cases.
                predicate = triple["predicate"]["surfaceform"]
                if predicate is None:
                    continue
                subj = triple["subject"]["surfaceform"]
                obj = triple["object"]["surfaceform"]
                predicate_to_triplets[predicate].append(Triplet(subj=subj, obj=obj, predicate=predicate))
    return predicate_to_triplets


def download_and_extract_trex_raw_data(raw_file_path: Path, *, raw_extraction_dir: Optional[Path] = None):
    """
    Download the raw TREx file which is a zip and extract it into a directory. If the extraction
    directory is not provided, extract into the same directory that contains the raw zip file.

    :param raw_file_path: The full path where the raw zip file should be stored.
    :param raw_extraction_dir: The directory where the trex zip file should be extracted.
    """
    trex_source = 'https://figshare.com/ndownloader/files/8760241' # Full
    #trex_source = "https://figshare.com/ndownloader/files/8768701"  # Sample
    download_if_not_exists(str(raw_file_path), trex_source, logger)
    logger.info("Extracting raw data file.")
    if raw_extraction_dir is None:
        extraction_dir = raw_file_path.parent
        logger.info(f"TREX extraction directory not provided. Unzipping in {extraction_dir}.")
    with zipfile.ZipFile(raw_file_path, "r") as zip_ref:
        zip_ref.extractall(raw_extraction_dir)
    logger.info(f"Extraction complete.")


def cleanup_triplets(predicate_to_triplets: TripletDict) -> TripletDict:
    """
    Different functions to cleanup the TREx data.

    :param predicate_to_triplets: Dictionary mapping predicates to triplets.
    :returns: Cleaned version of the input where the offending triplets caught by the filters
        or the predicates that contain no predicates are dropped.
    """
    predicate_to_triplets = {k: filter_invalid_subject_objects(v) for k, v in predicate_to_triplets.items()}
    predicate_to_triplets = remove_duplicate_triplets(predicate_to_triplets)
    predicate_to_triplets = remove_empty_predicates(predicate_to_triplets)
    return predicate_to_triplets


def filter_invalid_subject_objects(triplets: List[Triplet]) -> List[Triplet]:
    """
    Filter out invalid subjects and objects from the TREX data. Subjects and objects are invalid
    if they consist of pronouns.

    :param triplets: List of triplets
    :returns: Cleaned list of triplets where the offending ones are dropped.
    """

    valid_triplets = []
    pronouns = ["i", "you", "he", "she", "it", "we", "they"]
    for triplet in triplets:
        subj, obj = triplet.subj, triplet.obj
        # Remove the cases where the subject or object is a pronoun
        if subj.lower() in pronouns or obj.lower() in pronouns:
            continue
        # Remove cases when we have all uppercase subjects or objects like CBRM
        if subj == subj.upper() or obj == obj.upper():
            continue
        valid_triplets.append(triplet)
    return valid_triplets


def remove_duplicate_triplets(predicate_to_triplets: TripletDict) -> TripletDict:
    """
    TREx might contain duplicate triplets where the subject, object and the predicate are the same.
    Filter out the duplicates.

    :param predicate_to_triplets: Dictionary mapping predicates to triplets.
    :returns: De-duplited version of the input.
    """
    return {k: list(set(v)) for k, v in predicate_to_triplets.items()}


def remove_empty_predicates(predicate_to_triplets: TripletDict) -> TripletDict:
    """
    After filtering out triplets, we might end up with cases where some predicates do not have any
    triplets remaining because they were caught by our filters. Remove these empty predicates.

    :param predicate_to_triplets: Dictionary mapping predicates to triplets.
    :returns: Same as the input but with empty predicates removed.
    """
    all_predicates = list(predicate_to_triplets.keys())
    for predicate in all_predicates:
        if len(predicate_to_triplets[predicate]) == 0:
            del predicate_to_triplets[predicate]
    return predicate_to_triplets


if __name__ == "__main__":
    # download_and_extract_trex_raw_data(Path("./trex.zip"))
    # load_trex_triplets(Path("../trex.zip"), Path("../cache.json"))
    trex_raw_path = Path("trex.zip")
    categories = list(KNOWLEDGE_CATEGORY_TO_TREX_PREDICATES.keys())
    cache_path = Path("./cache.json")
    fact_data = load_prompts_from_trex(
        trex_raw_path,
        categories,
        cache_path,
        max_prompts_per_category=None,
        seed=11223,
        save_by_category=True
    )
    #print (fact_data.head(10))