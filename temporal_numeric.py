import numpy as np
import json
import csv
import re
import spacy
from tqdm import tqdm
from datetime import datetime
from dateutil import parser
from typing import Optional, List
import statistics


nlp = spacy.load("en_core_web_sm")
months = {
    "January": ["January", "Jan", "Januar", "Janeiro", "Enero", "Gennaio", "Ene", "Gen"],
    "February": ["February", "Feb", "Februar", "Fevereiro", "Febrero", "Febbraio", "Fev"],
    "March": ["March", "Mar", "März", "Março", "Marzo", "Mär"],
    "April": ["April", "Apr", "Avril", "Abril", "Aprile", "Avr", "Abr",],
    "May": ["May", "Mai", "Mayo", "Maio", "Maggio", "Mag"],
    "June": ["June", "Jun", "Juni", "Juno", "Junho", "Giugno", "Giu", "Juin"],
    "July": ["July", "Jul", "Juli", "Julho", "Julio", "Luglio", "Lug"],
    "August": ["August", "Aug", "Agosto", "Août", "Aoû", "Ago", "Aot", "Aou"],
    "September": ["September", "Sep", "Sept", "Septiembre", "Settembre", "Setembro", "Set", "Sett"],
    "October": ["October", "Oct", "Oktober", "Outubro", "Octubre", "Ottobre", "Out", "Ott", "Okt"],
    "November": ["November", "Nov", "Novembre", "Novembro", "Noi"],
    "December": ["December", "Dec", "Dezember", "Diciembre", "Dezembro", "Dicembre", "Dez", "Dic"]
}

months_pattern = "|".join(
    f"{name}|{'|'.join(short.lower() for short in versions)}|{'|'.join(short.capitalize() for short in versions)}"
    for name, versions in months.items()
)
date_pattern = re.compile(rf'\b(\d{{1,2}}\s({months_pattern})\s\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})\b', re.IGNORECASE)

def extract_first_date_with_multilang_regex(text: str) -> Optional[str]:
    """
    Extracts the first date from a given text using regular expressions that account for multiple languages.
    
    Parameters:
    text (str): The input text from which to extract the date.
    
    Returns:
    Optional[str]: The first date found in the text or None if no date is found.
    """
    match = date_pattern.search(text)
    return match.group(0) if match else None


def extract_first_date_with_nlp(text: str) -> Optional[str]:
    """
    Extracts the first date from a given text using spaCy.
    
    Parameters:
    text (str): The input text from which to extract the date.
    
    Returns:
    Optional[str]: The first date found in the text or None if no date is found.
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DATE":
            return ent.text
    return None


def parse_date(date_str: str) -> Optional[str]:
    """
    Parses a date string and handles incomplete dates.
    
    Parameters:
    date_str (str): The date string to parse.
    
    Returns:
    Optional[str]: A formatted date string or None if no valid date is found.
    """
    try:
        for english_month, month_variants in months.items():
            for variant in month_variants:
                if variant.lower() in date_str.lower():
                    date_str = date_str.lower().replace(variant.lower(), english_month)
                    break

        date = parser.parse(date_str, fuzzy=True, default=parser.parse("01 Jan 2022"))
        return date.strftime("%Y-%m-%d")
    except parser.ParserError:
        return None


def extract_and_format_first_date(text: str) -> Optional[str]:
    """
    Extracts the first date from text using regex and spaCy and formats it.
    
    Parameters:
    text (str): The input text from which to extract the date.
    
    Returns:
    Optional[str]: A formatted date string or None if no valid date is found.
    """
    if text is None:
        return None

    date_str = extract_first_date_with_multilang_regex(text)
    if not date_str:
        date_str = extract_first_date_with_nlp(text)
    
    if date_str:
        try:
            parsed_date = parse_date(date_str)
            return parsed_date
        except Exception:
            if any(month in date_str.lower() for month in months_pattern.split("|")):
                try:
                    parsed_date = parser.parse(date_str, default=parser.parse("01 Jan 2022")).strftime("%Y-%m-%d")
                    return parsed_date
                except Exception:
                    pass
            if any(year.isdigit() for year in re.findall(r'\d{4}', date_str)):
                year = re.findall(r'\d{4}', date_str)[0]
                parsed_date = f"{year}-01-01"
                return parsed_date
    return None

# prompt = "In June 2018, the House of Representatives passed a bill that would deny all veterans access to food stamps."
# print(extract_and_format_first_date(prompt))  # Output: 2018-06-01

def compute_temporal_difference(time1, time2, alpha=1.0):
    if time1 and time2:
        time1 = datetime.strptime(time1, "%Y-%m-%d")
        time2 = datetime.strptime(time2, "%Y-%m-%d")
        
        return alpha / abs((time1 - time2).days + 1e5)  
    return 0


def extract_numbers(text: str) -> List[float]:
    int_numbers = [int(word) for word in re.findall(r'\b\d+\b', text)]
    float_numbers = [float(word) for word in re.findall(r'\b\d+\.\d+\b', text)]
    percent_numbers = [float(word.replace('%', '')) / 100 for word in re.findall(r'\b\d+%\b', text)]
    numbers = list(set(int_numbers + float_numbers + percent_numbers))
    return numbers


def filter_date_numbers(numbers: List[int], text: str) -> List[int]:
    dates = re.findall(r'\b\d{1,2} \w+ \d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}\b', text)
    date_numbers = set()
    for date in dates:
        date_numbers.update(extract_numbers(date))
    filtered_numbers = [num for num in numbers if num not in date_numbers]
    return filtered_numbers


def compute_numerical_relevance(claim, evidence):
    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(evidence)
    filtered_claim_numbers = filter_date_numbers(claim_numbers, claim)
    filtered_evidence_numbers = filter_date_numbers(evidence_numbers, evidence)
    common_numbers = set(filtered_claim_numbers).intersection(set(filtered_evidence_numbers))
    return len(common_numbers)


def reorder_evidences(claim, evidences, semantics, normalization=False, claim_type='temporal', crawled_date=None):
    # semantic_scores:
    # semantic_scores = [evidence['score'] for evidence in evidences]
    # semantic_scores = [max(100 - i,0) for i in range(len(evidences))]
    semantic_scores = semantics
    # print(len(semantic_scores))
    # print(semantic_scores)
    if normalization is True:
        max_semantic_score = max(semantic_scores)
        min_semantic_score = min(semantic_scores)
        if max_semantic_score == min_semantic_score:
            normalized_semantic_scores = [0 for _ in semantic_scores]
        else:
            normalized_semantic_scores = [(score - min_semantic_score) / (max_semantic_score - min_semantic_score) for score in semantic_scores]
    else:
        normalized_semantic_scores = semantic_scores

    # temporal scores
    mean_s = statistics.mean(semantic_scores)
    stdev_s = statistics.stdev(semantic_scores)
    claim_date = extract_and_format_first_date(crawled_date)
    if not claim_date:
        claim_date = extract_and_format_first_date(claim)
    temporal_scores = [compute_temporal_difference(claim_date, extract_and_format_first_date(evidence)) for evidence in evidences]
    mean_t = statistics.mean(temporal_scores)
    stdev_t = statistics.stdev(temporal_scores)
    if stdev_t == 0:
        normalized_temporal_scores = [0 for _ in temporal_scores]
    else:
        normalized_temporal_scores = [(score - mean_t) / stdev_t * stdev_s + mean_s for score in temporal_scores]

    # numerical_scores:
    numerical_scores = [compute_numerical_relevance(claim, evidence) for evidence in evidences]
    max_numerical_score = max(numerical_scores)
    min_numerical_score = min(numerical_scores)
    if max_numerical_score == min_numerical_score:
        normalized_numerical_scores = [0 for _ in numerical_scores]
    else:
        normalized_numerical_scores = [(score - min_numerical_score) / (max_numerical_score - min_numerical_score) for score in numerical_scores]
    
    if claim_type == 'temporal':
        weights = np.array([1, 1, 0])
    elif claim_type == 'statistical':
        weights = np.array([1, 0, 1])
    else:
        weights = np.array([1, 0, 0])
    
    combined_scores = np.dot(np.array([normalized_semantic_scores, 
                                    normalized_temporal_scores, 
                                    normalized_numerical_scores]).T, 
                            weights)

    sorted_evidences_scores = list(sorted(zip(evidences, combined_scores), key=lambda x: x[1], reverse=True))
    sorted_evidences = [ev for ev, _ in sorted_evidences_scores]
    return sorted_evidences


def convert_to_float(element):
    try:
        return float(element)
    except ValueError:
        return element
    

def modify_dataset():
    def run(data):
        with open(f'output/bm25_top_100_{data}_reordered.json', 'r') as f:
            claims = json.load(f)

        with open(f'output/bm25_top_100_{data}_reordered_scores.csv', 'r') as f:
            reader = csv.reader(f)
            semantic_scores = [[convert_to_float(item) for item in row] for row in reader]
            semantic_scores = [row for row in semantic_scores if row]

        reordered_claims = []
        i = 0
        for claim in tqdm(claims):
            cl = claim['claim']
            evidence = claim['top_n']
            crawled_date = claim['crawled_date'] if 'crawled_date' in claim else None
            if claim['taxonomy_label'] == 'temporal' or claim['taxonomy_label'] == 'statistical':
                evidence = reorder_evidences(cl, evidence, semantic_scores[i], False, claim['taxonomy_label'], crawled_date)
            
            claim['top_n'] = evidence 
            reordered_claims.append(claim)
            
            i += 1
        
        with open(f'output/bm25_top_100_{data}_reordered_improved.json', 'w') as f:
            json.dump(reordered_claims, f, indent=4)
    
    run('train')
    run('val')
    run('test')

modify_dataset()
