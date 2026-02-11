import pandas as pd
import unicodedata
import glob
import os
import re
from pathlib import Path
from Levenshtein import distance as levenshtein_distance
import math
import ast
import cmudict

# Get the directory where this utils.py file is located
UTILS_DIR = Path(__file__).parent
# The speech_to_text directory contains the data files
SPEECH_TO_TEXT_DIR = UTILS_DIR / "speech_to_text"


def normalize_text(text):
    if pd.isna(text):
        return str(text)
    
    # Convert to string and lowercase
    text = str(text).lower()
    text = text.strip()

    # Remove accents/diacritics using Unicode normalization
    # NFD = decompose characters into base + combining characters
    # Then filter out combining characters (accents)
    # This handles é→e, è→e, ê→e, ë→e, ç→c, etc.
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Normalize different types of apostrophes to a standard one
    apostrophe_variants = ["'", "'", "'", "`",  "ʿ", "'", "†", "'", "'"]
    for variant in apostrophe_variants:
        text = text.replace(variant, "'")
    
    # Normalize different types of spaces to regular space
    # This includes non-breaking space (\xa0), em space, en space, etc.
    space_variants = ['\xa0', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', 
                      '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a',
                      '\u202f', '\u205f', '\u3000']
    for variant in space_variants:
        text = text.replace(variant, ' ')
    
    # Normalize multiple spaces to single space
    text = ' '.join(text.split())
    
        
    text = text.replace(".", "").replace("!", "").replace("?", "").replace("°", "")
    text = text.replace("i am", "i'm")
    
    # Remove duplicate "I'm on" patterns (e.g., "i'm on i'm on street" → "i'm on street")
    # This handles cases where "I'm on" audio is prepended to recordings that already say "I'm on"
    text = re.sub(r"^i'm on\s+(i'm on|im on|i'm on|im on)\s+", "i'm on ", text)

    if text == "i'm on bay shore":
        text = "i'm on bayshore"
    if text == "i'm on hunters point":
        text = "i'm on hunter's point"
    if text == "i'm on caesar chavez":
        text = "i'm on cesar chavez"
    if text == "i'm on monterrey":
        text = "i'm on monterey"
    if text == "i'm on servantes":
        text = "i'm on cervantes"
    if text == "i'm on almany":
        text = "i'm on alemany"
    if text == "i'm on alamany":
        text = "i'm on alemany"
    if text == "i'm on twinpeaks":
        text = "i'm on twin peaks"
    if text == "i'm on arguelo":
        text = "i'm on arguello"
    if text == "i'm on burnal heights":
        text = "i'm on bernal heights"
    if text == "i'm on terry, francois":
        text = "i'm on terry francois"
    return text

def process_primary_language(text):
    # Split languages and normalize Chinese variants
    languages = text.split(", ")
    normalized = []
    for lang in languages:
        # Merge Mandarin and Cantonese into Chinese
        if lang.lower() in ['mandarin', 'cantonese']:
            normalized.append('Chinese')
        else:
            normalized.append(lang)
    
    # Remove duplicates by converting to set, then sort
    unique_languages = sorted(set(normalized))
    return str(unique_languages)[1:-1]

    
# Create a column to identify language groups
def categorize_language(lang_str):
    if pd.isna(lang_str):
        return 'Non-English'
    # Check if English is present
    has_english = 'English' in lang_str or 'english' in lang_str.lower()
    # Check if it's ONLY English (no commas or other languages)
    is_only_english = has_english and ',' not in lang_str and lang_str.strip().lower() in ['english', "'english'"]
    
    if is_only_english:
        return 'English only'
    elif has_english:
        return 'Multilingual (English)'
    else:
        return 'Non-English'

def group_language_by_family(lang_str):
    """
    Group a language (or list of languages) by their language family.
    Excludes English from classification - focuses on non-English languages.
    
    Args:
        lang_str: String containing one or more languages (comma-separated)
        
    Returns:
        String with the language family, or 'English only' if only English is present
        
    Examples:
        >>> group_language_by_family("'English'")
        'English only'
        >>> group_language_by_family("'Spanish', 'English'")
        'Romance'
        >>> group_language_by_family("'Chinese'")
        'Sino-Tibetan'
    """
    # Define language family mappings (excluding English)
    language_families = {
        # Germanic languages (excluding English)
        'german': 'Germanic',
        'dutch': 'Germanic',
        'swedish': 'Germanic',
        'norwegian': 'Germanic',
        'danish': 'Germanic',
        'icelandic': 'Germanic',
        'afrikaans': 'Germanic',
        
        # Romance languages
        'spanish': 'Romance',
        'french': 'Romance',
        'italian': 'Romance',
        'portuguese': 'Romance',
        'romanian': 'Romance',
        'catalan': 'Romance',
        
        # Slavic languages
        'russian': 'Slavic',
        'polish': 'Slavic',
        'czech': 'Slavic',
        'ukrainian': 'Slavic',
        'bulgarian': 'Slavic',
        'serbian': 'Slavic',
        'croatian': 'Slavic',
        'slovak': 'Slavic',
        
        # Sino-Tibetan languages
        'chinese': 'Sino-Tibetan',
        'mandarin': 'Sino-Tibetan',
        'cantonese': 'Sino-Tibetan',
        'tibetan': 'Sino-Tibetan',
        
        # Austronesian languages
        'filipino': 'Austronesian',
        'tagalog': 'Austronesian',
        'indonesian': 'Austronesian',
        'malay': 'Austronesian',
        
        # Austroasiatic languages
        'vietnamese': 'Austroasiatic',
        'khmer': 'Austroasiatic',
        
        # Koreanic
        'korean': 'Koreanic',
        
        # Japonic
        'japanese': 'Japonic',
        
        # Afro-Asiatic languages
        'arabic': 'Afro-Asiatic',
        'hebrew': 'Afro-Asiatic',
        'amharic': 'Afro-Asiatic',
        
        # Indo-Iranian languages
        'hindi': 'Indo-Iranian',
        'urdu': 'Indo-Iranian',
        'persian': 'Indo-Iranian',
        'farsi': 'Indo-Iranian',
        'bengali': 'Indo-Iranian',
        'punjabi': 'Indo-Iranian',
        
        # Dravidian languages
        'tamil': 'Dravidian',
        'telugu': 'Dravidian',
        'malayalam': 'Dravidian',
        'kannada': 'Dravidian',
        
        # Turkic languages
        'turkish': 'Turkic',
        'uzbek': 'Turkic',
        'kazakh': 'Turkic',
        
        # Other
        'greek': 'Hellenic',
        'armenian': 'Indo-European (Armenian)',
        'albanian': 'Indo-European (Albanian)',
        'basque': 'Language isolate',
    }
    
    if pd.isna(lang_str):
        return 'Unknown'
    
    # Clean the string and extract language names
    lang_str_clean = str(lang_str).replace("'", "").replace('"', '').lower()
    languages = [lang.strip() for lang in lang_str_clean.split(',')]
    
    # Filter out English
    non_english_languages = [lang for lang in languages if lang != 'english']
    
    # If only English, return special category
    if len(non_english_languages) == 0:
        return 'English only'
    
    # Get families for each non-English language, preserving order
    families = []
    for lang in non_english_languages:
        family = language_families.get(lang, 'Other')
        if family not in families:  # Preserve order, avoid duplicates
            families.append(family)
    
    # Return the first non-English language family found
    if len(families) == 0:
        return 'Unknown'
    else:
        return families[0]

# Create binary language columns
def check_english_only(lang_str):
    """Return 1 if only English, 0 otherwise"""
    if pd.isna(lang_str):
        return 0
    has_english = 'English' in lang_str or 'english' in lang_str.lower()
    is_only_english = has_english and ',' not in lang_str and lang_str.strip().lower() in ['english', "'english'"]
    return 1 if is_only_english else 0

def check_multilingual(lang_str):
    """Return 1 if English AND something else, 0 otherwise"""
    if pd.isna(lang_str):
        return 0
    has_english = 'English' in lang_str or 'english' in lang_str.lower()
    has_comma = ',' in lang_str
    return 1 if (has_english and has_comma) else 0

def check_not_english(lang_str):
    """Return 1 if no English at all, 0 otherwise"""
    if pd.isna(lang_str):
        return 1
    has_english = 'English' in lang_str or 'english' in lang_str.lower()
    return 0 if has_english else 1


def read_transcription_data(MODEL_FAMILY, verbose=True, allowed_models=None):
    # Load all TSV files from the selected model family
    # Use absolute paths relative to the speech_to_text directory
    if MODEL_FAMILY == 'all':
        tsv_files = glob.glob(str(SPEECH_TO_TEXT_DIR / "transcriptions" / "*" / "*.tsv"))
    else:
        tsv_files = glob.glob(str(SPEECH_TO_TEXT_DIR / "transcriptions" / MODEL_FAMILY / "*.tsv"))

    if len(tsv_files) == 0:
        raise ValueError(f"No TSV files found for MODEL_FAMILY='{MODEL_FAMILY}'")
    elif len(tsv_files) == 1:
        data = pd.read_csv(tsv_files[0], sep='\t')
    else:
        data = pd.concat([pd.read_csv(f, sep='\t') for f in tsv_files], ignore_index=True)

    if verbose:
        print(f"Loaded {len(data)} rows from {len(tsv_files)} file(s) ({MODEL_FAMILY})")

    data['transcription_og'] = data['transcription']
    data['prompt'] = data['prompt'].fillna("No prompt")
    data = data.drop_duplicates()

    # Load all demographic data files from the demographic_data folder
    demographic_folder = SPEECH_TO_TEXT_DIR / "demographic_data"
    demo_files = glob.glob(str(demographic_folder / "*.csv"))
    
    # Load and concatenate all demographic files
    demo_dfs = []
    for demo_file in demo_files:
        df = pd.read_csv(demo_file)[['Status', 'Participant id', 'Primary language', 'Age', 'Sex', 'Language']]
        df = df[df['Status'] == 'APPROVED']
        demo_dfs.append(df)
        if verbose:
            print(f"Loaded {len(df)} approved records from {Path(demo_file).name}")
    
    if len(demo_dfs) == 0:
        raise ValueError("No demographic CSV files found")
    elif len(demo_dfs) == 1:
        demo = demo_dfs[0]
    else:
        demo = pd.concat(demo_dfs, ignore_index=True)
    if verbose:
        print(f"Total approved demographic records: {len(demo)}")

    demo['Primary language'] = demo['Primary language'].apply(process_primary_language)
    data = data.set_index("participant_id").join(demo.set_index("Participant id"), how='left').reset_index()
    data.columns = ['participant_id', 'index', 'model', 'prompt', 'original_text', 'transcription',
       'transcription_og', 'Status', 'Primary language', 'Age', 'Sex',
       'Language']
    
    data = data[data['Status'] == 'APPROVED']
    data = data.drop_duplicates()


    data['english_only'] = data['Primary language'].apply(check_english_only)
    data['multilingual'] = data['Primary language'].apply(check_multilingual)
    data['not_english'] = data['Primary language'].apply(check_not_english)

    # Answer contained in the original instruction that was given to the users e.g., "Recording yourself saying the following text ONLY ONCE: "I’m at LAURA."
    data['answer'] = data['original_text'].apply(lambda x: str(x).split(":")[-1].replace('"', "").lower())
    data['answer'] = data['answer'].apply(lambda x: normalize_text(x))
    data['transcription'] = data['transcription'].apply(lambda x: normalize_text(x))


    data['levenshtein_distance'] = data.apply(lambda row: levenshtein_distance(row['answer'], row['transcription']), axis=1)
    data['is_correct'] = data['levenshtein_distance'] == 0

    data['Age'] = data['Age'].astype(int)
    data['age_decade'] = data['Age'] // 10

    #prefer not to say grouped as 1
    data['Sex'] = data['Sex'].apply(lambda x: 0 if x == 'Male' else 1)

    return data

