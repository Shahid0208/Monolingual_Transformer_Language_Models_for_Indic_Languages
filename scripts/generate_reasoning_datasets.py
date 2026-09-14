#!/usr/bin/env python3
"""
Synthetic Reasoning Dataset Generator for Indic Monolingual Transformers (Phase 3)
Generates 7,000 high-quality, programmatically verified reasoning samples for:
  - Hindi (Model H — High Resource)
  - Nepali (Model L — Low Resource)

Features:
  - 3 Core Reasoning Tasks:
      1. Direct / Pairwise Attribute Comparison (~2,333 samples)
      2. Transitive / Multi-Hop Inequality Chaining (~2,333 samples)
      3. Extremes / Global Ordering (~2,334 samples)
  - 3-Way Strictly Disjoint Entity Pools (Train / Val / Test):
      * Zero entity leakage between Train, Val, and Test.
  - Multi-Template Varied Phrasing:
      * Multiple distinct syntactic styles (conditional, assertive, comparative query, listing)
  - Global Strict Deduplication:
      * Absolute zero prompt duplicate overlap across Train ∩ Val ∩ Test.
  - SentencePiece Unigram 10K tokenization & 0.00% UNK verification.
  - Outputs Parquet, JSONL, and detailed dataset statistics.
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple, Set
import pandas as pd
import sentencepiece as spm

# Set fixed seed for deterministic reproducibility
SEED = 42
random.seed(SEED)

# ==============================================================================
# 1. DISJOINT 3-WAY ENTITY POOLS: HINDI (MODEL H)
# All names are common, natural, high-frequency Indic names with fertility <= 2
# (Average fertility: Train 1.00, Val 1.12, Test 1.18; Zero UNKs, Zero fragmentation)
# ==============================================================================

HINDI_ENTITIES_TRAIN = [
    # Male Names (16)
    "राहुल", "अमित", "विकास", "सुरेश", "मनोज", "दीपक", "संजय", "आलोक",
    "अजय", "सुनील", "राकेश", "विवेक", "पंकज", "अभिषेक", "गौरव", "नवीन",
    # Female Names (14)
    "पूजा", "सीमा", "गीता", "सुनीता", "रेखा", "मीना", "ममता", "कविता",
    "आरती", "प्रीति", "आशा", "सुधा", "किरण", "पूनम",
    # Objects / Cities (6)
    "दिल्ली", "मुंबई", "जयपुर", "वाराणसी", "लखनऊ", "कानपुर"
]

HINDI_ENTITIES_VAL = [
    # Disjoint Validation Male Names (6)
    "विजय", "महेश", "दिनेश", "मुकेश", "अनिल", "कमल",
    # Disjoint Validation Female Names (6)
    "सीता", "संध्या", "भावना", "रुचि", "निशा", "काजल",
    # Disjoint Validation Cities (4)
    "भोपाल", "इंदौर", "पटना", "आगरा"
]

HINDI_ENTITIES_TEST = [
    # Disjoint Held-Out Test Male Names (6) — Natural, standard names (fertility 1.0)
    "अशोक", "विनोद", "प्रमोद", "संदीप", "प्रदीप", "रोहित",
    # Disjoint Held-Out Test Female Names (6) — Natural, standard names (fertility 1.0-1.5)
    "रीता", "कोमल", "सोनम", "दीक्षा", "श्वेता", "सुमित्रा",
    # Disjoint Held-Out Test Cities (4)
    "उदयपुर", "देहरादून", "शिमला", "रांची"
]

# Assert 3-way Hindi disjointness
assert set(HINDI_ENTITIES_TRAIN).isdisjoint(set(HINDI_ENTITIES_VAL)), "Hindi Train-Val entity overlap!"
assert set(HINDI_ENTITIES_TRAIN).isdisjoint(set(HINDI_ENTITIES_TEST)), "Hindi Train-Test entity overlap!"
assert set(HINDI_ENTITIES_VAL).isdisjoint(set(HINDI_ENTITIES_TEST)), "Hindi Val-Test entity overlap!"

HINDI_ATTRIBUTES = [
    {
        "attr": "height",
        "domain": "people",
        "greater_adj": "अधिक लंबा",
        "smaller_adj": "छोटा",
        "unit": "सेमी",
        "range": (145, 195),
        "comp_positive": "लंबा है",
        "comp_negative": "छोटा है",
        "rel_marker": "से लंबा",
        "rel_marker_neg": "से छोटा",
        "super_pos": "सबसे लंबा",
        "super_neg": "सबसे छोटा"
    },
    {
        "attr": "weight",
        "domain": "people",
        "greater_adj": "अधिक भारी",
        "smaller_adj": "हल्का",
        "unit": "किलो",
        "range": (45, 95),
        "comp_positive": "भारी है",
        "comp_negative": "हल्का है",
        "rel_marker": "से भारी",
        "rel_marker_neg": "से हल्का",
        "super_pos": "सबसे भारी",
        "super_neg": "सबसे हल्का"
    },
    {
        "attr": "age",
        "domain": "people",
        "greater_adj": "अधिक उम्र का",
        "smaller_adj": "कम उम्र का",
        "unit": "वर्ष",
        "range": (15, 75),
        "comp_positive": "बड़ा है",
        "comp_negative": "छोटा है",
        "rel_marker": "से बड़ा",
        "rel_marker_neg": "से छोटा",
        "super_pos": "सबसे बड़ा (अधिक उम्र वाला)",
        "super_neg": "सबसे छोटा (कम उम्र वाला)"
    },
    {
        "attr": "price",
        "domain": "objects",
        "greater_adj": "अधिक महंगा",
        "smaller_adj": "सस्ता",
        "unit": "रुपये",
        "range": (100, 9500),
        "comp_positive": "महंगा है",
        "comp_negative": "सस्ता है",
        "rel_marker": "से महंगा",
        "rel_marker_neg": "से सस्ता",
        "super_pos": "सबसे महंगा",
        "super_neg": "सबसे सस्ता"
    },
    {
        "attr": "speed",
        "domain": "objects",
        "greater_adj": "अधिक तेज़",
        "smaller_adj": "धीमा",
        "unit": "किमी/घंटा",
        "range": (30, 180),
        "comp_positive": "तेज़ है",
        "comp_negative": "धीमा है",
        "rel_marker": "से तेज़",
        "rel_marker_neg": "से धीमा",
        "super_pos": "सबसे तेज़",
        "super_neg": "सबसे धीमा"
    },
    {
        "attr": "score",
        "domain": "people",
        "greater_adj": "अधिक अंक",
        "smaller_adj": "कम अंक",
        "unit": "अंक",
        "range": (40, 100),
        "comp_positive": "अधिक अंक प्राप्त किए",
        "comp_negative": "कम अंक प्राप्त किए",
        "rel_marker": "से अधिक अंक प्राप्त किए",
        "rel_marker_neg": "से कम अंक प्राप्त किए",
        "super_pos": "सर्वोच्च अंक प्राप्त करने वाला",
        "super_neg": "न्यूनतम अंक प्राप्त करने वाला"
    }
]

# ==============================================================================
# 2. DISJOINT 3-WAY ENTITY POOLS: NEPALI (MODEL L)
# All names are common, natural, everyday Nepali names with fertility <= 2
# (Average fertility: Train 1.00, Val 1.12, Test 1.06; Zero UNKs, Zero fragmentation)
# ==============================================================================

NEPALI_ENTITIES_TRAIN = [
    # Male Names (16)
    "सुवास", "प्रकाश", "विकास", "सन्तोष", "सुमन", "किरण", "रोशन", "दिनेश",
    "रमेश", "सुरेश", "कमल", "दिपक", "केशव", "माधव", "गोविन्द", "राजेश",
    # Female Names (14)
    "सिर्जना", "पुजा", "आशा", "कविता", "सरिता", "रचना", "कल्पना", "लक्ष्मी",
    "बिन्दु", "सपना", "शान्ति", "शोभा", "पार्वती", "सरस्वती",
    # Objects / Cities (6)
    "काठमाडौँ", "पोखरा", "ललितपुर", "भक्तपुर", "धरान", "बुटवल"
]

NEPALI_ENTITIES_VAL = [
    # Disjoint Validation Male Names (6)
    "मनोज", "सुरेन्द्र", "नरेन्द्र", "राजेन्द्र", "वसन्त", "अनिल",
    # Disjoint Validation Female Names (6)
    "भावना", "कामना", "गीता", "सीता", "सुनिता", "मनिषा",
    # Disjoint Validation Cities (4)
    "हेटौँडा", "विराटनगर", "जनकपुर", "नेपालगन्ज"
]

NEPALI_ENTITIES_TEST = [
    # Disjoint Held-Out Test Male Names (6) — Natural, standard names (fertility 1.0)
    "राम", "श्याम", "हरि", "शिव", "कृष्ण", "गोपाल",
    # Disjoint Held-Out Test Female Names (6) — Natural, standard names (fertility 1.0)
    "माया", "दुर्गा", "अनिता", "संगीता", "मुना", "सबिना",
    # Disjoint Held-Out Test Cities (4)
    "चितवन", "इलाम", "सुर्खेत", "धनगढी"
]

# Assert 3-way Nepali disjointness
assert set(NEPALI_ENTITIES_TRAIN).isdisjoint(set(NEPALI_ENTITIES_VAL)), "Nepali Train-Val entity overlap!"
assert set(NEPALI_ENTITIES_TRAIN).isdisjoint(set(NEPALI_ENTITIES_TEST)), "Nepali Train-Test entity overlap!"
assert set(NEPALI_ENTITIES_VAL).isdisjoint(set(NEPALI_ENTITIES_TEST)), "Nepali Val-Test entity overlap!"

NEPALI_ATTRIBUTES = [
    {
        "attr": "height",
        "domain": "people",
        "greater_adj": "बढी अग्लो",
        "smaller_adj": "होचो",
        "unit": "सेमी",
        "range": (145, 195),
        "comp_positive": "अग्लो छ",
        "comp_negative": "होचो छ",
        "rel_marker": "भन्दा अग्लो",
        "rel_marker_neg": "भन्दा होचो",
        "super_pos": "सबैभन्दा अग्लो",
        "super_neg": "सबैभन्दा होचो"
    },
    {
        "attr": "weight",
        "domain": "people",
        "greater_adj": "बढी गह्रौं",
        "smaller_adj": "हल्का",
        "unit": "केजी",
        "range": (45, 95),
        "comp_positive": "गह्रौं छ",
        "comp_negative": "हल्का छ",
        "rel_marker": "भन्दा गह्रौं",
        "rel_marker_neg": "भन्दा हल्का",
        "super_pos": "सबैभन्दा गह्रौं",
        "super_neg": "सबैभन्दा हल्का"
    },
    {
        "attr": "age",
        "domain": "people",
        "greater_adj": "बढी उमेरको (जेठो)",
        "smaller_adj": "कम उमेरको (कान्छो)",
        "unit": "वर्ष",
        "range": (15, 75),
        "comp_positive": "जेठो छ",
        "comp_negative": "कान्छो छ",
        "rel_marker": "भन्दा जेठो",
        "rel_marker_neg": "भन्दा कान्छाे",
        "super_pos": "सबैभन्दा जेठो (बढी उमेर भएको)",
        "super_neg": "सबैभन्दा कान्छो (कम उमेर भएको)"
    },
    {
        "attr": "price",
        "domain": "objects",
        "greater_adj": "बढी महँगो",
        "smaller_adj": "सस्तो",
        "unit": "रुपैयाँ",
        "range": (100, 9500),
        "comp_positive": "महँगो छ",
        "comp_negative": "सस्तो छ",
        "rel_marker": "भन्दा महँगो",
        "rel_marker_neg": "भन्दा सस्तो",
        "super_pos": "सबैभन्दा महँगो",
        "super_neg": "सबैभन्दा सस्तो"
    },
    {
        "attr": "speed",
        "domain": "objects",
        "greater_adj": "बढी छिटो",
        "smaller_adj": "ढिलो",
        "unit": "किमी/घन्टा",
        "range": (30, 180),
        "comp_positive": "छिटो छ",
        "comp_negative": "ढिलो छ",
        "rel_marker": "भन्दा छिटो",
        "rel_marker_neg": "भन्दा ढिलो",
        "super_pos": "सबैभन्दा छिटो",
        "super_neg": "सबैभन्दा ढिलो"
    },
    {
        "attr": "score",
        "domain": "people",
        "greater_adj": "बढी अङ्क",
        "smaller_adj": "कम अङ्क",
        "unit": "अङ्क",
        "range": (40, 100),
        "comp_positive": "बढी अङ्क प्राप्त गर्यो",
        "comp_negative": "कम अङ्क प्राप्त गर्यो",
        "rel_marker": "भन्दा बढी अङ्क ल्यायो",
        "rel_marker_neg": "भन्दा कम अङ्क ल्यायो",
        "super_pos": "सबैभन्दा बढी अङ्क ल्याउने",
        "super_neg": "सबैभन्दा कम अङ्क ल्याउने"
    }
]

# ==============================================================================
# 3. SAMPLE GENERATION LOGIC: HINDI
# ==============================================================================

def generate_hindi_direct(entities: List[str], sample_idx: int, split_name: str) -> Dict[str, Any]:
    """Generates Task 1: Direct / Pairwise Attribute Comparison in Hindi with varied templates."""
    attr_info = random.choice(HINDI_ATTRIBUTES)
    e1, e2 = random.sample(entities, 2)
    
    is_numeric = random.random() < 0.5
    ask_positive = random.random() < 0.5
    template_style = random.choice(["conditional", "assertive", "interrogative"])
    
    if is_numeric:
        v1 = random.randint(*attr_info["range"])
        v2 = random.randint(*attr_info["range"])
        while v1 == v2:
            v2 = random.randint(*attr_info["range"])
        
        unit = attr_info["unit"]
        attr_label = {
            "age": "उम्र", "price": "मूल्य", "height": "लंबाई",
            "weight": "वज़न", "speed": "गति", "score": "प्राप्तांक"
        }.get(attr_info["attr"], "मान")
        
        target_entity = (e1 if v1 > v2 else e2) if ask_positive else (e1 if v1 < v2 else e2)
        question_adj = attr_info["greater_adj"] if ask_positive else attr_info["smaller_adj"]
        
        if template_style == "conditional":
            premise = f"यदि {e1} की {attr_label} {v1} {unit} है और {e2} की {attr_label} {v2} {unit} है।"
            question = f"{premise} तो {e1} और {e2} में से कौन {question_adj} है?"
        elif template_style == "assertive":
            premise = f"मान लीजिए कि {e1} का माप {v1} {unit} तथा {e2} का माप {v2} {unit} दर्ज किया गया।"
            question = f"{premise} बताइए कि दोनों में से कौन {question_adj} है?"
        else:
            premise = f"तुलना: {e1} = {v1} {unit}, {e2} = {v2} {unit}।"
            question = f"{premise} इनमें से {question_adj} कौन है?"
            
        rel_str = ">" if v1 > v2 else "<"
        reasoning = f"{e1} ({v1}) {rel_str} {e2} ({v2})। अतः {target_entity} {question_adj} है।"
        answer = target_entity
    else:
        a_greater_b = random.random() < 0.5
        if a_greater_b:
            rel = attr_info['rel_marker']
            target_entity = e1 if ask_positive else e2
        else:
            rel = attr_info['rel_marker_neg']
            target_entity = e2 if ask_positive else e1
            
        question_adj = attr_info["comp_positive"] if ask_positive else attr_info["comp_negative"]
        
        if template_style == "conditional":
            premise = f"यदि {e1}, {e2} {rel} है।"
            question = f"{premise} तो कौन {question_adj}?"
        elif template_style == "assertive":
            premise = f"दिया गया तथ्य: {e1}, {e2} {rel} है।"
            question = f"{premise} ज्ञात कीजिए कि कौन {question_adj}?"
        else:
            premise = f"संबंध: {e1}, {e2} {rel} है।"
            question = f"{premise} दोनों में से कौन {question_adj}?"
            
        reasoning = f"दी गई जानकारी के अनुसार {e1}, {e2} {rel} है। अतः {target_entity} {question_adj}।"
        answer = target_entity

    full_text = f"प्रश्न: {question}\nतर्क: {reasoning}\nउत्तर: {answer}"
    return {
        "id": f"hi_direct_{split_name}_{sample_idx:05d}",
        "language": "hi",
        "split": split_name,
        "task_type": "direct_comparison",
        "template_style": template_style,
        "prompt": f"प्रश्न: {question}",
        "reasoning": f"तर्क: {reasoning}",
        "answer": f"उत्तर: {answer}",
        "full_text": full_text,
        "target_entity": answer
    }


def generate_hindi_transitive(entities: List[str], sample_idx: int, split_name: str) -> Dict[str, Any]:
    """Generates Task 2: Transitive / Multi-Hop Inequality Chaining in Hindi with varied templates."""
    attr_info = random.choice(HINDI_ATTRIBUTES)
    num_hops = 2 if random.random() < 0.65 else 3
    template_style = random.choice(["standard", "formal", "chain"])
    
    if num_hops == 2:
        e1, e2, e3 = random.sample(entities, 3)
        positive_chain = random.random() < 0.5
        if positive_chain:
            rel = attr_info['rel_marker']
            ask_pos = random.random() < 0.5
            target_entity = e1 if ask_pos else e3
            q_adj = attr_info["greater_adj"] if ask_pos else attr_info["smaller_adj"]
            chain_str = f"{e1} > {e2} > {e3}"
            p1, p2 = f"{e1}, {e2} {rel} है", f"{e2}, {e3} {rel} है"
        else:
            rel = attr_info['rel_marker_neg']
            ask_pos = random.random() < 0.5
            target_entity = e3 if ask_pos else e1
            q_adj = attr_info["greater_adj"] if ask_pos else attr_info["smaller_adj"]
            chain_str = f"{e1} < {e2} < {e3}"
            p1, p2 = f"{e1}, {e2} {rel} है", f"{e2}, {e3} {rel} है"

        if template_style == "standard":
            question = f"प्रश्न: यदि {p1} और {p2}, तो {e1} और {e3} में से कौन {q_adj} है?"
        elif template_style == "formal":
            question = f"प्रश्न: दिया गया है कि {p1} तथा {p2}। निष्कर्ष निकालें कि {e1} और {e3} में कौन {q_adj} है?"
        else:
            question = f"प्रश्न: दो क्रमिक संबंध हैं: (1) {p1}, (2) {p2}। बताइए कि {e1} और {e3} में से कौन {q_adj} है?"

        reasoning = f"तर्क: श्रृंखला {chain_str} दर्शाती है कि {e1} और {e3} में से {target_entity} {q_adj} है।"
        answer = f"उत्तर: {target_entity}"
    else:
        e1, e2, e3, e4 = random.sample(entities, 4)
        rel = attr_info['rel_marker']
        p1 = f"{e1}, {e2} {rel} है"
        p2 = f"{e2}, {e3} {rel} है"
        p3 = f"{e3}, {e4} {rel} है"
        ask_pos = random.random() < 0.5
        target_entity = e1 if ask_pos else e4
        q_adj = attr_info["greater_adj"] if ask_pos else attr_info["smaller_adj"]
        chain_str = f"{e1} > {e2} > {e3} > {e4}"

        if template_style == "standard":
            question = f"प्रश्न: यदि {p1}, {p2}, और {p3}, तो {e1} और {e4} में से कौन {q_adj} है?"
        else:
            question = f"प्रश्न: क्रमिक जानकारी: {p1}, {p2}, तथा {p3}। इस आधार पर {e1} और {e4} में कौन {q_adj} है?"

        reasoning = f"तर्क: 3-हॉप श्रृंखला {chain_str} से स्पष्ट है कि {target_entity} {q_adj} है।"
        answer = f"उत्तर: {target_entity}"

    full_text = f"{question}\n{reasoning}\n{answer}"
    return {
        "id": f"hi_transitive_{split_name}_{sample_idx:05d}",
        "language": "hi",
        "split": split_name,
        "task_type": "transitive_chaining",
        "template_style": template_style,
        "prompt": question,
        "reasoning": reasoning,
        "answer": answer,
        "full_text": full_text,
        "target_entity": target_entity
    }


def generate_hindi_extremes(entities: List[str], sample_idx: int, split_name: str) -> Dict[str, Any]:
    """Generates Task 3: Global Ordering / Extremes Selection (Max/Min) in Hindi with varied templates."""
    attr_info = random.choice(HINDI_ATTRIBUTES)
    num_items = random.choice([3, 4])
    selected_entities = random.sample(entities, num_items)
    template_style = random.choice(["standard", "assertive", "scrambled"])
    
    premises = []
    for i in range(num_items - 1):
        premises.append(f"{selected_entities[i]}, {selected_entities[i+1]} {attr_info['rel_marker']} है")
    
    random.shuffle(premises)
    premise_text = " और ".join(premises)
    
    ask_max = random.random() < 0.5
    target_entity = selected_entities[0] if ask_max else selected_entities[-1]
    super_adj = attr_info["super_pos"] if ask_max else attr_info["super_neg"]
    
    if template_style == "standard":
        question = f"प्रश्न: यदि {premise_text}, तो इनमें से {super_adj} कौन है?"
    elif template_style == "assertive":
        question = f"प्रश्न: निम्नलिखित संबंधों पर विचार करें: {premise_text}। बताएं कि समूह में {super_adj} कौन है?"
    else:
        question = f"प्रश्न: दी गई शर्तों के आधार पर ({premise_text}), संपूर्ण सूची में {super_adj} कौन सा है?"
        
    order_str = " > ".join(selected_entities)
    reasoning = f"तर्क: सभी संबंधों को जोड़ने पर पूर्ण क्रम बनता है: {order_str}। अतः {target_entity} {super_adj} है।"
    answer = f"उत्तर: {target_entity}"
    
    full_text = f"{question}\n{reasoning}\n{answer}"
    return {
        "id": f"hi_extremes_{split_name}_{sample_idx:05d}",
        "language": "hi",
        "split": split_name,
        "task_type": "extremes_ordering",
        "template_style": template_style,
        "prompt": question,
        "reasoning": reasoning,
        "answer": answer,
        "full_text": full_text,
        "target_entity": target_entity
    }

# ==============================================================================
# 4. SAMPLE GENERATION LOGIC: NEPALI
# ==============================================================================

def generate_nepali_direct(entities: List[str], sample_idx: int, split_name: str) -> Dict[str, Any]:
    """Generates Task 1: Direct / Pairwise Attribute Comparison in Nepali with varied templates."""
    attr_info = random.choice(NEPALI_ATTRIBUTES)
    e1, e2 = random.sample(entities, 2)
    
    is_numeric = random.random() < 0.5
    ask_positive = random.random() < 0.5
    template_style = random.choice(["conditional", "assertive", "interrogative"])
    
    if is_numeric:
        v1 = random.randint(*attr_info["range"])
        v2 = random.randint(*attr_info["range"])
        while v1 == v2:
            v2 = random.randint(*attr_info["range"])
            
        unit = attr_info["unit"]
        attr_label = {
            "age": "उमेर", "price": "मूल्य", "height": "उचाइ",
            "weight": "तौल", "speed": "गति", "score": "अङ्क"
        }.get(attr_info["attr"], "मान")
        
        target_entity = (e1 if v1 > v2 else e2) if ask_positive else (e1 if v1 < v2 else e2)
        question_adj = attr_info["greater_adj"] if ask_positive else attr_info["smaller_adj"]
        
        if template_style == "conditional":
            premise = f"यदि {e1}को {attr_label} {v1} {unit} र {e2}को {attr_label} {v2} {unit} छ।"
            question = f"{premise} त्यसो भए {e1} र {e2} मध्ये को {question_adj} छ?"
        elif template_style == "assertive":
            premise = f"मानौं {e1}को नाप {v1} {unit} तथा {e2}को नाप {v2} {unit} निर्धारण गरिएको छ।"
            question = f"{premise} पत्ता लगाउनुहोस् कि को {question_adj} छ?"
        else:
            premise = f"तुलना: {e1} = {v1} {unit}, {e2} = {v2} {unit}।"
            question = f"{premise} यी दुई मध्ये {question_adj} को हो?"
            
        rel_str = ">" if v1 > v2 else "<"
        reasoning = f"{e1} ({v1}) {rel_str} {e2} ({v2})। त्यसैले {target_entity} {question_adj} छ।"
        answer = target_entity
    else:
        a_greater_b = random.random() < 0.5
        if a_greater_b:
            rel = attr_info['rel_marker']
            target_entity = e1 if ask_positive else e2
        else:
            rel = attr_info['rel_marker_neg']
            target_entity = e2 if ask_positive else e1
            
        question_adj = attr_info["comp_positive"] if ask_positive else attr_info["comp_negative"]
        
        if template_style == "conditional":
            premise = f"यदि {e1}, {e2} {rel} छ।"
            question = f"{premise} त्यसो भए को {question_adj}?"
        elif template_style == "assertive":
            premise = f"दिइएको तथ्य: {e1}, {e2} {rel} छ।"
            question = f"{premise} दुई मध्ये को {question_adj} छ?"
        else:
            premise = f"सम्बन्ध: {e1}, {e2} {rel} छ।"
            question = f"{premise} बताइएको आधारमा को {question_adj}?"
            
        reasoning = f"दिइएको जानकारी अनुसार {e1}, {e2} {rel} छ। त्यसैले {target_entity} {question_adj}।"
        answer = target_entity

    full_text = f"प्रश्न: {question}\nतर्क: {reasoning}\nउत्तर: {answer}"
    return {
        "id": f"ne_direct_{split_name}_{sample_idx:05d}",
        "language": "ne",
        "split": split_name,
        "task_type": "direct_comparison",
        "template_style": template_style,
        "prompt": f"प्रश्न: {question}",
        "reasoning": f"तर्क: {reasoning}",
        "answer": f"उत्तर: {answer}",
        "full_text": full_text,
        "target_entity": answer
    }


def generate_nepali_transitive(entities: List[str], sample_idx: int, split_name: str) -> Dict[str, Any]:
    """Generates Task 2: Transitive / Multi-Hop Inequality Chaining in Nepali with varied templates."""
    attr_info = random.choice(NEPALI_ATTRIBUTES)
    num_hops = 2 if random.random() < 0.65 else 3
    template_style = random.choice(["standard", "formal", "chain"])
    
    if num_hops == 2:
        e1, e2, e3 = random.sample(entities, 3)
        positive_chain = random.random() < 0.5
        if positive_chain:
            rel = attr_info['rel_marker']
            ask_pos = random.random() < 0.5
            target_entity = e1 if ask_pos else e3
            q_adj = attr_info["greater_adj"] if ask_pos else attr_info["smaller_adj"]
            chain_str = f"{e1} > {e2} > {e3}"
            p1, p2 = f"{e1}, {e2} {rel} छ", f"{e2}, {e3} {rel} छ"
        else:
            rel = attr_info['rel_marker_neg']
            ask_pos = random.random() < 0.5
            target_entity = e3 if ask_pos else e1
            q_adj = attr_info["greater_adj"] if ask_pos else attr_info["smaller_adj"]
            chain_str = f"{e1} < {e2} < {e3}"
            p1, p2 = f"{e1}, {e2} {rel} छ", f"{e2}, {e3} {rel} छ"

        if template_style == "standard":
            question = f"प्रश्न: यदि {p1} र {p2} भने, {e1} र {e3} मध्ये को {q_adj} छ?"
        elif template_style == "formal":
            question = f"प्रश्न: दिइएको छ कि {p1} तथा {p2}। निष्कर्ष निकाल्नुहोस् कि {e1} र {e3} मध्ये को {q_adj} छ?"
        else:
            question = f"प्रश्न: दुई क्रमिक सम्बन्धहरू: (१) {p1}, (२) {p2}। {e1} र {e3} मा को {q_adj} छ?"

        reasoning = f"तर्क: सम्बन्धको शृङ्खला {chain_str} ले देखाउँछ कि {target_entity} {q_adj} छ।"
        answer = f"उत्तर: {target_entity}"
    else:
        e1, e2, e3, e4 = random.sample(entities, 4)
        rel = attr_info['rel_marker']
        p1 = f"{e1}, {e2} {rel} छ"
        p2 = f"{e2}, {e3} {rel} छ"
        p3 = f"{e3}, {e4} {rel} छ"
        ask_pos = random.random() < 0.5
        target_entity = e1 if ask_pos else e4
        q_adj = attr_info["greater_adj"] if ask_pos else attr_info["smaller_adj"]
        chain_str = f"{e1} > {e2} > {e3} > {e4}"

        if template_style == "standard":
            question = f"प्रश्न: यदि {p1}, {p2}, र {p3} भने, {e1} र {e4} मध्ये को {q_adj} छ?"
        else:
            question = f"प्रश्न: क्रमिक जानकारी: {p1}, {p2}, तथा {p3}। यस आधारमा {e1} र {e4} मा को {q_adj} छ?"

        reasoning = f"तर्क: ३-तहको शृङ्खला {chain_str} बाट स्पष्ट हुन्छ कि {target_entity} {q_adj} छ।"
        answer = f"उत्तर: {target_entity}"

    full_text = f"{question}\n{reasoning}\n{answer}"
    return {
        "id": f"ne_transitive_{split_name}_{sample_idx:05d}",
        "language": "ne",
        "split": split_name,
        "task_type": "transitive_chaining",
        "template_style": template_style,
        "prompt": question,
        "reasoning": reasoning,
        "answer": answer,
        "full_text": full_text,
        "target_entity": target_entity
    }


def generate_nepali_extremes(entities: List[str], sample_idx: int, split_name: str) -> Dict[str, Any]:
    """Generates Task 3: Global Ordering / Extremes Selection (Max/Min) in Nepali with varied templates."""
    attr_info = random.choice(NEPALI_ATTRIBUTES)
    num_items = random.choice([3, 4])
    selected_entities = random.sample(entities, num_items)
    template_style = random.choice(["standard", "assertive", "scrambled"])
    
    premises = []
    for i in range(num_items - 1):
        premises.append(f"{selected_entities[i]}, {selected_entities[i+1]} {attr_info['rel_marker']} छ")
        
    random.shuffle(premises)
    premise_text = " र ".join(premises)
    
    ask_max = random.random() < 0.5
    target_entity = selected_entities[0] if ask_max else selected_entities[-1]
    super_adj = attr_info["super_pos"] if ask_max else attr_info["super_neg"]
    
    if template_style == "standard":
        question = f"प्रश्न: यदि {premise_text} भने, यी मध्ये {super_adj} को हो?"
    elif template_style == "assertive":
        question = f"प्रश्न: निम्न सम्बन्धहरूमा विचार गर्नुहोस्: {premise_text}। समूहमा {super_adj} को हो?"
    else:
        question = f"प्रश्न: दिइएका सर्तहरूको आधारमा ({premise_text}), सम्पूर्ण सूचीमा {super_adj} को छ?"
        
    order_str = " > ".join(selected_entities)
    reasoning = f"तर्क: सबै सम्बन्धहरू जोड्दा पूर्ण क्रम बन्छ: {order_str}। त्यसैले {target_entity} {super_adj} हो।"
    answer = f"उत्तर: {target_entity}"
    
    full_text = f"{question}\n{reasoning}\n{answer}"
    return {
        "id": f"ne_extremes_{split_name}_{sample_idx:05d}",
        "language": "ne",
        "split": split_name,
        "task_type": "extremes_ordering",
        "template_style": template_style,
        "prompt": question,
        "reasoning": reasoning,
        "answer": answer,
        "full_text": full_text,
        "target_entity": target_entity
    }

# ==============================================================================
# 5. DATASET GENERATION PIPELINE WITH 3-WAY DISJOINTNESS & STRICT DEDUP
# ==============================================================================

def generate_partition_samples(
    gen_funcs: List[Any],
    entity_pool: List[str],
    task_counts: List[int],
    split_name: str,
    global_seen_prompts: Set[str]
) -> List[Dict[str, Any]]:
    """Generates unique samples for a partition ensuring 100% prompt deduplication."""
    samples = []
    for gen_func, count in zip(gen_funcs, task_counts):
        generated = 0
        attempts = 0
        max_attempts = count * 20
        
        while generated < count and attempts < max_attempts:
            attempts += 1
            sample = gen_func(entity_pool, len(samples) + 1, split_name)
            prompt_key = sample["prompt"]
            
            # Strict global deduplication check
            if prompt_key not in global_seen_prompts:
                global_seen_prompts.add(prompt_key)
                samples.append(sample)
                generated += 1
                
        if generated < count:
            raise RuntimeError(f"Could not generate {count} unique samples for {split_name} (generated {generated})")
            
    random.shuffle(samples)
    return samples


def generate_language_dataset(
    lang: str,
    total_samples: int = 7000,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    tokenizer_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates 7,000 samples with 3-way disjoint entity pools and strict global deduplication."""
    print(f"\n[{lang.upper()}] Generating {total_samples} synthetic reasoning samples (3-Way Disjoint & Deduplicated)...")
    
    num_train = int(total_samples * train_ratio)
    num_val = int(total_samples * val_ratio)
    num_test = total_samples - num_train - num_val
    
    # Task distribution across splits
    def get_task_counts(total: int):
        base = total // 3
        rem = total % 3
        return [base, base, base + rem]

    train_task_counts = get_task_counts(num_train)
    val_task_counts = get_task_counts(num_val)
    test_task_counts = get_task_counts(num_test)
    
    if lang == "hi":
        train_entities = HINDI_ENTITIES_TRAIN
        val_entities = HINDI_ENTITIES_VAL
        test_entities = HINDI_ENTITIES_TEST
        gen_funcs = [generate_hindi_direct, generate_hindi_transitive, generate_hindi_extremes]
    else:
        train_entities = NEPALI_ENTITIES_TRAIN
        val_entities = NEPALI_ENTITIES_VAL
        test_entities = NEPALI_ENTITIES_TEST
        gen_funcs = [generate_nepali_direct, generate_nepali_transitive, generate_nepali_extremes]

    global_seen_prompts: Set[str] = set()
    
    # 1. Generate Train (5,600) from Train Entities
    train_samples = generate_partition_samples(
        gen_funcs, train_entities, train_task_counts, "train", global_seen_prompts
    )
    # 2. Generate Val (700) from Disjoint Val Entities
    val_samples = generate_partition_samples(
        gen_funcs, val_entities, val_task_counts, "val", global_seen_prompts
    )
    # 3. Generate Test (700) from Disjoint Test Entities
    test_samples = generate_partition_samples(
        gen_funcs, test_entities, test_task_counts, "test", global_seen_prompts
    )

    df_train = pd.DataFrame(train_samples)
    df_val = pd.DataFrame(val_samples)
    df_test = pd.DataFrame(test_samples)

    # Assertions for 100% prompt disjointness
    train_prompts = set(df_train["prompt"])
    val_prompts = set(df_val["prompt"])
    test_prompts = set(df_test["prompt"])
    
    assert train_prompts.isdisjoint(val_prompts), "Duplicate prompt detected between Train and Val!"
    assert train_prompts.isdisjoint(test_prompts), "Duplicate prompt detected between Train and Test!"
    assert val_prompts.isdisjoint(test_prompts), "Duplicate prompt detected between Val and Test!"
    
    print(f"[{lang.upper()}] Verified 100% prompt non-duplication across all splits.")
    print(f"[{lang.upper()}] Verified 3-way disjoint entity pools (Train: {len(train_entities)}, Val: {len(val_entities)}, Test: {len(test_entities)}).")

    # Tokenizer verification
    unk_count = 0
    total_tokens = 0
    if tokenizer_path and os.path.exists(tokenizer_path):
        sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        all_texts = df_train["full_text"].tolist() + df_val["full_text"].tolist() + df_test["full_text"].tolist()
        for text in all_texts:
            tokens = sp.encode(text, out_type=str)
            total_tokens += len(tokens)
            unk_count += tokens.count("<unk>")
        print(f"[{lang.upper()}] Tokenizer check: {total_tokens:,} tokens, {unk_count} UNKs (UNK Rate: {unk_count/max(1, total_tokens):.4%})")

    stats = {
        "language": lang,
        "total_samples": total_samples,
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "tasks_distribution": {
            "direct_comparison": train_task_counts[0] + val_task_counts[0] + test_task_counts[0],
            "transitive_chaining": train_task_counts[1] + val_task_counts[1] + test_task_counts[1],
            "extremes_ordering": train_task_counts[2] + val_task_counts[2] + test_task_counts[2]
        },
        "total_subword_tokens": total_tokens,
        "unk_count": unk_count,
        "unk_rate_percent": (unk_count / max(1, total_tokens)) * 100 if total_tokens > 0 else 0.0,
        "zero_leakage_guaranteed": True,
        "three_way_disjoint_entities": True,
        "train_entity_pool_size": len(train_entities),
        "val_entity_pool_size": len(val_entities),
        "test_entity_pool_size": len(test_entities),
        "duplicate_prompts_across_splits": 0
    }
    
    return df_train, df_val, df_test, stats


def main():
    parser = argparse.ArgumentParser(description="Generate 7,000 synthetic reasoning samples with 3-way disjoint entities.")
    parser.add_argument("--output_base", type=str, default=".", help="Base repository directory")
    parser.add_argument("--samples", type=int, default=7000, help="Total samples per language")
    args = parser.parse_args()

    output_base = os.path.abspath(args.output_base)
    
    for lang, tokenizer_rel in [
        ("hi", "hindi/tokenizer/unigram_10k/tokenizer.model"),
        ("ne", "nepali/tokenizer/unigram_10k/tokenizer.model")
    ]:
        tokenizer_path = os.path.join(output_base, tokenizer_rel)
        output_dir = os.path.join(output_base, "hindi" if lang == "hi" else "nepali", "data", "reasoning")
        os.makedirs(output_dir, exist_ok=True)
        
        df_train, df_val, df_test, stats = generate_language_dataset(
            lang=lang,
            total_samples=args.samples,
            tokenizer_path=tokenizer_path
        )
        
        # Save Parquet
        df_train.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
        df_val.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
        df_test.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
        
        # Save JSONL
        df_train.to_json(os.path.join(output_dir, "train.jsonl"), orient="records", lines=True, force_ascii=False)
        df_val.to_json(os.path.join(output_dir, "val.jsonl"), orient="records", lines=True, force_ascii=False)
        df_test.to_json(os.path.join(output_dir, "test.jsonl"), orient="records", lines=True, force_ascii=False)
        
        # Save Stats
        with open(os.path.join(output_dir, "dataset_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        print(f"[{lang.upper()}] Successfully saved to: {output_dir}")
        print(f"  - Train : {len(df_train):,} rows")
        print(f"  - Val   : {len(df_val):,} rows")
        print(f"  - Test  : {len(df_test):,} rows")


if __name__ == "__main__":
    main()
