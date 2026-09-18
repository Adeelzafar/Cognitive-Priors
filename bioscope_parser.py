"""
BioScope Corpus Parser
======================

Correctly parses the actual BioScope XML format with:
  - Nested <xcope> and <cue> inline annotations
  - Character-to-token label alignment
  - Document structure (sections, document types)

Produces BioScopeInstance objects ready for the epistemic architecture.
"""

import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CueAnnotation:
    cue_type: str           # "negation" or "speculation"
    cue_text: str           # the cue words
    char_start: int         # start position in plain text
    char_end: int           # end position in plain text
    xcope_id: str           # which scope this cue belongs to


@dataclass
class ScopeAnnotation:
    scope_id: str           # xcope id
    scope_type: str         # "negation" or "speculation" (from its cue)
    char_start: int         # start in plain text
    char_end: int           # end in plain text


@dataclass
class BioScopeInstance:
    text: str
    tokens: List[str] = field(default_factory=list)
    cue_labels: List[int] = field(default_factory=list)    # 0=O, 1=B-cue, 2=I-cue
    scope_labels: List[int] = field(default_factory=list)   # 0=out, 1=negated, 2=speculated
    cue_type_labels: List[int] = field(default_factory=list) # 0=none, 1=neg-cue, 2=spec-cue
    doc_type: str = "abstract"
    section_type: str = "unknown"
    section_id: int = 14
    doc_id: str = ""
    sentence_id: str = ""
    sentence_idx: int = 0
    raw_xml: str = ""
    cue_annotations: List[CueAnnotation] = field(default_factory=list)
    scope_annotations: List[ScopeAnnotation] = field(default_factory=list)


# Section type mapping
SECTION_MAP = {
    "title": 0, "abstracttext": 13, "text": 14,
    "sectiontitle": 14, "subsectiontitle": 14,
    "tablelegend": 14, "figurelegend": 14,
    # Clinical document parts
    "impression": 1, "history": 2,
    "findings": 0, "indication": 3,
    "plan": 4, "technique": 5, "comparison": 6,
    "recommendation": 7,
    # Detected from section title content:
    "abstract": 13, "background": 9, "introduction": 9,
    "methods": 10, "materials and methods": 10,
    "results": 11, "discussion": 12,
    "conclusion": 8, "conclusions": 8,
    "results and discussion": 11,
    "unknown": 14,
}


# ---------------------------------------------------------------------------
# Core XML sentence parser
# ---------------------------------------------------------------------------

def parse_annotated_sentence(raw_xml: str) -> Tuple[str, List[CueAnnotation], List[ScopeAnnotation]]:
    """
    Parse a BioScope annotated sentence into plain text + annotations.
    
    Handles nested scopes, multi-word cues, and complex keyword structures.
    
    Strategy: walk through the XML character by character, tracking open
    scopes and cues, building plain text while recording character positions.
    """
    cues = []
    scopes = []
    
    # Track scope stack: (xcope_id, plain_text_start_pos)
    scope_stack = []
    # Track cue state: (cue_type, ref, plain_text_start_pos)
    cue_state = None
    
    plain_text = []
    pos = 0  # position in raw_xml
    plain_pos = 0  # position in plain text
    
    while pos < len(raw_xml):
        # Check for XML tags
        if raw_xml[pos] == '<':
            tag_end = raw_xml.find('>', pos)
            if tag_end == -1:
                # Malformed - treat as text
                plain_text.append('<')
                plain_pos += 1
                pos += 1
                continue
            
            tag_content = raw_xml[pos+1:tag_end]
            full_tag = raw_xml[pos:tag_end+1]
            
            if tag_content.startswith('xcope '):
                # Opening xcope tag
                xcope_id = _extract_attr(tag_content, 'id')
                scope_stack.append((xcope_id, plain_pos))
                
            elif tag_content == '/xcope':
                # Closing xcope tag
                if scope_stack:
                    xcope_id, start_pos = scope_stack.pop()
                    scopes.append(ScopeAnnotation(
                        scope_id=xcope_id,
                        scope_type="",  # will be filled from cue
                        char_start=start_pos,
                        char_end=plain_pos,
                    ))
                    
            elif tag_content.startswith('cue '):
                # Opening cue tag
                cue_type = _extract_attr(tag_content, 'type')
                cue_ref = _extract_attr(tag_content, 'ref')
                cue_state = (cue_type, cue_ref, plain_pos)
                
            elif tag_content == '/cue':
                # Closing cue tag
                if cue_state:
                    cue_type, cue_ref, start_pos = cue_state
                    cue_text = ''.join(plain_text[start_pos:])
                    # Actually we need the text from start_pos to current plain_pos
                    full_plain = ''.join(plain_text)
                    cue_text = full_plain[start_pos:plain_pos]
                    
                    cues.append(CueAnnotation(
                        cue_type=cue_type,
                        cue_text=cue_text,
                        char_start=start_pos,
                        char_end=plain_pos,
                        xcope_id=cue_ref,
                    ))
                    cue_state = None
            
            pos = tag_end + 1
        else:
            plain_text.append(raw_xml[pos])
            plain_pos += 1
            pos += 1
    
    final_text = ''.join(plain_text).strip()
    
    # Adjust positions for leading whitespace
    leading_ws = len(''.join(plain_text)) - len(''.join(plain_text).lstrip())
    if leading_ws > 0:
        for cue in cues:
            cue.char_start -= leading_ws
            cue.char_end -= leading_ws
        for scope in scopes:
            scope.char_start -= leading_ws
            scope.char_end -= leading_ws
    
    # Fill scope types from their cues
    cue_by_ref = {}
    for cue in cues:
        cue_by_ref[cue.xcope_id] = cue.cue_type
    
    for scope in scopes:
        scope.scope_type = cue_by_ref.get(scope.scope_id, "unknown")
    
    return final_text, cues, scopes


def _extract_attr(tag_content: str, attr_name: str) -> str:
    """Extract attribute value from tag content."""
    pattern = f'{attr_name}="([^"]*)"'
    match = re.search(pattern, tag_content)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Token-level label alignment
# ---------------------------------------------------------------------------

def align_labels_to_tokens(
    text: str,
    tokens: List[str],
    cues: List[CueAnnotation],
    scopes: List[ScopeAnnotation],
) -> Tuple[List[int], List[int], List[int]]:
    """
    Convert character-level annotations to token-level labels.
    
    Returns:
        cue_labels: 0=O, 1=B-cue, 2=I-cue
        scope_labels: 0=outside, 1=in-negation-scope, 2=in-speculation-scope
        cue_type_labels: 0=not-cue, 1=negation-cue, 2=speculation-cue
    """
    # Find character spans for each token
    token_spans = []
    search_start = 0
    for token in tokens:
        idx = text.find(token, search_start)
        if idx == -1:
            # Try case-insensitive or partial match
            idx = search_start
        token_spans.append((idx, idx + len(token)))
        search_start = idx + len(token)
    
    n = len(tokens)
    cue_labels = [0] * n
    scope_labels = [0] * n
    cue_type_labels = [0] * n
    
    # Mark scope labels
    for scope in scopes:
        label = 1 if scope.scope_type == "negation" else 2
        for i, (ts, te) in enumerate(token_spans):
            # Token overlaps with scope
            if ts < scope.char_end and te > scope.char_start:
                # If already labeled with different type, prefer the innermost scope
                # (which comes later in our list due to stack-based parsing)
                scope_labels[i] = label
    
    # Mark cue labels
    for cue in cues:
        cue_type_val = 1 if cue.cue_type == "negation" else 2
        first_cue_token = True
        
        for i, (ts, te) in enumerate(token_spans):
            # Token overlaps with cue span
            if ts < cue.char_end and te > cue.char_start:
                if first_cue_token:
                    cue_labels[i] = 1  # B-cue
                    first_cue_token = False
                else:
                    cue_labels[i] = 2  # I-cue
                cue_type_labels[i] = cue_type_val
    
    return cue_labels, scope_labels, cue_type_labels


# ---------------------------------------------------------------------------
# Full corpus parser
# ---------------------------------------------------------------------------

def parse_bioscope_file(
    filepath: str,
    doc_type: str = "abstract",
) -> List[BioScopeInstance]:
    """
    Parse a BioScope XML file into a list of annotated instances.
    
    Args:
        filepath: Path to abstracts.xml or full_papers.xml
        doc_type: "abstract" or "full_paper"
    
    Returns:
        List of BioScopeInstance with token-level labels
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    instances = []
    current_section = "unknown"
    current_doc_id = ""
    sentence_counter = 0
    
    # Process line by line to handle document structure
    # Extract document boundaries
    doc_pattern = re.compile(r'<Document\s+type="([^"]*)">')
    docid_pattern = re.compile(r'<DocID[^>]*>([^<]*)</DocID>')
    part_pattern = re.compile(r'<DocumentPart\s+type="([^"]*)">')
    sent_pattern = re.compile(r'<sentence\s+id="([^"]*)">(.*?)</sentence>', re.DOTALL)
    section_title_sent = re.compile(
        r'<DocumentPart\s+type="(?:SectionTitle|SubSectionTitle)">\s*<sentence[^>]*>([^<]*)</sentence>',
        re.DOTALL
    )
    
    # Track section context from SectionTitle/SubSectionTitle
    for match in section_title_sent.finditer(content):
        pass  # just verifying pattern works
    
    # Process document by document
    doc_splits = re.split(r'(<Document\s)', content)
    
    current_doc_type_attr = ""
    
    for chunk in doc_splits:
        # Update document ID
        docid_match = docid_pattern.search(chunk)
        if docid_match:
            current_doc_id = docid_match.group(1)
        
        # Update document type
        doc_match = doc_pattern.search(chunk)
        if doc_match:
            current_doc_type_attr = doc_match.group(1)
        
        # Track sections from DocumentPart and section titles
        lines = chunk.split('\n')
        current_section_name = "unknown"
        
        for i, line in enumerate(lines):
            # Check for DocumentPart type
            part_match = part_pattern.search(line)
            if part_match:
                part_type = part_match.group(1).lower()
                if part_type in ("sectiontitle", "subsectiontitle"):
                    # Next sentence has the section name
                    title_match = re.search(r'<sentence[^>]*>([^<]*)</sentence>', 
                                           '\n'.join(lines[i:i+3]))
                    if title_match:
                        section_name = title_match.group(1).strip().lower()
                        current_section_name = SECTION_MAP.get(
                            section_name, SECTION_MAP.get("unknown", 14)
                        )
                elif part_type == "abstracttext":
                    current_section_name = "abstract"
                elif part_type == "title":
                    current_section_name = "title"
        
        # Parse all sentences in this chunk
        for sent_match in sent_pattern.finditer(chunk):
            sent_id = sent_match.group(1)
            sent_raw = sent_match.group(2).strip()
            
            # Skip section title sentences (they're structural, not content)
            # Check if this sentence is inside a SectionTitle/SubSectionTitle
            sent_start = sent_match.start()
            preceding = chunk[max(0, sent_start-200):sent_start]
            if re.search(r'<DocumentPart\s+type="(?:SectionTitle|SubSectionTitle)">\s*$', preceding):
                # This is a section title - use it for context but don't label it
                title_text = re.sub(r'<[^>]+>', '', sent_raw).strip().lower()
                if title_text in SECTION_MAP:
                    current_section_name = title_text
                continue
            
            # Detect section from preceding DocumentPart
            part_before = re.findall(r'<DocumentPart\s+type="([^"]*)">', 
                                      chunk[:sent_start])
            if part_before:
                last_part = part_before[-1].lower()
                if last_part == "abstracttext":
                    section_name = "abstract"
                elif last_part == "title":
                    section_name = "title"
                else:
                    section_name = current_section_name if isinstance(current_section_name, str) else "unknown"
            else:
                section_name = current_section_name if isinstance(current_section_name, str) else "unknown"
            
            # Parse annotations from sentence XML
            plain_text, cues, scopes = parse_annotated_sentence(sent_raw)
            
            if not plain_text.strip():
                continue
            
            # Tokenize (whitespace-based, matching BioScope convention)
            tokens = plain_text.split()
            
            if not tokens:
                continue
            
            # Align labels to tokens
            cue_labels, scope_labels, cue_type_labels = align_labels_to_tokens(
                plain_text, tokens, cues, scopes
            )
            
            # Resolve section ID
            if isinstance(section_name, int):
                section_id = section_name
            else:
                section_id = SECTION_MAP.get(section_name, 14)
            
            instance = BioScopeInstance(
                text=plain_text,
                tokens=tokens,
                cue_labels=cue_labels,
                scope_labels=scope_labels,
                cue_type_labels=cue_type_labels,
                doc_type=doc_type,
                section_type=section_name if isinstance(section_name, str) else "unknown",
                section_id=section_id,
                doc_id=current_doc_id,
                sentence_id=sent_id,
                sentence_idx=sentence_counter,
                raw_xml=sent_raw,
                cue_annotations=cues,
                scope_annotations=scopes,
            )
            
            instances.append(instance)
            sentence_counter += 1
    
    return instances


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def load_and_split(
    abstracts_path: str,
    full_papers_path: str = None,
    clinical_path: str = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[BioScopeInstance], List[BioScopeInstance], List[BioScopeInstance]]:
    """
    Load BioScope corpus files and create train/val/test splits.
    
    Split is done at document level to prevent data leakage.
    Accepts abstracts, full papers, and optionally merged clinical data.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    
    all_instances = []
    
    # Parse abstracts
    print(f"Parsing abstracts from {abstracts_path}...")
    abstracts = parse_bioscope_file(abstracts_path, "abstract")
    all_instances.extend(abstracts)
    print(f"  Parsed {len(abstracts)} sentences")
    
    # Parse full papers if provided
    if full_papers_path and os.path.exists(full_papers_path):
        print(f"Parsing full papers from {full_papers_path}...")
        papers = parse_bioscope_file(full_papers_path, "full_paper")
        all_instances.extend(papers)
        print(f"  Parsed {len(papers)} sentences")
    
    # Parse clinical data if provided (requires merged XML from ScopeMerger)
    if clinical_path and os.path.exists(clinical_path):
        print(f"Parsing clinical records from {clinical_path}...")
        clinical = parse_bioscope_file(clinical_path, "clinical")
        # Filter out anonymized sentences (all * tokens)
        real = [i for i in clinical 
                if not all(t.replace('*','').replace('.','').strip() == '' 
                          for t in i.tokens)]
        all_instances.extend(real)
        print(f"  Parsed {len(clinical)} sentences ({len(real)} with real text)")
    
    # Group by document
    docs = defaultdict(list)
    for inst in all_instances:
        docs[inst.doc_id].append(inst)
    
    doc_ids = list(docs.keys())
    rng.shuffle(doc_ids)
    
    n_train = int(len(doc_ids) * train_ratio)
    n_val = int(len(doc_ids) * val_ratio)
    
    train_docs = doc_ids[:n_train]
    val_docs = doc_ids[n_train:n_train + n_val]
    test_docs = doc_ids[n_train + n_val:]
    
    train = [inst for did in train_docs for inst in docs[did]]
    val = [inst for did in val_docs for inst in docs[did]]
    test = [inst for did in test_docs for inst in docs[did]]
    
    return train, val, test


# ---------------------------------------------------------------------------
# Verification & statistics
# ---------------------------------------------------------------------------

def print_corpus_stats(instances: List[BioScopeInstance], name: str = ""):
    """Print detailed statistics about parsed data."""
    n = len(instances)
    has_neg = sum(1 for i in instances if 1 in i.scope_labels)
    has_spec = sum(1 for i in instances if 2 in i.scope_labels)
    has_both = sum(1 for i in instances if 1 in i.scope_labels and 2 in i.scope_labels)
    plain = n - has_neg - has_spec + has_both
    
    n_neg_cues = sum(sum(1 for c in i.cue_labels if c == 1) for i in instances)
    n_spec_cues = sum(
        sum(1 for c, t in zip(i.cue_labels, i.cue_type_labels) if c == 1 and t == 2)
        for i in instances
    )
    
    avg_len = sum(len(i.tokens) for i in instances) / max(n, 1)
    
    print(f"\n{'='*50}")
    print(f"  {name} ({n} sentences)")
    print(f"{'='*50}")
    print(f"  With negation:    {has_neg:5d} ({100*has_neg/max(n,1):.1f}%)")
    print(f"  With speculation: {has_spec:5d} ({100*has_spec/max(n,1):.1f}%)")
    print(f"  With both:        {has_both:5d} ({100*has_both/max(n,1):.1f}%)")
    print(f"  Plain:            {plain:5d} ({100*plain/max(n,1):.1f}%)")
    print(f"  Avg tokens/sent:  {avg_len:.1f}")
    print(f"  Negation cues:    {n_neg_cues}")
    
    # Section distribution
    sections = defaultdict(int)
    for i in instances:
        sections[i.section_type] += 1
    print(f"  Sections: {dict(sections)}")


def verify_annotations(instances: List[BioScopeInstance], n_samples: int = 5):
    """Print sample sentences with labels for manual verification."""
    # Show some annotated sentences
    annotated = [i for i in instances if any(c > 0 for c in i.cue_labels)]
    
    print(f"\n{'='*60}")
    print(f"SAMPLE ANNOTATED SENTENCES (verify correctness)")
    print(f"{'='*60}")
    
    import random
    random.seed(42)
    samples = random.sample(annotated, min(n_samples, len(annotated)))
    
    for inst in samples:
        print(f"\nID: {inst.sentence_id} | Section: {inst.section_type}")
        print(f"Text: {inst.text}")
        
        # Show labeled tokens
        parts = []
        for tok, cue, scope, ctype in zip(
            inst.tokens, inst.cue_labels, inst.scope_labels, inst.cue_type_labels
        ):
            labels = []
            if cue == 1:
                labels.append("B-CUE")
            elif cue == 2:
                labels.append("I-CUE")
            
            if scope == 1:
                labels.append("NEG")
            elif scope == 2:
                labels.append("SPEC")
            
            if labels:
                parts.append(f"[{tok}]({'|'.join(labels)})")
            else:
                parts.append(tok)
        
        print(f"Labels: {' '.join(parts)}")
        print("-" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    abstracts_path = "/mnt/user-data/uploads/abstracts.xml"
    papers_path = "/mnt/user-data/uploads/full_papers.xml"
    
    # Parse
    print("Parsing BioScope corpus...")
    abstracts = parse_bioscope_file(abstracts_path, "abstract")
    papers = parse_bioscope_file(papers_path, "full_paper")
    
    print_corpus_stats(abstracts, "Abstracts")
    print_corpus_stats(papers, "Full Papers")
    
    # Verify
    verify_annotations(abstracts, 5)
    verify_annotations(papers, 3)
    
    # Split
    print("\n\nCreating train/val/test splits...")
    train, val, test = load_and_split(abstracts_path, papers_path)
    
    print_corpus_stats(train, "Train")
    print_corpus_stats(val, "Val")
    print_corpus_stats(test, "Test")
    
    print(f"\nDone! Total: {len(train)+len(val)+len(test)} sentences")
