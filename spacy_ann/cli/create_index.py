# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import spacy
import srsly
import typer
from spacy.kb import KnowledgeBase
from spacy_ann.candidate_generator import CandidateGenerator
from spacy_ann.types import kb_type_vs_index
from wasabi import Printer
from tqdm import tqdm
from itertools import tee

INPUT_DIM = 300  # dimension of pretrained input vectors
DESC_WIDTH = 300  # dimension of output entity vectors


def create_index(
    model: str,
    kb_dir: Path,
    output_dir: Path,
    new_model_name: str = "ann_linker",
    cg_threshold: float = 0.8,
    n_iter: int = 5,
    verbose: bool = True,
):

    """Create an AnnLinker based on the Character N-Gram
    TF-IDF vectors for aliases in a KnowledgeBase

    model (str): spaCy language model directory or name to load
    kb_dir (Path): path to the directory with kb entities.jsonl and aliases.jsonl files
    output_dir (Path): path to output_dir for spaCy model with ann_linker pipe


    kb File Formats
    
    e.g. entities.jsonl

    {"id": "a1", "description": "Machine learning (ML) is the scientific study of algorithms and statistical models..."}
    {"id": "a2", "description": "ML (\"Meta Language\") is a general-purpose functional programming language. It has roots in Lisp, and has been characterized as \"Lisp with types\"."}

    e.g. aliases.jsonl
    {"alias": "ML", "entities": ["a1", "a2"], "probabilities": [0.5, 0.5]}
    """
    msg = Printer(hide_animation=not verbose)

    msg.divider("Load Model")
    with msg.loading(f"Loading model {model}"):
        nlp = spacy.load(model)
        msg.good("Done.")

    if output_dir is not None:
        output_dir = Path(output_dir / new_model_name)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    entities, entities_copy = tee(srsly.read_jsonl(kb_dir / "entities.jsonl"))
    total_entities = sum(1 for _ in entities_copy)
    
    aliases, aliases_copy = tee(srsly.read_jsonl(kb_dir / "aliases.jsonl"))
    total_aliases = sum(1 for _ in aliases_copy)

    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=INPUT_DIM)

    empty_doc = nlp.make_doc('').vector

    for entity in tqdm(entities, desc='Adding entities to KB', total=total_entities):
        id = entity['id']
        if not kb.contains_entity(id):
            embedding = nlp.make_doc(entity['description']).vector if 'description' in entity else empty_doc
            label = entity['label'] if 'label' in entity else 0
            if label: label = kb_type_vs_index[label]
            kb.add_entity(entity=id, 
                          freq=label, #TODO: Add a proper "label" field (repurposed freq field as the type label)
                          entity_vector=embedding) 
            
    for alias in tqdm(aliases, desc="Setting kb entities and aliases", total=total_aliases):
        entities = [e for e in alias["entities"] if kb.contains_entity(e)]
        num_entities = len(entities)
        if num_entities > 0:
            prior_probabilities = alias['probabilities'] if len(alias['probabilities']) == num_entities else [1.0 / num_entities] * num_entities
            kb.add_alias(alias=alias["alias"], entities=entities, probabilities=prior_probabilities)

    msg.divider("Create ANN Index")
    alias_strings = kb.get_alias_strings()
    cg = CandidateGenerator().fit(alias_strings, verbose=True)

    ann_linker = nlp.create_pipe("ann_linker")
    ann_linker.set_kb(kb)
    ann_linker.set_cg(cg)

    nlp.add_pipe(ann_linker, last=True)

    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    nlp.from_disk(output_dir)


if __name__ == "__main__":
    typer.run(create_index)
