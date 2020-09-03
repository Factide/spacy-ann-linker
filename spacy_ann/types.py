# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pydantic import BaseModel


class AliasCandidate(BaseModel):
    """A data class representing a candidate alias
    that a NER mention may be linked to.
    """

    alias: str
    similarity: float


class KnowledgeBaseCandidate(BaseModel):
    entity: str
    context_similarity: float
    prior_probability: float
    type_label: str

index_vs_kb_type = {
    0: 'UNK',
    1: 'ORG',
    2: 'GPE',
    3: 'PERSON'
}
kb_type_vs_index = {value: key for key, value in index_vs_kb_type.items()}