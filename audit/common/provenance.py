"""
Generic provenance for the audit framework.

Implements a lightweight PROV-style model (Entities, Activities, Agents,
and the relations used/generated/derived_from/associated_with) per the
methodology's provenance requirements. This is intentionally NOT the
same module as audit.dataset's existing dataset/common/provenance.py
(which stays untouched, is D-pillar-scoped, and is not imported here) -
this is the generic, cross-pillar version used by Implementation,
Experimental, Cryptographic, and Integration.

Knows nothing about Gohr, Speck, datasets, models, or any specific
cryptanalytic artifact - only the general shape of "entity was used by
activity which generated entity, associated with agent".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class RelationType(str, Enum):
    USED = "used"
    GENERATED = "generated"
    DERIVED_FROM = "derived_from"
    ASSOCIATED_WITH = "associated_with"


@dataclass(frozen=True)
class Entity:
    """A PROV Entity: something that exists and can be described (a
    dataset snapshot, a checkpoint, a metric, an evidence object)."""
    entity_id: str
    entity_type: str
    content_hash: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id, "entity_type": self.entity_type,
            "content_hash": self.content_hash, "attributes": self.attributes,
        }


@dataclass(frozen=True)
class Agent:
    """A PROV Agent: something responsible for an activity (a person, a
    script, a pipeline)."""
    agent_id: str
    agent_type: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "agent_type": self.agent_type, "attributes": self.attributes}


@dataclass(frozen=True)
class Activity:
    """A PROV Activity: something that occurs over time and acts upon
    or with entities (training, evaluation, statistical analysis).

    An activity with ended_at_utc=None is INCOMPLETE. A completed
    activity must carry an end timestamp - see
    ProvenanceGraph.complete_activity() and
    ProvenanceGraph.validate(), which refuses to certify a graph whose
    activities generated evidence but never recorded completion.
    """
    activity_id: str
    activity_type: str
    started_at_utc: str
    ended_at_utc: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.ended_at_utc is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity_id": self.activity_id, "activity_type": self.activity_type,
            "started_at_utc": self.started_at_utc, "ended_at_utc": self.ended_at_utc,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class Relation:
    relation_type: RelationType
    subject_id: str
    object_id: str

    def to_dict(self) -> dict[str, str]:
        return {"relation_type": self.relation_type.value, "subject_id": self.subject_id, "object_id": self.object_id}


class ProvenanceCollisionError(RuntimeError):
    """
    Raised when an entity/agent/activity ID is reused for DIFFERENT
    content. A provenance graph in which a later record silently
    overwrote an earlier one with the same ID is not a trustworthy
    audit trail: the overwritten record would vanish without trace,
    and any evidence citing that ID would point at something other
    than what it actually used.
    """


class ProvenanceValidationError(RuntimeError):
    pass


@dataclass
class ProvenanceGraph:
    """
    An accumulating, append-only provenance graph. Pillars build one of
    these while they work and attach it to their evidence; the
    integration layer merges graphs from all pillars (Phase 5) without
    ever needing to know what a "checkpoint" or "dataset" actually is.

    Append-only is enforced: re-adding the SAME id with identical
    content is idempotent and allowed; re-adding it with DIFFERENT
    content raises ProvenanceCollisionError rather than overwriting.
    """
    entities: dict[str, Entity] = field(default_factory=dict)
    agents: dict[str, Agent] = field(default_factory=dict)
    activities: dict[str, Activity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    @staticmethod
    def _check_collision(store: dict, key: str, new_value: Any, kind: str) -> bool:
        """Returns True if the caller should write; raises on a genuine collision."""
        if key not in store:
            return True
        existing = store[key]
        if existing == new_value:
            return False  # idempotent re-add; nothing to do
        raise ProvenanceCollisionError(
            f"{kind} id {key!r} is already registered with different content. "
            f"Existing: {existing!r}. New: {new_value!r}. Provenance records are "
            "append-only; silently overwriting would break the audit trail."
        )

    def add_entity(self, entity: Entity) -> None:
        if self._check_collision(self.entities, entity.entity_id, entity, "Entity"):
            self.entities[entity.entity_id] = entity

    def add_agent(self, agent: Agent) -> None:
        if self._check_collision(self.agents, agent.agent_id, agent, "Agent"):
            self.agents[agent.agent_id] = agent

    def add_activity(self, activity: Activity) -> None:
        if self._check_collision(self.activities, activity.activity_id, activity, "Activity"):
            self.activities[activity.activity_id] = activity

    def complete_activity(self, activity_id: str, *, ended_at_utc: Optional[str] = None) -> None:
        """
        Record that an activity finished. Replacing an activity with its
        own completed form is the one permitted in-place update, since
        it adds information rather than destroying it.
        """
        if activity_id not in self.activities:
            raise ProvenanceValidationError(f"Unknown activity_id {activity_id!r}; cannot complete it.")
        existing = self.activities[activity_id]
        if existing.is_complete:
            return
        self.activities[activity_id] = Activity(
            activity_id=existing.activity_id, activity_type=existing.activity_type,
            started_at_utc=existing.started_at_utc,
            ended_at_utc=ended_at_utc or utc_timestamp(), attributes=existing.attributes,
        )

    def used(self, activity_id: str, entity_id: str) -> None:
        self.relations.append(Relation(RelationType.USED, activity_id, entity_id))

    def generated(self, entity_id: str, activity_id: str) -> None:
        self.relations.append(Relation(RelationType.GENERATED, entity_id, activity_id))

    def derived_from(self, entity_id: str, source_entity_id: str) -> None:
        self.relations.append(Relation(RelationType.DERIVED_FROM, entity_id, source_entity_id))

    def associated_with(self, activity_id: str, agent_id: str) -> None:
        self.relations.append(Relation(RelationType.ASSOCIATED_WITH, activity_id, agent_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "activities": {k: v.to_dict() for k, v in self.activities.items()},
            "relations": [r.to_dict() for r in self.relations],
        }

    def merge(self, other: "ProvenanceGraph") -> "ProvenanceGraph":
        """
        Non-destructive merge (used by Integration to combine per-pillar
        graphs). Raises ProvenanceCollisionError if the two graphs use
        the same ID for different content - a silent dict-update merge
        would let one pillar's record shadow another's.
        """
        merged = ProvenanceGraph()
        for graph in (self, other):
            for entity in graph.entities.values():
                merged.add_entity(entity)
            for agent in graph.agents.values():
                merged.add_agent(agent)
            for activity in graph.activities.values():
                merged.add_activity(activity)
        seen: set[tuple] = set()
        for relation in [*self.relations, *other.relations]:
            key = (relation.relation_type, relation.subject_id, relation.object_id)
            if key not in seen:
                seen.add(key)
                merged.relations.append(relation)
        return merged

    def validate(self, *, require_completed_activities: bool = True) -> list[str]:
        """
        Structural integrity check. Returns a list of problems (empty
        means valid). Checks:
          - every relation endpoint refers to a registered node;
          - every activity that GENERATED something has completed
            (an activity cannot have produced a finished artifact while
            still being in progress).
        """
        problems: list[str] = []
        known = set(self.entities) | set(self.agents) | set(self.activities)

        for relation in self.relations:
            for endpoint in (relation.subject_id, relation.object_id):
                if endpoint not in known:
                    problems.append(
                        f"Relation {relation.relation_type.value} references unknown node {endpoint!r}."
                    )

        if require_completed_activities:
            generating = {
                r.object_id for r in self.relations if r.relation_type == RelationType.GENERATED
            }
            for activity_id in generating:
                activity = self.activities.get(activity_id)
                if activity is not None and not activity.is_complete:
                    problems.append(
                        f"Activity {activity_id!r} generated an entity but has no ended_at_utc "
                        "timestamp; a completed activity must record its end time."
                    )
        return problems

    def trace_backward(self, entity_id: str) -> dict[str, Any]:
        """
        Given an entity (e.g. an evidence object), walk relations
        backward (derived_from / generated-by chains) to assemble the
        traceability chain a reviewer needs: evidence -> ... -> raw
        inputs. Returns the subgraph of relevant relations, not a
        narrative - callers render it.
        """
        visited_entities = {entity_id}
        visited_activities: set[str] = set()
        frontier = [entity_id]
        relevant_relations: list[Relation] = []

        while frontier:
            current = frontier.pop()
            for rel in self.relations:
                if rel.relation_type == RelationType.DERIVED_FROM and rel.subject_id == current:
                    relevant_relations.append(rel)
                    if rel.object_id not in visited_entities:
                        visited_entities.add(rel.object_id)
                        frontier.append(rel.object_id)
                if rel.relation_type == RelationType.GENERATED and rel.subject_id == current:
                    relevant_relations.append(rel)
                    if rel.object_id not in visited_activities:
                        visited_activities.add(rel.object_id)
                        for used_rel in self.relations:
                            if used_rel.relation_type == RelationType.USED and used_rel.subject_id == rel.object_id:
                                relevant_relations.append(used_rel)
                                if used_rel.object_id not in visited_entities:
                                    visited_entities.add(used_rel.object_id)
                                    frontier.append(used_rel.object_id)

        return {
            "root_entity": entity_id,
            "entities": {eid: self.entities[eid].to_dict() for eid in visited_entities if eid in self.entities},
            "activities": {aid: self.activities[aid].to_dict() for aid in visited_activities if aid in self.activities},
            "relations": [r.to_dict() for r in relevant_relations],
        }
