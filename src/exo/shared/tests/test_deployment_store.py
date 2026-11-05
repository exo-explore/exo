# pyright: reportPrivateUsage=false
"""Every node keeps the deployments it has seen on disk, and merges views of them the same
way: a launch on the other side of a split, or one a new master never saw, survives, and a
stop is never undone."""

import json
from pathlib import Path

from exo.master.deployments import spec_of
from exo.master.tests.test_deployments_running_master import (
    LEAVES,
    MODEL,
    STAYS,
    instance_on,
)
from exo.shared.deployment_store import (
    FORMAT,
    REMOVED_DEPLOYMENTS_KEPT,
    DeploymentKeeper,
    Deployments,
    load,
    record_of,
    save,
)
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import DeploymentId, NodeId
from exo.shared.types.deployments import Deployment, DeploymentRecord, Hold
from exo.shared.types.state import State
from exo.utils.channels import channel

FIXTURES = Path(__file__).parent / "fixtures"
EARLIER = "2026-11-03T11:00:00+00:00"
LATER = "2026-11-03T12:00:00+00:00"


def record(
    revision: int = 0, last_nodes: list[NodeId] | None = None
) -> DeploymentRecord:
    return DeploymentRecord(
        deployment_id=DeploymentId(),
        spec=spec_of(instance_on(STAYS)),
        model_card=MODEL,
        revision=revision,
        last_nodes=last_nodes or [],
    )


def deployment(of: DeploymentRecord) -> Deployment:
    return Deployment(
        deployment_id=of.deployment_id,
        spec=of.spec,
        model_card=of.model_card,
        revision=of.revision,
        last_nodes=of.last_nodes,
    )


# Merging


def test_views_merge_into_their_union() -> None:
    mine, theirs = record(), record()
    view = Deployments(records={mine.deployment_id: mine})

    assert view.merge([theirs], {})
    assert set(view.records) == {mine.deployment_id, theirs.deployment_id}
    assert not view.merge([theirs], {})


def test_the_higher_revision_wins() -> None:
    old = record(revision=1)
    resumed = old.model_copy(update={"revision": 2})
    view = Deployments(records={old.deployment_id: resumed})

    view.merge([old], {})

    assert view.records[old.deployment_id] == resumed


def test_where_it_last_ran_is_the_latest_word_on_it() -> None:
    """Not the user's to change: a record of the same revision updates it, and one that
    doesn't say keeps it."""
    old = record(last_nodes=[STAYS])
    view = Deployments(records={old.deployment_id: old})
    moved = old.model_copy(update={"last_nodes": [LEAVES]})

    assert view.merge([moved], {})
    assert not view.merge([old.model_copy(update={"last_nodes": []})], {})
    assert view.records[old.deployment_id].last_nodes == [LEAVES]


def test_a_stop_always_wins_and_stays() -> None:
    stopped = record(revision=5)
    view = Deployments(removed={stopped.deployment_id: EARLIER})

    view.merge([stopped], {})

    assert view.records == {}
    assert view.removed == {stopped.deployment_id: EARLIER}


def test_only_the_latest_stops_are_kept() -> None:
    view = Deployments()
    stops = {
        DeploymentId(): f"2026-11-03T12:{minute:02d}:00+00:00"
        for minute in range(REMOVED_DEPLOYMENTS_KEPT // 10)
    }
    for i in range(10):
        view.merge(
            [],
            {
                DeploymentId(): f"2026-11-0{i % 2 + 1}T00:{i:02d}:00+00:00"
                for _ in range(REMOVED_DEPLOYMENTS_KEPT // 10)
            },
        )
    view.merge([], stops)

    assert len(view.removed) == REMOVED_DEPLOYMENTS_KEPT
    assert stops.keys() <= view.removed.keys()


# The file


def test_what_is_saved_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "deployments.json"
    kept = record(revision=2, last_nodes=[STAYS])
    view = Deployments(
        records={kept.deployment_id: kept}, removed={DeploymentId(): LATER}
    )

    save(path, view)
    loaded = load(path)

    assert loaded.writable
    assert loaded.deployments == view
    assert not path.with_name("deployments.json.tmp").exists()


def test_a_file_of_the_first_format_loads() -> None:
    """Every later exo must still read what this one wrote: a format bump comes with a
    migration, and this file keeps that honest."""
    loaded = load(FIXTURES / "deployments-format1.json")

    assert loaded.writable
    assert [
        (
            record.spec.model_id,
            record.spec.min_nodes,
            record.revision,
            record.last_nodes,
        )
        for record in loaded.deployments.records.values()
    ] == [("mlx-community/Llama-3.2-1B-Instruct-4bit", 2, 1, ["12D3KooWExampleNodeA"])]
    assert list(loaded.deployments.removed.values()) == [EARLIER]


def test_a_file_of_a_newer_format_is_left_alone(tmp_path: Path) -> None:
    path = tmp_path / "deployments.json"
    newer = json.dumps({"format": FORMAT + 1, "deployments": [{"what": "unknown"}]})
    path.write_text(newer)
    state = State(deployments={(r := record()).deployment_id: deployment(r)})
    keeper = DeploymentKeeper(path, lambda: state, channel[ForwarderCommand]()[0])

    keeper.sync()

    assert path.read_text() == newer


def test_a_file_that_cant_be_read_is_moved_aside(tmp_path: Path) -> None:
    path = tmp_path / "deployments.json"
    path.write_text("{ not json")

    loaded = load(path)

    assert loaded.writable and loaded.deployments == Deployments()
    (aside,) = tmp_path.glob("deployments.json.unreadable-*")
    assert aside.read_text() == "{ not json"


def test_a_deployment_that_cant_be_read_is_left_out(tmp_path: Path) -> None:
    path = tmp_path / "deployments.json"
    kept = record()
    path.write_text(
        json.dumps(
            {
                "format": FORMAT,
                "deployments": [
                    {"deploymentId": "broken"},
                    kept.model_dump(mode="json", by_alias=True),
                ],
                "removed": {},
            }
        )
    )

    assert list(load(path).deployments.records) == [kept.deployment_id]


# The keeper


def keeper_at(
    path: Path, state: State | None
) -> tuple[DeploymentKeeper, list[State | None]]:
    """A keeper reading `states[0]`, which the test can replace."""
    states: list[State | None] = [state]
    return DeploymentKeeper(
        path, lambda: states[0], channel[ForwarderCommand]()[0]
    ), states


def test_an_empty_or_late_state_never_deletes_anything(tmp_path: Path) -> None:
    path = tmp_path / "deployments.json"
    kept = record()
    save(path, Deployments(records={kept.deployment_id: kept}))
    keeper, _ = keeper_at(path, State())

    keeper.sync()

    assert list(load(path).deployments.records) == [kept.deployment_id]


def test_what_the_cluster_launches_and_stops_is_saved(tmp_path: Path) -> None:
    path = tmp_path / "deployments.json"
    launched = record()
    keeper, states = keeper_at(
        path, State(deployments={launched.deployment_id: deployment(launched)})
    )
    keeper.sync()
    assert list(load(path).deployments.records) == [launched.deployment_id]

    states[0] = State(removed_deployments={launched.deployment_id: LATER})
    keeper.sync()

    assert load(path).deployments == Deployments(
        removed={launched.deployment_id: LATER}
    )


def test_a_deployment_from_disk_is_restored_until_seen_running(tmp_path: Path) -> None:
    path = tmp_path / "deployments.json"
    kept = record()
    save(path, Deployments(records={kept.deployment_id: kept}))
    keeper, states = keeper_at(path, State())
    assert keeper.announcement().restored == [kept.deployment_id]

    # The master has it, but held as restored: that isn't running it
    held = deployment(kept).model_copy(update={"hold": Hold(kind="Restored")})
    states[0] = State(deployments={kept.deployment_id: held})
    keeper.sync()
    assert keeper.announcement().restored == [kept.deployment_id]

    states[0] = State(deployments={kept.deployment_id: deployment(kept)})
    keeper.sync()
    assert keeper.announcement().restored == []


def test_each_new_master_is_told_once_and_again_while_it_lacks_some(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deployments.json"
    kept = record()
    save(path, Deployments(records={kept.deployment_id: kept}))
    knows = State(deployments={kept.deployment_id: deployment(kept)})
    keeper, _ = keeper_at(path, knows)

    # A node tells the first master it follows
    assert keeper.announcement_due(knows, now=0.0)
    keeper._announce_now = False
    # ...and no more while the master knows everything
    assert not keeper.announcement_due(knows, now=100.0)
    # ...but again while the master lacks one
    assert keeper.announcement_due(State(), now=100.0)
    # ...and once to each new master
    keeper.follow_new_master()
    assert keeper.announcement_due(knows, now=100.0)


def test_a_node_with_nothing_to_tell_says_nothing(tmp_path: Path) -> None:
    keeper, _ = keeper_at(tmp_path / "deployments.json", State())

    assert not keeper.announcement_due(State(), now=0.0)
    announcement = keeper.announcement()
    assert (
        announcement.deployments,
        announcement.removed,
        announcement.restored,
    ) == ([], {}, [])


def test_records_keep_what_the_user_asked_for_and_where_it_ran() -> None:
    """Not the master's holds and waits: they are its own side's decisions."""
    kept = record(revision=3, last_nodes=[STAYS])
    full = deployment(kept).model_copy(
        update={
            "hold": Hold(kind="CrashLoop"),
            "losses": [LATER],
            "not_before": LATER,
        }
    )

    assert record_of(full) == kept
    # Where it runs now, if it does
    assert record_of(full, [LEAVES]).last_nodes == [LEAVES]


def test_a_new_master_is_told_what_the_previous_one_was_running(tmp_path: Path) -> None:
    """In the first announcement, as this node last saw it before the new master's state
    replaced it: a hold the new master has for one of them is its own side's. Later ones tell
    what runs now, so an old view doesn't lift a hold the new master made since."""
    kept = record()
    instance = instance_on(STAYS).model_copy(
        update={"deployment_id": kept.deployment_id}
    )
    keeper, states = keeper_at(
        tmp_path / "deployments.json",
        State(
            deployments={kept.deployment_id: deployment(kept)},
            instances={instance.instance_id: instance},
        ),
    )
    keeper.sync()

    keeper.follow_new_master()
    # The new master's state, without it
    states[0] = State(deployments={kept.deployment_id: deployment(kept)})
    keeper.sync()

    assert keeper.announcement().running == [kept.deployment_id]
    # ...and where it ran is kept
    assert load(tmp_path / "deployments.json").deployments.records[
        kept.deployment_id
    ].last_nodes == [STAYS]
    # Told again later: what runs now
    keeper.told()
    assert keeper.announcement().running == []


def test_a_new_master_is_told_again_what_it_lacks(tmp_path: Path) -> None:
    """The first announcement to a new master may be lost (a command
    dropped under load, or sent to a master that just died): a node tells it again while it
    lacks a stop, or the user's later change, as well as a deployment. A stop it would drop
    again (older than every one it keeps) doesn't count."""
    stopped = DeploymentId()
    resumed = record(revision=2)
    has_all = State(
        deployments={resumed.deployment_id: deployment(resumed)},
        removed_deployments={stopped: LATER},
    )
    keeper, _ = keeper_at(tmp_path / "deployments.json", has_all)
    keeper.sync()
    keeper._announce_now = False
    assert not keeper.announcement_due(has_all, now=100.0)
    # It lacks the stop
    lacks_stop = has_all.model_copy(update={"removed_deployments": {}})
    assert keeper.announcement_due(lacks_stop, now=100.0)
    # It has an older revision
    older = deployment(resumed).model_copy(update={"revision": 1})
    lacks_resume = has_all.model_copy(
        update={"deployments": {resumed.deployment_id: older}}
    )
    assert keeper.announcement_due(lacks_resume, now=100.0)
    # A full list of later stops: it would drop this one again
    full = {
        DeploymentId(): "2026-11-03T13:00:00+00:00"
        for _ in range(REMOVED_DEPLOYMENTS_KEPT)
    }
    assert not keeper.announcement_due(
        has_all.model_copy(update={"removed_deployments": full}), now=100.0
    )
