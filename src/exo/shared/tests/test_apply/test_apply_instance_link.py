from exo.shared.apply import (
    apply_instance_deleted,
    apply_instance_link_created,
    apply_instance_link_deleted,
)
from exo.shared.types.events import (
    InstanceDeleted,
    InstanceLinkCreated,
    InstanceLinkDeleted,
)
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId


def _link(
    prefill: list[InstanceId],
    decode: list[InstanceId],
    link_id: InstanceLinkId | None = None,
) -> InstanceLink:
    return InstanceLink(
        link_id=link_id or InstanceLinkId(),
        prefill_instances=prefill,
        decode_instances=decode,
    )


def test_create_link() -> None:
    state = State()
    link = _link([InstanceId("a")], [InstanceId("b")])
    new_state = apply_instance_link_created(InstanceLinkCreated(link=link), state)
    assert new_state.instance_links == {link.link_id: link}


def test_update_replaces_existing_link() -> None:
    a, b, c = InstanceId("a"), InstanceId("b"), InstanceId("c")
    link = _link([a], [b])
    state = State(instance_links={link.link_id: link})

    updated = link.model_copy(update={"decode_instances": [b, c]})
    new_state = apply_instance_link_created(InstanceLinkCreated(link=updated), state)
    assert set(new_state.instance_links[link.link_id].decode_instances) == {b, c}


def test_delete_link() -> None:
    link = _link([InstanceId("a")], [InstanceId("b")])
    state = State(instance_links={link.link_id: link})

    new_state = apply_instance_link_deleted(
        InstanceLinkDeleted(link_id=link.link_id), state
    )
    assert new_state.instance_links == {}


def test_instance_deleted_strips_from_links() -> None:
    a, b, c = InstanceId("a"), InstanceId("b"), InstanceId("c")
    link = _link([a, c], [b])
    state = State(instance_links={link.link_id: link})

    new_state = apply_instance_deleted(InstanceDeleted(instance_id=a), state)
    remaining = new_state.instance_links[link.link_id]
    assert remaining.prefill_instances == [c]
    assert remaining.decode_instances == [b]


def test_instance_deleted_drops_link_when_role_empties() -> None:
    a, b = InstanceId("a"), InstanceId("b")
    link = _link([a], [b])
    state = State(instance_links={link.link_id: link})

    new_state = apply_instance_deleted(InstanceDeleted(instance_id=a), state)
    assert link.link_id not in new_state.instance_links
