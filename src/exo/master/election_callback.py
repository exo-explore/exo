from loguru import logger

from exo.master.forwarder_supervisor import ForwarderRole, ForwarderSupervisor


class ElectionCallbacks:
    """
    Simple callbacks for the Rust election system to invoke.
    No event system involvement - just direct forwarder control.
    """

    def __init__(self, forwarder_supervisor: ForwarderSupervisor):
        self._forwarder_supervisor = forwarder_supervisor

    async def on_became_master(self) -> None:
        """Called when this node is elected as master"""
        logger.info("Node elected as master")
        await self._forwarder_supervisor.notify_role_change(ForwarderRole.MASTER)

    async def on_became_replica(self) -> None:
        """Called when this node becomes a replica"""
        logger.info("Node demoted to replica")
        await self._forwarder_supervisor.notify_role_change(ForwarderRole.REPLICA)
