"""Run explicitly with ``pytest tst/manual/smol_vm_agent_check.py`` on a KVM host with Docker and Smol installed."""

import uuid

import httpx
import pytest

from agent_env.a2a_agent.a2a_agent import DEFAULT_A2A_PORT
from agent_env.config import get_config, set_image_store
from agent_env.providers.sandbox_providers.smol_vm.provider import SmolVmSandboxProvider
from agent_env.store.routing import LocalRunImageStore
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from tst.integration.store.local_parallel_deploy_test import local_stack  # noqa: F401
from tst.util.a2a_test_agent import build_test_agent_image, put_test_agent


@pytest.mark.asyncio
async def test_agent_answers_from_smol_vm_and_cleans_up(local_stack):
    uid = uuid.uuid4().hex[:8]
    agent = put_test_agent(f"smol-agent-{uid}")
    context = TaskStepContext(instance_id=f"smol-agent-run-{uid}")

    try:
        await DeploySandboxTaskStep(
            id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="smol_vm",
            exposed_ports=[DEFAULT_A2A_PORT], cpu=2, memory_mb=4096, disk_size_gb=16,
        ).execute(context)
        await DeployAgentTaskStep(
            id="agent", version=None, env_ids=[], a2a_agent_id=agent.id, agent_name="solver", sandbox_name="box",
        ).execute(context)

        assert len(context.deployed_agents) == 1
        url = context.deployed_agents[0].a2a_url
        assert url == context.deployed_sandboxes[0].tunnel_urls[str(DEFAULT_A2A_PORT)]
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/.well-known/agent.json", timeout=30)
        assert response.status_code == 200
    finally:
        report = await teardown_run(context)
        assert not report.still_up


@pytest.mark.asyncio
async def test_agent_deploys_into_its_own_smol_vm(local_stack):
    agent = put_test_agent(f"smol-own-vm-{uuid.uuid4().hex[:8]}")

    try:
        deployed = await agent.deploy(sandbox_type="smol_vm", cpu=2, memory=4096, disk_size_gb=16)
        assert deployed.sandbox_type == "smol_vm"
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{deployed.a2a_url}/.well-known/agent.json", timeout=30)
        assert response.status_code == 200
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_local_run_registry_image_pulls_inside_smol_vm(local_stack):
    artifact = build_test_agent_image(f"smol-local-run-{uuid.uuid4().hex[:8]}")
    registry = get_config().get_image_store()
    set_image_store(LocalRunImageStore(registry, registry))
    sandbox = await SmolVmSandboxProvider().create_vm(cpu=2, memory=4096, disk_size_gb=16)

    try:
        await sandbox.pull_images([artifact.image_name])
        code, _, stderr = await sandbox.exec_with_output("docker", "image", "inspect", artifact.image_name)
        assert code == 0, stderr
    finally:
        await sandbox.terminate()
