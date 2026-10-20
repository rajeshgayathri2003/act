"""Runs the W&B launch agent with a Kubernetes token that renews, so its Job watcher keeps seeing Jobs finish.

The watcher keeps one client for the agent's life, and kubernetes_asyncio runs the kubeconfig's exec login (NRP's oidc-login, 30-minute tokens) only once; this re-runs it once the token expires.
Run from act/ as `uv run --env-file .env python -m ragmani.agent -e ragmani -q <queue> -v -j <jobs>`, with wandb launch-agent's own flags.
"""
import sys

from kubernetes_asyncio.config.kube_config import KubeConfigLoader
from wandb.cli.cli import cli

set_config = KubeConfigLoader._set_config


def renewing(loader, configuration):
  """KubeConfigLoader._set_config, plus a hook that runs before each request and re-runs the exec login once its token has expired."""
  set_config(loader, configuration)
  if not loader._user or "exec" not in loader._user: return

  async def renew(configuration):
    if await loader.load_from_exec_plugin(): set_config(loader, configuration)  # it returns at once while the token is still valid
  configuration.refresh_api_key_hook = renew


KubeConfigLoader._set_config = renewing  # private API: recheck after a kubernetes_asyncio upgrade


if __name__ == "__main__":
  cli(["launch-agent", *sys.argv[1:]])
