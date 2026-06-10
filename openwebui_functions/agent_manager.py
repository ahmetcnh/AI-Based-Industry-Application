"""
OpenWebUI Function Plugin — Coiltech Agent Manager
Install in Open WebUI: Admin → Functions → New Function → paste this file.

Commands (type in chat):
  /create-agent name=MyAgent prompt=You are... model=ollama:gemma3 collection=service_maintenance
  /list-agents
  /delete-agent id=3
"""
import json
import re
import httpx
from pydantic import BaseModel

BACKEND_URL = "http://backend:8000"


class EventEmitter:
    def __init__(self, event_emitter=None):
        self.event_emitter = event_emitter

    async def emit(self, description="", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter({
                "type": "status",
                "data": {"description": description, "status": status, "done": done},
            })


class Tools:
    class Valves(BaseModel):
        backend_url: str = BACKEND_URL
        tenant_id: int = 1
        user_id: int = 1

    def __init__(self):
        self.valves = self.Valves()

    async def run_command(
        self, command: str, __event_emitter__=None, __user__=None
    ) -> str:
        emitter = EventEmitter(__event_emitter__)
        headers = {
            "X-Tenant-ID": str(self.valves.tenant_id),
            "X-User-ID": str(self.valves.user_id),
        }
        base = self.valves.backend_url.rstrip("/")
        cmd = command.strip()

        if cmd.startswith("/list-agents"):
            await emitter.emit("Fetching agents...")
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{base}/agents", headers=headers)
                r.raise_for_status()
                agents = r.json()
            await emitter.emit("Done", status="complete", done=True)
            if not agents:
                return "No agents found."
            lines = [
                f"**ID {a['id']}**: {a['name']} | model: {a.get('model', '?')} | collection: {a.get('collection', '?')}"
                for a in agents
            ]
            return "\n".join(lines)

        if cmd.startswith("/create-agent"):
            params = dict(re.findall(r'(\w+)=([^\s]+)', cmd))
            name = params.get("name", "NewAgent")
            prompt = params.get("prompt", "You are a helpful assistant.")
            model = params.get("model", "ollama:gemma3")
            collection = params.get("collection", "service_maintenance")
            await emitter.emit(f"Creating agent '{name}'...")
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{base}/agents",
                    headers=headers,
                    json={"name": name, "system_prompt": prompt, "model": model, "collection": collection},
                )
                r.raise_for_status()
                agent = r.json()
            await emitter.emit("Done", status="complete", done=True)
            return f"Agent created: **{agent['name']}** (ID: {agent['id']}) — model: {agent.get('model')}, collection: {agent.get('collection')}"

        if cmd.startswith("/delete-agent"):
            params = dict(re.findall(r'(\w+)=([^\s]+)', cmd))
            agent_id = params.get("id")
            if not agent_id:
                return "Usage: /delete-agent id=<id>"
            await emitter.emit(f"Deleting agent {agent_id}...")
            async with httpx.AsyncClient() as client:
                r = await client.delete(f"{base}/agents/{agent_id}", headers=headers)
                r.raise_for_status()
            await emitter.emit("Done", status="complete", done=True)
            return f"Agent {agent_id} deleted."

        return f"Unknown command: {cmd}. Available: /list-agents, /create-agent, /delete-agent"
