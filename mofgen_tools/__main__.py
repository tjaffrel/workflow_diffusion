"""Entry point: ``python -m mofgen_tools`` starts the MCP server."""

from mofgen_tools.server import mcp

if __name__ == "__main__":
    mcp.run()
