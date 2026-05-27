import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SQLCL_COMMAND = "d:/sqlcl/bin/sql.exe"

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command=SQLCL_COMMAND,
    args=["-mcp"],
)

async def call_tool(session: ClientSession, tool_command: str, arguments: dict | None = None):
  """
  Helper to call an MCP tool and print the result.
  """
  arguments = arguments or {}
  print(f"\n=== Calling tool command: {tool_command} ===")
  print(f"Arguments: {json.dumps(arguments, indent=2)}")

  result = await session.call_tool(tool_command, arguments)

  print("Result:")
  try:
    # Try to print structured content if present
    if hasattr(result, "content"):
      for item in result.content:
        if hasattr(item, "text"):
          print(item.text)
        else:
          print(item)
    else:
      print(result)
  except Exception:
    print(result)

  return result

async def run():
  async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
      # Initialize the connection
      await session.initialize()

      # List tools
      tools = await session.list_tools()
      print(f"Available commands: {[t.name for t in tools.tools]}")

      connection_list = await call_tool(session, "list-connections")
      connection = await call_tool(session, "connect", { "connection_name": "testpdb01" })
      run_sql = await call_tool(session, "run-sql", { "sql": "select to_char(sysdate, 'DD-MM-YYYY HH24:MI:SS') as current_time from dual" })
      run_sqlcl_result = await call_tool(session, "run-sqlcl", { "sqlcl": "show user" })
      disconnect_result = await call_tool(session, "disconnect")

def main():
  """Entry point for the client script."""
  asyncio.run(run())

if __name__ == "__main__":
  main()