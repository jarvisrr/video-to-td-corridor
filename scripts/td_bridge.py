"""Minimal stdio JSON-RPC client for a TouchDesigner MCP bridge (optional helper).

Usage:
    python3 td_bridge.py list
    python3 td_bridge.py -q execute_code '{}' code snippet.py

Set TD_BRIDGE_SERVER to the bridge's server script if it is not found
automatically (default: the newest TDMCP codex-bridge under the user's
Application Support folder). Needs `node` on PATH.
"""
import glob, json, os, shutil, subprocess, sys, threading


def find_server():
    env = os.environ.get('TD_BRIDGE_SERVER')
    if env:
        return env
    hits = sorted(glob.glob(os.path.expanduser(
        '~/Library/Application Support/Derivative/TDMCP/codex-bridge/*/server/index.js')))
    if not hits:
        raise SystemExit('No bridge found: set TD_BRIDGE_SERVER to its server script')
    return hits[-1]


NODE = shutil.which('node') or os.path.expanduser('~/.local/bin/node')
env = dict(os.environ, NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
quiet = '-q' in sys.argv
p = subprocess.Popen([NODE, find_server()], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL if quiet else sys.stderr, env=env, text=True)
_id = 0


def rpc(method, params=None, notify=False):
    global _id
    m = {'jsonrpc': '2.0', 'method': method}
    if params is not None:
        m['params'] = params
    if not notify:
        _id += 1
        m['id'] = _id
    p.stdin.write(json.dumps(m) + '\n')
    p.stdin.flush()
    if notify:
        return
    while True:
        line = p.stdout.readline()
        if not line:
            raise SystemExit('bridge closed (is TouchDesigner open with the bridge project?)')
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('id') == _id:
            return r


def text(r):
    if 'error' in r:
        return 'ERROR ' + json.dumps(r['error'])
    return '\n'.join(c.get('text') if c.get('type') == 'text' else '<%s>' % c.get('type')
                     for c in r['result'].get('content', []))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '-q']
    threading.Timer(100, lambda: (print('TIMEOUT'), os._exit(3))).start()
    rpc('initialize', {'protocolVersion': '2025-03-26', 'capabilities': {},
                       'clientInfo': {'name': 'td-bridge-script', 'version': '1'}})
    rpc('notifications/initialized', notify=True)
    if args[0] == 'list':
        for t in rpc('tools/list')['result']['tools']:
            print(t['name'], '|', json.dumps(t['inputSchema'].get('properties', {}))[:300])
    else:
        a = json.loads(args[1]) if len(args) > 1 else {}
        if len(args) > 2:
            a[args[2]] = open(args[3]).read()
        print(text(rpc('tools/call', {'name': args[0], 'arguments': a})))
    sys.stdout.flush()
    p.kill()
    os._exit(0)
