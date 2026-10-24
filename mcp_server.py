import sys, json
from client import DynamicPromptFewShotSelectorClient
sys.stdout.reconfigure(encoding='utf-8')

def handle_mcp():
    client = DynamicPromptFewShotSelectorClient()
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print(json.dumps({"status": "ready", "skill": "genpark-dynamic-prompt-few-shot-selector-skill"}))
        return
    req = json.loads(sys.stdin.read() or "{}")
    query = req.get("query", "Summarize user profile")
    bank = req.get("exemplar_bank", [])
    k = req.get("k", 3)
    res = client.select_few_shot_exemplars(query, bank, k=k)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    handle_mcp()
