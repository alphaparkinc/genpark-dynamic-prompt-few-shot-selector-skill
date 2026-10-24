from client import DynamicPromptFewShotSelectorClient
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

def main():
    client = DynamicPromptFewShotSelectorClient(diversity_lambda=0.75)
    
    bank = [
        {"id": "sql_1", "input": "Get total revenue by customer country", "output": "SELECT country, SUM(revenue) FROM orders GROUP BY country;"},
        {"id": "sql_2", "input": "Find active users who subscribed in October", "output": "SELECT * FROM users WHERE status='active' AND subscription_date >= '2026-10-01';"},
        {"id": "sql_3", "input": "Calculate average order value per country and order status", "output": "SELECT country, order_status, AVG(order_total) FROM orders GROUP BY country, order_status;"},
        {"id": "nlp_1", "input": "Summarize customer feedback comment", "output": "Customer expressed satisfaction with shipping speed."},
        {"id": "sql_4", "input": "List top 5 products by review score", "output": "SELECT product_name, AVG(rating) as score FROM reviews GROUP BY product_name ORDER BY score DESC LIMIT 5;"}
    ]
    
    query = "Show aggregate sales revenue grouped by country"
    res = client.select_few_shot_exemplars(query, bank, k=2, max_token_budget=150)
    
    print("=== DYNAMIC PROMPT FEW-SHOT SELECTOR ===")
    print(f"Target Query: '{res['target_query']}'")
    print(f"Selected {res['exemplars_selected']} exemplars ({res['total_tokens_consumed']} tokens):")
    for ex in res['selected_exemplars']:
        print(f"  - [{ex['id']}] Sim: {ex['relevance_similarity']} | MMR: {ex['mmr_score']}")
        print(f"    Input : {ex['input']}")
        print(f"    Output: {ex['output']}")

if __name__ == "__main__":
    main()
