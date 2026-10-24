import math, re, sys
sys.stdout.reconfigure(encoding='utf-8')

class DynamicPromptFewShotSelectorClient:
    """
    Selects optimal few-shot exemplars from a candidate exemplar bank using
    TF-IDF term vector cosine similarity, Maximal Marginal Relevance (MMR) for diversity,
    and knapsack token budgeting.
    """
    def __init__(self, diversity_lambda=0.70):
        self.diversity_lambda = diversity_lambda

    def tokenize(self, text):
        return re.findall(r'[a-zA-Z0-9_]+', text.lower())

    def compute_tf_vector(self, text, vocabulary):
        tokens = self.tokenize(text)
        vec = [0.0] * len(vocabulary)
        if not tokens:
            return vec
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for i, word in enumerate(vocabulary):
            if word in tf:
                vec[i] = tf[word] / float(len(tokens))
        return vec

    def cosine_similarity(self, vec_a, vec_b):
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def select_few_shot_exemplars(self, target_query, exemplar_bank, k=3, max_token_budget=400):
        if not exemplar_bank:
            return {"status": "success", "selected_exemplars": [], "total_tokens": 0}

        # Build vocabulary across target query and exemplar bank
        vocab_set = set(self.tokenize(target_query))
        for item in exemplar_bank:
            vocab_set.update(self.tokenize(item.get("input", "") + " " + item.get("output", "")))
        vocabulary = sorted(list(vocab_set))

        query_vec = self.compute_tf_vector(target_query, vocabulary)
        
        # Prepare candidate representations
        candidates = []
        for i, ex in enumerate(exemplar_bank):
            full_text = ex.get("input", "") + " " + ex.get("output", "")
            est_tokens = max(1, int(len(full_text.split()) * 1.3))
            vec = self.compute_tf_vector(full_text, vocabulary)
            sim_to_query = self.cosine_similarity(query_vec, vec)
            candidates.append({
                "id": ex.get("id", f"ex_{i+1}"),
                "input": ex.get("input", ""),
                "output": ex.get("output", ""),
                "tokens": est_tokens,
                "vec": vec,
                "sim": sim_to_query
            })

        selected = []
        consumed_tokens = 0
        remaining = list(candidates)

        while remaining and len(selected) < k:
            best_item = None
            best_mmr = -1e9
            best_idx = -1

            for idx, cand in enumerate(remaining):
                if consumed_tokens + cand["tokens"] > max_token_budget:
                    continue
                # Calculate max similarity to already selected exemplars
                if selected:
                    max_sim_selected = max(self.cosine_similarity(cand["vec"], s["vec"]) for s in selected)
                else:
                    max_sim_selected = 0.0

                mmr_score = self.diversity_lambda * cand["sim"] - (1.0 - self.diversity_lambda) * max_sim_selected
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_item = cand
                    best_idx = idx

            if best_item and best_idx != -1:
                selected.append({
                    "id": best_item["id"],
                    "input": best_item["input"],
                    "output": best_item["output"],
                    "tokens": best_item["tokens"],
                    "relevance_similarity": round(best_item["sim"], 4),
                    "mmr_score": round(best_mmr, 4),
                    "vec": best_item["vec"]
                })
                consumed_tokens += best_item["tokens"]
                remaining.pop(best_idx)
            else:
                break

        # Remove vector arrays from final serialization
        sanitized = []
        for s in selected:
            sanitized.append({
                "id": s["id"],
                "input": s["input"],
                "output": s["output"],
                "tokens": s["tokens"],
                "relevance_similarity": s["relevance_similarity"],
                "mmr_score": s["mmr_score"]
            })

        return {
            "status": "success",
            "target_query": target_query,
            "exemplars_requested": k,
            "exemplars_selected": len(sanitized),
            "total_tokens_consumed": consumed_tokens,
            "max_token_budget": max_token_budget,
            "selected_exemplars": sanitized
        }
