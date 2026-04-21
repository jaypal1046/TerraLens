import random

subjects = ["I", "You", "It", "The system", "TerraLens", "The user", "The engine", "The data", "The model"]
verbs = ["am", "are", "is", "was", "will be", "can be", "provides", "optimizes", "calculates"]
objects = ["fast", "ready", "optimized", "a tool", "an engine", "the solution", "performance", "efficiency"]
questions = ["What is", "How does", "Where is", "Who is", "Can I"]

with open("brain_data/foundation/grammar_foundation.txt", "w") as f:
    for _ in range(5000):
        # Declarative patterns
        s = random.choice(subjects)
        v = random.choice(verbs)
        o = random.choice(objects)
        f.write(f"{s} {v} {o}.\n")
        
        # Interrogative patterns
        q = random.choice(questions)
        t = random.choice(objects)
        f.write(f"{q} {t}?\n")

print(f"[SUCCESS] Generated 10,000 grammar patterns in brain_data/foundation/grammar_foundation.txt")
