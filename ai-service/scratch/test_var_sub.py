import re

def robust_cleanup(resolved):
    # 1. First, replace bold variables/placeholders like **$$Survivor Name** or **[Cinematic_Style]**
    # The prefix ($$ or [) is mandatory so we don't match already substituted bold values like **John**
    resolved = re.sub(r"\*\*+(?:\$\$|\[)[a-zA-Z0-9_'\s-]+?(?:\$\$|\])?\*\*+", "", resolved)
    
    # 2. Replace any remaining bracketed variables: [Cinematic_Style] or [Cinematic Style]
    resolved = re.sub(r"\[[a-zA-Z0-9_'\s-]+?\]", "", resolved)
    
    # 3. Replace any remaining double-dollar variables without spaces (like $$survivor_name)
    resolved = re.sub(r"\$\$[a-zA-Z0-9_']+\b", "", resolved)
    
    # 4. Replace remaining stray asterisks groups of 2 or more (like **, ***, ****) that might be left
    # But wait! If we do resolved = re.sub(r"\*\*+", "", resolved), it will remove bolding from valid markdown!
    # Let's think: do we want to strip all bolding? No! If a user has valid bolding, we shouldn't strip it.
    # We only want to strip empty bolding like **** or *** that resulted from deleting the variables.
    # So let's replace empty bolding or spaces-only bolding:
    resolved = re.sub(r"\*\*+\s*\*+", "", resolved)
    
    # 5. Clean up any double spaces or leading/trailing whitespace
    resolved = re.sub(r"\s+", " ", resolved).strip()
    return resolved

# Let's test different cases
test_cases = [
    "perhaps a **$$Survivor Name** navigating through the desolate streets.",
    "I'm looking for a **[Cinematic_Style]** visual mood and color grade.",
    "Execute using my exact input Survivor name: $$survivor_name.",
    "What if we have **** left over?",
    "What if we have *** left over?",
    "A **$$Survivor Name** in a **[Background_Scene]** with a **$$Color_Accent** highlight.",
    "A already substituted: perhaps a **John** navigating through the desolate streets of **New York**.",
]

for i, tc in enumerate(test_cases):
    print(f"Case {i+1} Original: {tc}")
    print(f"Case {i+1} Cleaned:  {robust_cleanup(tc)}")
    print("-" * 50)

