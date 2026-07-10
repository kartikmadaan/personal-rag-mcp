import re


# Patterns observed in public PI corpora (Lakera/PromptArmor) — extend over time
_BAD_PATTERNS = [
   r"ignore (all|previous|above) (instructions|prompts)",
   r"system prompt",
   r"reveal (your )?(initial )?prompt",
   r"<\|im_start\|>system",
   r"###\s*system",
   r"act as (a )?(developer|root|admin)",
]
_BAD_RE = re.compile("|".join(_BAD_PATTERNS), re.IGNORECASE)




class PromptInjectionDetected(Exception):
   pass




def guard_query(q: str) -> str:
   if _BAD_RE.search(q):
       raise PromptInjectionDetected(q[:120])
   return q.strip()[:4000]   # length cap




def guard_context(text: str) -> str:
   # Strip control sequences; cap length per chunk
   text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
   return text[:2000]