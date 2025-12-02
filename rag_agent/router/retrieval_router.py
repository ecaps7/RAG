from __future__ import annotations

from ..common.config import TOP_K
from ..common.types import Intent, RetrievalPlan


class RetrievalRouter:
    def plan(self, intent: Intent) -> RetrievalPlan:
        if intent in {Intent.data_lookup, Intent.definition_lookup, Intent.meta_query}:
            return RetrievalPlan(
                use_local=True,
                use_web=False,
                local_top_k=TOP_K["local"],
                web_top_k=TOP_K["web"],
                hybrid_strategy="balance",
            )
        elif intent in {Intent.external_context, Intent.forecast}:
            return RetrievalPlan(
                use_local=False,
                use_web=True,
                local_top_k=TOP_K["local"],
                web_top_k=TOP_K["web"],
                hybrid_strategy="balance",
            )
        else:  # reasoning
            return RetrievalPlan(
                use_local=True,
                use_web=True,
                local_top_k=TOP_K["local"],
                web_top_k=TOP_K["web"],
                hybrid_strategy="balance",
            )