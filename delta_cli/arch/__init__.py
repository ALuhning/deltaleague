"""Architecture track: participant model code scored on tokenizer-free loss.

The participant's ``delta_model.Model`` runs only inside a sandboxed process
(``inner``). Everything it returns is a claim; ``harness`` verifies the claims
with a fresh, stateless process before any score is accepted.
"""
