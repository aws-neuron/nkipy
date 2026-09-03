def token_embedding(tok_embedding, token_ids):
    # tok_embedding: [N, H]
    # token_ids: [B, S]

    ids_1d = token_ids.reshape(-1)
    hidden = tok_embedding[ids_1d, :]

    hidden = hidden.reshape(*token_ids.shape, -1)

    return hidden
