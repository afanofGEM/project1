import torch

from src.mlp.mlp_model import TicketClassifierMLP


def test_mlp_output_shape():
    model = TicketClassifierMLP(
        vocab_size=100,
        embedding_dim=16,
        padding_id=0,
        hidden_dim=32,
        dropout=0.0,
        num_classes=5,
    )

    text_ids = torch.tensor([
        [2, 3, 4, 0],
        [5, 6, 0, 0],
    ])

    output = model(text_ids)

    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()


def test_mlp_accepts_all_padding():
    model = TicketClassifierMLP(
        vocab_size=100,
        embedding_dim=16,
        padding_id=0,
        hidden_dim=32,
        dropout=0.0,
        num_classes=5,
    )

    text_ids = torch.zeros((2, 4), dtype=torch.long)
    output = model(text_ids)

    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()