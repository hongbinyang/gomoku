import pytest
import torch

from gomoku_muzero.model.networks import MuZeroNetwork


def test_initial_inference_shapes() -> None:
    network = MuZeroNetwork(board_size=5, hidden_channels=16)
    observation = torch.zeros(2, 3, 5, 5)

    output = network.initial_inference(observation)

    assert output.hidden_state.shape == (2, 16, 5, 5)
    assert output.policy_logits.shape == (2, 25)
    assert output.value.shape == (2, 1)
    assert torch.all(output.value >= -1)
    assert torch.all(output.value <= 1)


def test_recurrent_inference_shapes() -> None:
    network = MuZeroNetwork(board_size=5, hidden_channels=16)
    root = network.initial_inference(torch.zeros(2, 3, 5, 5))

    output = network.recurrent_inference(
        root.hidden_state, torch.tensor([0, 24])
    )

    assert output.hidden_state.shape == (2, 16, 5, 5)
    assert output.reward.shape == (2, 1)
    assert output.policy_logits.shape == (2, 25)
    assert output.value.shape == (2, 1)


def test_unrolled_loss_backpropagates_through_all_three_networks() -> None:
    network = MuZeroNetwork(board_size=5, hidden_channels=8)
    output = network.initial_inference(torch.randn(2, 3, 5, 5))
    recurrent = network.recurrent_inference(
        output.hidden_state, torch.tensor([3, 17])
    )

    loss = (
        output.policy_logits.square().mean()
        + output.value.square().mean()
        + recurrent.reward.square().mean()
        + recurrent.policy_logits.square().mean()
        + recurrent.value.square().mean()
    )
    loss.backward()

    for module in (
        network.representation,
        network.dynamics,
        network.prediction,
    ):
        assert any(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad) > 0
            for parameter in module.parameters()
        )


@pytest.mark.parametrize("action", [-1, 25])
def test_recurrent_inference_rejects_out_of_range_actions(action: int) -> None:
    network = MuZeroNetwork(board_size=5)
    root = network.initial_inference(torch.zeros(1, 3, 5, 5))

    with pytest.raises(ValueError):
        network.recurrent_inference(root.hidden_state, torch.tensor([action]))
