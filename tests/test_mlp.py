"""Testes unitários para a arquitetura MLP."""

from __future__ import annotations

import pytest
import torch

from src.models.mlp import MLP


class TestMLPShape:
    def test_output_shape_single_sample(self, tiny_mlp: MLP, batch_input: torch.Tensor) -> None:
        # BatchNorm1d requer > 1 amostra em training mode; eval() usa running stats
        tiny_mlp.eval()
        with torch.no_grad():
            out = tiny_mlp(batch_input[:1])
        assert out.shape == (1, 1)

    def test_output_shape_batch(self, tiny_mlp: MLP, batch_input: torch.Tensor) -> None:
        out = tiny_mlp(batch_input)
        assert out.shape == (4, 1)

    def test_variable_batch_sizes(self, tiny_mlp: MLP) -> None:
        tiny_mlp.eval()
        for bs in (1, 8, 32, 64):
            x = torch.randn(bs, 10)
            with torch.no_grad():
                out = tiny_mlp(x)
            assert out.shape == (bs, 1), f"Falhou para batch_size={bs}"

    def test_custom_architecture(self) -> None:
        model = MLP(input_dim=30, output_dim=1, hidden_dims=[128, 64, 32])
        x = torch.randn(2, 30)
        assert model(x).shape == (2, 1)


class TestMLPNumerics:
    def test_no_nan_output(self, tiny_mlp: MLP, batch_input: torch.Tensor) -> None:
        assert not torch.isnan(tiny_mlp(batch_input)).any()

    def test_no_inf_output(self, tiny_mlp: MLP, batch_input: torch.Tensor) -> None:
        assert not torch.isinf(tiny_mlp(batch_input)).any()

    def test_sigmoid_output_in_range(self, tiny_mlp: MLP, batch_input: torch.Tensor) -> None:
        tiny_mlp.eval()
        with torch.no_grad():
            probs = torch.sigmoid(tiny_mlp(batch_input))
        assert (probs >= 0).all() and (probs <= 1).all()


class TestMLPParameters:
    def test_num_parameters_positive(self, tiny_mlp: MLP) -> None:
        assert tiny_mlp.num_parameters > 0

    def test_parameters_trainable(self, tiny_mlp: MLP) -> None:
        for p in tiny_mlp.parameters():
            assert p.requires_grad

    def test_gradients_flow(self, tiny_mlp: MLP, batch_input: torch.Tensor) -> None:
        loss = tiny_mlp(batch_input).mean()
        loss.backward()
        grads = [p.grad for p in tiny_mlp.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)


class TestMLPEvalMode:
    def test_deterministic_in_eval(self, batch_input: torch.Tensor) -> None:
        """Em eval(), dropout desligado — saída deve ser idêntica."""
        model = MLP(input_dim=10, output_dim=1, hidden_dims=[8], dropout=0.9)
        model.eval()
        with torch.no_grad():
            out1 = model(batch_input)
            out2 = model(batch_input)
        assert torch.allclose(out1, out2)
