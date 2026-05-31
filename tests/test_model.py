from pathlib import Path

import torch

from rmc.model import CoreConfig, EventBatch, RecurrentMemoryCore, RecurrentState
from rmc.runtime import PersistentRuntime, SandboxedTerminal, load_model
from rmc.tasks import DEFAULT_CURRICULUM, SwitchbackBanditFactory
from rmc.train import TrainConfig, curriculum_stage, rollout_loss


def small_config() -> CoreConfig:
    return CoreConfig(
        input_size=16,
        model_size=24,
        memory_slots=5,
        memory_size=16,
        num_channels=4,
        num_actions=4,
        max_reasoning_ticks=3,
    )


def test_forward_updates_recurrent_memory() -> None:
    config = small_config()
    model = RecurrentMemoryCore(config)
    state = model.initial_state(2)
    event = EventBatch(
        content=torch.randn(2, config.input_size),
        type_id=torch.tensor([0, 1]),
        channel_id=torch.tensor([0, 2]),
    )

    output = model(event, state)

    assert output.action_logits.shape == (2, config.num_actions)
    assert output.halt_probabilities.shape == (2, config.max_reasoning_ticks)
    assert output.read_weights.shape == (2, config.max_reasoning_ticks, config.memory_slots)
    assert output.write_weights.shape == (2, config.memory_slots)
    assert torch.allclose(output.halt_probabilities.sum(-1), torch.ones(2))
    assert output.write_weights.std(dim=-1).gt(0).all()
    assert not torch.allclose(output.state.memory, state.memory)
    assert output.state.steps.tolist() == [1, 1]


def test_state_can_persist_without_plaintext_memory(tmp_path: Path) -> None:
    config = small_config()
    model = RecurrentMemoryCore(config)
    state_path = tmp_path / "opaque-state.pt"
    runtime = PersistentRuntime(model, state_path=state_path)
    event = EventBatch(
        content=torch.randn(1, config.input_size),
        type_id=torch.tensor([0]),
        channel_id=torch.tensor([1]),
    )

    runtime.process(event)
    restored = RecurrentState.load(state_path)

    assert state_path.exists()
    assert restored.steps.item() == 1
    assert restored.memory.shape == (1, config.memory_slots, config.memory_size)


def test_curriculum_contains_switchback_and_rollout_backpropagates() -> None:
    config = small_config()
    factory = SwitchbackBanditFactory(
        input_size=config.input_size,
        num_actions=config.num_actions,
        max_channels=config.num_channels,
    )
    stage = DEFAULT_CURRICULUM[1]
    lifetime = factory.sample(stage, batch_size=2)
    feedback_event = lifetime.event_at(1, reward=torch.zeros(2), done=torch.zeros(2))
    model = RecurrentMemoryCore(config)
    loss, metrics = rollout_loss(
        model,
        factory,
        stage,
        batch_size=2,
        train_config=TrainConfig(),
        device=torch.device("cpu"),
    )
    loss.backward()

    assert lifetime.channel_id[0].eq(0).all()
    assert lifetime.channel_id[-1].eq(0).all()
    assert lifetime.channel_id.eq(1).any()
    assert lifetime.revisit_mask[-1].all()
    assert feedback_event.content[:, config.num_actions : config.num_actions * 2].sum().item() == 2
    assert feedback_event.content[:, config.num_actions * 2 : config.num_actions * 3].sum().item() == 2
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["query_accuracy"] <= 1.0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_terminal_adapter_only_renders_reviewed_commands() -> None:
    terminal = SandboxedTerminal((("pwd",), ("ls", "-la")))

    assert terminal.render(1) == "ls -la"
    try:
        terminal.render(2)
    except ValueError as exc:
        assert "not allowlisted" in str(exc)
    else:
        raise AssertionError("unknown commands must be refused")


def test_untrained_runtime_model_is_stable_across_launches() -> None:
    first = load_model(None)
    second = load_model(None)

    assert all(
        torch.equal(first_value, second_value)
        for first_value, second_value in zip(first.state_dict().values(), second.state_dict().values())
    )


def test_curriculum_progression_is_independent_of_total_run_length() -> None:
    assert curriculum_stage(999, 1_000) == DEFAULT_CURRICULUM[0]
    assert curriculum_stage(1_000, 1_000) == DEFAULT_CURRICULUM[1]
    assert curriculum_stage(100_000, 1_000) == DEFAULT_CURRICULUM[-1]
