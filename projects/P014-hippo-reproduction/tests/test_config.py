from p014.config import PaperTable, ScenarioName, load_named_config


def test_free_speaking_config_matches_paper_defaults() -> None:
    config = load_named_config(ScenarioName.FREE_SPEAKING)

    assert config.model.hidden_units == 24
    assert config.model.phone_encoder_blocks == 3
    assert config.model.word_encoder_blocks == 2
    assert config.model.utterance_encoder_blocks == 1
    assert config.model.word_embedding_backend == "modernbert"
    assert config.training.batch_size == 25
    assert config.training.epochs == 100
    assert config.training.trials == 5
    assert config.training.use_cono_regularizer is True
    assert config.training.use_curriculum_learning is True
    assert config.scenario.paper_table is PaperTable.TABLE_1


def test_read_aloud_config_disables_curriculum_and_cono() -> None:
    config = load_named_config(ScenarioName.READ_ALOUD)

    assert config.training.use_cono_regularizer is False
    assert config.training.use_curriculum_learning is False
    assert config.scenario.name is ScenarioName.READ_ALOUD
    assert config.scenario.paper_table is PaperTable.TABLE_2
