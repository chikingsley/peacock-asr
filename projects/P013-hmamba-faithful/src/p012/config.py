from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
SslDims = Annotated[list[PositiveInt], Field(min_length=1)]


class HMambaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    embed_dim: PositiveInt
    gop_dim: PositiveInt | None = None
    ssl_dim: SslDims | PositiveInt | None = None
    raw_dim: PositiveInt | None = None
    kernel_size: PositiveInt = 3
    d_state: PositiveInt = 16
    d_conv: PositiveInt = 4
    expand: PositiveInt = 4
    drop: Probability = 0.0
    feat_drop: Probability = 0.0
    max_len: PositiveInt = 50
    vocab_size: PositiveInt = 81
    use_bies: bool = False
    use_cano: bool = True
    use_pos: bool = True
    use_conv: bool = False

    @model_validator(mode="after")
    def validate_feature_layout(self) -> HMambaConfig:
        if self.gop_dim is None and self.ssl_dim is None and self.raw_dim is None:
            raise ValueError("At least one of gop_dim, ssl_dim, or raw_dim must be configured.")
        if self.raw_dim is not None and self.raw_dim < 8:
            raise ValueError("raw_dim must be at least 8 because HMamba expects 1 duration + 7 energy features.")
        return self
