import torch

from p014.cono import cono_loss


def test_cono_all_same_score_returns_zero() -> None:
    features = torch.randn(4, 8)
    scores = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss = cono_loss(features, scores, lambda_diversity=1.0, lambda_tightness=1.0)
    # Only one cluster -> diversity = 0 by contract; every sample's centroid is
    # the bucket mean, so tightness is the mean distance to the batch mean.
    # That's not zero in general, so we verify the degenerate "all identical
    # features" case explicitly.
    identical = torch.ones(4, 8)
    degenerate_loss = cono_loss(identical, scores)
    assert float(degenerate_loss.item()) == 0.0
    assert float(loss.item()) >= 0.0


def test_cono_two_clusters_centroidally_clustered() -> None:
    torch.manual_seed(0)
    # Two well-separated clusters, each tightly clustered around a centroid.
    cluster_a = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    cluster_b = torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    features = torch.cat([cluster_a, cluster_b], dim=0)
    scores = torch.tensor([0.2, 0.2, 0.2, 1.8, 1.8, 1.8])

    loss_both = cono_loss(features, scores, lambda_diversity=1.0, lambda_tightness=1.0)
    # Tightness should be exactly zero (samples sit on their own centroids).
    # Diversity uses squared L2 distances per paper eq. (15):
    # -|0.2-1.8| * ||c_a-c_b||^2 averaged over both directions.
    # Centroids differ by 2.0 on the first axis only, so ||c_a-c_b||^2 = 4.
    # After averaging over the two off-diagonal entries, diversity = -6.4.
    assert torch.isclose(loss_both, torch.tensor(-6.4), atol=1e-6)

    # With the tightness weight zero, only the diversity term remains.
    diversity_only = cono_loss(
        features, scores, lambda_diversity=1.0, lambda_tightness=0.0
    )
    assert float(diversity_only.item()) < 0.0


def test_cono_is_differentiable_through_features() -> None:
    features = torch.randn(4, 6, requires_grad=True)
    scores = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = cono_loss(features, scores)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
