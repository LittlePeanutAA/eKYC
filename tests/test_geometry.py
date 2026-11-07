import numpy as np

from ekyc.geometry import CARD_SIZE, complete_quad, is_plausible_quad, order_quad, warp_quad

TL, TR, BR, BL = (10.0, 20.0), (210.0, 25.0), (205.0, 145.0), (5.0, 140.0)


def test_order_quad_shuffled():
    quad = order_quad(np.array([BR, TL, BL, TR]))
    np.testing.assert_allclose(quad, [TL, TR, BR, BL])


def test_complete_quad_infers_missing_corner():
    full = {"top_left": np.array(TL), "top_right": np.array(TR), "bottom_right": np.array(BR), "bottom_left": np.array(BL)}
    for missing in full:
        part = {k: v for k, v in full.items() if k != missing}
        quad = complete_quad(part)
        assert quad is not None
        # parallelogram completion is exact for parallelograms and close for mild perspective
        assert np.abs(quad - complete_quad(full)).max() < 15


def test_complete_quad_two_missing():
    assert complete_quad({"top_left": np.array(TL), "top_right": np.array(TR)}) is None


def test_plausible_quad():
    img_shape = (200, 250, 3)
    assert is_plausible_quad(np.array([TL, TR, BR, BL], np.float32), img_shape)
    bowtie = np.array([TL, BR, TR, BL], np.float32)
    assert not is_plausible_quad(bowtie, img_shape)
    tiny = np.array([(0, 0), (2, 0), (2, 2), (0, 2)], np.float32)
    assert not is_plausible_quad(tiny, img_shape)


def test_warp_size_and_content():
    img = np.zeros((200, 300, 3), np.uint8)
    img[50:150, 100:200] = 255
    quad = np.array([(100, 50), (199, 50), (199, 149), (100, 149)], np.float32)
    out = warp_quad(img, quad)
    assert out.shape == (CARD_SIZE[1], CARD_SIZE[0], 3)
    assert out[10:-10, 10:-10].mean() > 250
