import pytest

from pipeline_batiments.resources.sam3_resource import SAM3Resource


def test_sam3_resource_instantiation():
    res = SAM3Resource(url="http://127.0.0.1:8077", timeout_s=10)
    assert res.url == "http://127.0.0.1:8077"
    assert res.timeout_s == 10
