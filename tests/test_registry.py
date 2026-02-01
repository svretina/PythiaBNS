from pythiabns.core.registry import Registry


def test_registry_basic():
    reg = Registry()

    @reg.register("model_a", version=1)
    def model_v1():
        return "v1"

    @reg.register("model_a", version=2)
    def model_v2():
        return "v2"

    assert reg.get("model_a", version=1)() == "v1"
    assert reg.get("model_a", version=2)() == "v2"
    assert reg.get("non_existent") is None


def test_registry_filters():
    reg = Registry()

    @reg.register("m", type="fast", precision="low")
    def fast_low():
        pass

    @reg.register("m", type="fast", precision="high")
    def fast_high():
        pass

    assert reg.get("m", type="fast", precision="low") == fast_low
    assert reg.get("m", precision="high") == fast_high
    assert reg.get("m", type="slow") is None


def test_list_available():
    reg = Registry()
    reg.register("a")(lambda: None)
    reg.register("b")(lambda: None)

    available = reg.list_available()
    assert "a" in available
    assert "b" in available
    assert len(available) == 2
