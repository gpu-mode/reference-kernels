from libkernelbot import utils


def test_format_time():
    """Test time-formatting based on examples"""
    # without error
    assert utils.format_time(1) == "1.00 ns"
    assert utils.format_time("1") == "1.00 ns"
    assert utils.format_time(15.7) == "15.7 ns"
    assert utils.format_time(1012) == "1012 ns"
    assert utils.format_time(1518.4) == "1518 ns"

    assert utils.format_time(2152) == "2.15 µs"
    assert utils.format_time(51242) == "51.2 µs"
    assert utils.format_time(8_428_212) == "8.43 ms"

    # with error
    assert utils.format_time(1, "0.01") == "1.00 ± 0.010 ns"
    assert utils.format_time(52, 5.85) == "52.0 ± 5.85 ns"
    # TODO should we enforce that nonzero error never rounds to zero?
    assert utils.format_time(2152, 0.1) == "2.15 ± 0.000 µs"
    assert utils.format_time(3_754_123, 24_432) == "3.75 ± 0.024 ms"

    assert utils.format_time(None) == "–"


def test_limit_length():
    """Test utils.limit_length"""
    assert utils.limit_length("This is long text", 17) == "This is long text"
    assert utils.limit_length("This is long text", 10) == "This [...]"


def test_lru_basic_operations():
    """Test basic set, get, contains, and len operations"""
    cache = utils.LRUCache(3)

    # Test initial state
    assert len(cache) == 0
    assert "key" not in cache
    assert cache["nonexistent"] is None

    # Test set and get
    cache["key"] = "value"
    assert cache["key"] == "value"
    assert "key" in cache
    assert len(cache) == 1


def test_lru_eviction_and_ordering():
    """Test LRU eviction and queue ordering"""
    cache = utils.LRUCache(2)

    # Fill cache
    cache["a"] = 1
    cache["b"] = 2
    assert cache._q == ["a", "b"]

    # Access 'a' to move it to end
    _ = cache["a"]
    assert cache._q == ["b", "a"]

    # Add new item, should evict 'b'
    cache["c"] = 3
    assert "b" not in cache
    assert cache._q == ["a", "c"]

    # Writing also reorders
    cache["c"] = 5
    cache["a"] = 4
    assert cache._q == ["c", "a"]


def test_lru_invalidate():
    """Test invalidate clears cache and queue"""
    cache = utils.LRUCache(3)
    cache["a"] = 1
    cache["b"] = 2

    cache.invalidate()

    assert len(cache) == 0
    assert "a" not in cache
    assert cache._q == []
