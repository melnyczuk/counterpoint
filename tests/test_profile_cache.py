import threading

from counterpoint.profile import Profile, ProfileCache


def _profile(x: float) -> Profile:
    return Profile(tuple([x] * 12), x, (x, x, x))


def test_concurrent_writers_keep_each_others_entries(tmp_path):
    path = tmp_path / "profiles.json"
    a, b = ProfileCache(path), ProfileCache(path)
    a.put("a:1", _profile(0.1))
    b.put("b:1", _profile(0.2))
    a.save()
    b.save()
    reloaded = ProfileCache(path)
    assert reloaded.get("a:1") == _profile(0.1)
    assert reloaded.get("b:1") == _profile(0.2)


def test_save_leaves_no_temp_files(tmp_path):
    path = tmp_path / "profiles.json"
    c = ProfileCache(path)
    c.put("k:1", _profile(0.3))
    c.save()
    assert [p.name for p in tmp_path.iterdir()] == ["profiles.json"]


def test_save_tolerates_corrupt_file(tmp_path):
    path = tmp_path / "profiles.json"
    c = ProfileCache(path)
    path.write_text("{not json")
    c.put("k:1", _profile(0.3))
    c.save()
    assert ProfileCache(path).get("k:1") == _profile(0.3)


def test_threaded_saves_do_not_crash(tmp_path):
    path = tmp_path / "profiles.json"
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            for j in range(20):
                c = ProfileCache(path)
                c.put(f"{i}:{j}", _profile(i / 10))
                c.save()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert not list(tmp_path.glob("*.tmp"))
