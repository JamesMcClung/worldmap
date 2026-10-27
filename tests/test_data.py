from worldmap.data import is_labeled, load_countries, parent_index


def test_dependencies_and_parents() -> None:
    c = load_countries()
    parents = parent_index(c)
    names = {c.NAME_EN[d]: c.NAME_EN[p] for d, p in parents.items()}
    assert len(parents) == 46
    assert names["Greenland"] == "Denmark" and names["Hong Kong"] == "People's Republic of China"
    assert names["Falkland Islands"] == "United Kingdom" and names["Aruba"] == "Netherlands"
    assert not {"Baikonur", "Guantanamo Bay Naval Base", "Palestine", "Siachen Glacier"} & names.keys()
    assert all(is_labeled(c.loc[p]) for p in parents.values())
